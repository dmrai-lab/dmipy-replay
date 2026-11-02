"""The image of a time-segment layout on JAX and on torch (:class:`JaxBackend`, :class:`TorchBackend`): the same
arithmetic as the numpy backend of :mod:`~dmipy_replay.replay.time_segments`, on another array library and device,
registered in :attr:`TimeSegmentLayout.BACKENDS <dmipy_replay.replay.time_segments.TimeSegmentLayout.BACKENDS>` and
run where ``backend="jax"`` or ``backend="torch"`` is stated (the ``jax`` and ``torch`` extras, imported when a backend
runs)."""
from __future__ import annotations

import numpy as np

from .time_segments import _decode, _phase, _row_terms

__all__ = ["JaxBackend", "TorchBackend"]


# ------------------------------------------------------------------ the kernels
class _x64:
    """JAX's 64-bit types for the duration of a JAX image (the field channels in float64, the tile sums and the
    accumulator in complex128), the flag restored after; nothing for torch."""

    def __init__(self, backend):
        self.on = backend == "jax"

    def __enter__(self):
        if self.on:
            import jax
            self.was = bool(jax.config.read("jax_enable_x64"))
            jax.config.update("jax_enable_x64", True)
        return self

    def __exit__(self, *exc):
        if self.on:
            import jax
            jax.config.update("jax_enable_x64", self.was)


def _device_field_spec(fspec, backend, device):
    """:meth:`TimeSegmentLayout._field_spec` with its scales and weights on the backend's device (float64)."""
    layout, scale, coef, n_iso = fspec
    if backend == "torch":
        import torch
        return layout, torch.as_tensor(scale, device=device), torch.as_tensor(coef, device=device), n_iso
    import jax.numpy as jnp
    return layout, jnp.asarray(scale, jnp.float64), jnp.asarray(coef, jnp.float64), n_iso










_KERNELS = {}


def _compiled(n_halves, layout, per_tile, tiers, contact, f_static, bg_layout):
    """The JAX chunk kernel: the words decoded, the phase, the tiers and the background folded in, ``w exp(i ph)`` in
    complex64 summed over each tile's rows (a fused reduction), the tile sums scattered into their voxel halves in
    complex128. One per (voxel halves, field layout, delivery, tiers, contact, field channels, background layout)."""
    import jax
    import jax.numpy as jnp
    key = (n_halves, layout, per_tile, tiers, contact, f_static, bg_layout)
    if key not in _KERNELS:
        @jax.jit
        def kernel(words, scale, w, tiles, levels, slot, pool, tw, c_scale, f_scale, f_coef, logw_pool, rho_D, a_iso, a_aniso,
                   amp, nb, b_scale, G0, acc):
            comps = _decode(words, scale, layout)
            lev = levels[slot] if per_tile else levels
            ph = _phase(comps, lev, per_tile)
            if ph is None:
                ph = jnp.zeros(words.shape[:2] + (levels.shape[-3],), jnp.float32)
            fspec = None if f_static is None else (f_static[0], f_scale, f_coef, f_static[1])
            tier = (((contact, c_scale) if contact is not None else None, fspec, logw_pool, rho_D, a_iso, a_aniso, amp)
                    if tiers else None)
            bg = (bg_layout, b_scale, G0) if bg_layout is not None else None
            ph, wf = _row_terms(ph, w, pool, tw, nb, slot, tier, bg, False)
            E = jnp.exp(1j * ph) * wf[:, :, None]
            return acc + jax.ops.segment_sum(E.sum(1).astype(jnp.complex128), tiles, num_segments=n_halves)
        _KERNELS[key] = kernel
    return _KERNELS[key]


def _sums_jax(chunks, scale, layout, levels, per_tile, tier, bg, n_halves, backend=None, device=None):
    """The per-voxel-half sums ``(n_halves, M_pad)`` complex128 on JAX: the compiled kernel over the ``chunks``
    (:meth:`TimeSegmentLayout._chunks`: the partition's words, the weights, the tiles, then the slot with a delivery, the
    pool and the tier words with the tiers, the background words with a background)."""
    import jax.numpy as jnp
    contact = None if tier is None or tier[0] is None else tier[0][0]
    fspec = None if tier is None else tier[1]
    f_static = None if fspec is None else (fspec[0], fspec[3])
    bg_layout = None if bg is None else bg[0]
    kernel = _compiled(n_halves, layout, per_tile, tier is not None, contact, f_static, bg_layout)
    M = levels.shape[-3]
    acc = jnp.zeros((n_halves, M), jnp.complex128)
    z = jnp.zeros(1, jnp.float32); zi = jnp.zeros((1, 1, 1), jnp.int32); z64 = jnp.zeros(1, jnp.float64)
    lev, sc = jnp.asarray(levels), jnp.asarray(scale)
    if tier is not None:
        ct, _, lw, rho_D, a_iso, a_aniso, amp = tier
        c_scale = jnp.asarray(ct[1]) if ct is not None else z
        f_scale, f_coef = (z64, z64) if fspec is None else (jnp.asarray(fspec[1], jnp.float64), jnp.asarray(fspec[2], jnp.float64))
        tconst = (c_scale, f_scale, f_coef, jnp.asarray(lw), jnp.float32(rho_D), jnp.float32(a_iso), jnp.float32(a_aniso), jnp.float32(amp))
    if bg is not None:
        _, b_scale, G0 = bg
        bconst = (jnp.asarray(b_scale), jnp.asarray(G0))
    for cols in chunks:
        words, w, tiles = cols[:3]; k = 3
        slot = cols[k] if per_tile else z
        k += 1 if per_tile else 0
        if tier is not None:
            pool, tw = cols[k:k + 2]; k += 2
            targs = (pool, tw) + tconst
        else:
            targs = (z, zi, z, z64, z64, z, z, z, z, z)
        bargs = (cols[k],) + bconst if bg is not None else (zi, z, z)
        acc = kernel(words, sc, w, tiles, lev, slot, *targs, *bargs, acc)
    return np.asarray(acc)


def _tile_sums_torch(words, scale, layout, w, levels, slot, pool, tw, nb, tier, bg, per_tile):
    """One chunk's tile sums on torch, ``(chunk, M_pad)`` complex128: the decoded words, the phase with the tiers and
    the background, ``w exp(i ph)`` in complex64, the reduce over each tile's rows."""
    import torch
    comps = _decode(words, scale, layout)
    lev = levels[slot.long()] if per_tile else levels
    ph = _phase(comps, lev, per_tile)
    if ph is None:
        ph = torch.zeros(words.shape[:2] + (levels.shape[-3],), dtype=torch.float32, device=words.device)
    ph, wf = _row_terms(ph, w, pool, tw, nb, slot.long() if per_tile else None, tier, bg, True)
    return torch.polar(wf[:, :, None].expand_as(ph), ph).sum(1).to(torch.complex128)


_TORCH_KERNEL = {}


def _torch_kernel():
    """The chunk function fused by ``torch.compile`` when the compiler is available (eager otherwise; measured on the
    L40S, DiSCo's layout, 184 measurements: eager 3.3 s per image, compiled 1.44 s, JAX 0.97 s)."""
    if "fn" not in _TORCH_KERNEL:
        import torch
        try:
            _TORCH_KERNEL["fn"] = torch.compile(_tile_sums_torch, dynamic=False)
        except Exception:                                     # no compiler on this platform: eager
            _TORCH_KERNEL["fn"] = _tile_sums_torch
    return _TORCH_KERNEL["fn"]


def _sums_torch(chunks, scale, layout, levels, per_tile, tier, bg, n_halves, backend=None, device=None):
    """The same sums on torch: the tile sums of every chunk (:func:`_tile_sums_torch`, compiled when possible)
    ``index_add_``-ed into a complex128 accumulator (deterministic under ``torch.use_deterministic_algorithms(True)``)."""
    import torch
    dev = torch.device(device)
    lev = torch.as_tensor(levels, device=dev); sc = torch.as_tensor(scale, device=dev)
    t = None
    if tier is not None:
        ct, fspec, lw, rho_D, a_iso, a_aniso, amp = tier
        ct = None if ct is None else (ct[0], torch.as_tensor(ct[1], device=dev))
        fspec = None if fspec is None else _device_field_spec(fspec, "torch", dev)
        t = (ct, fspec, torch.as_tensor(lw, device=dev), rho_D, a_iso, a_aniso, amp)
    b = None
    if bg is not None:
        b_layout, b_scale, G0 = bg
        b = (b_layout, torch.as_tensor(b_scale, device=dev), torch.as_tensor(G0, device=dev))
    acc = torch.zeros((n_halves, levels.shape[-3]), dtype=torch.complex128, device=dev)
    fn = _torch_kernel()
    with torch.no_grad():
        for cols in chunks:
            words, w, tiles = cols[:3]; k = 3
            slot = cols[k] if per_tile else None
            k += 1 if per_tile else 0
            pool = tw = nb = None
            if tier is not None:
                pool, tw = cols[k:k + 2]; k += 2
            if bg is not None:
                nb = cols[k]
            args = (words, sc, layout, w, lev, slot, pool, tw, nb, t, b, per_tile)
            try:
                ts = fn(*args)
            except Exception:                                  # the compiled path failed at run time: eager, once for all
                _TORCH_KERNEL["fn"] = fn = _tile_sums_torch
                ts = fn(*args)
            acc.index_add_(0, tiles, ts)
    return acc.cpu().numpy()


def _voxel_sums_torch(rows, n_q, chunks, n_halves):
    """The weighted per-voxel-half sums ``(n_q, n_halves)`` float64 on torch of the ``n_q`` per-row quantities
    ``rows(w, pool, tw, torch)`` gives for each chunk ``(w, tiles, pool, tw)`` of ``chunks``."""
    import torch
    acc = None
    with torch.no_grad():
        for w, tiles, pool, tw in chunks:
            if acc is None:
                acc = torch.zeros((n_q, n_halves), dtype=torch.float64, device=w.device)
            ts = torch.stack([(q * w).sum(1) for q in rows(w, pool, tw, torch)]).to(torch.float64)
            acc.index_add_(1, tiles, ts)
    return acc.cpu().numpy()


def _voxel_sums_jax(rows, n_q, chunks, n_halves):
    """:func:`_voxel_sums_torch` on JAX (``segment_sum`` into the voxel halves)."""
    import jax
    import jax.numpy as jnp
    acc = jnp.zeros((n_halves, n_q), jnp.float64)
    for w, tiles, pool, tw in chunks:
        ts = jnp.stack([(q * w).sum(1) for q in rows(w, pool, tw, jnp)], axis=1).astype(jnp.float64)
        acc = acc + jax.ops.segment_sum(ts, tiles, num_segments=n_halves)
    return np.asarray(acc).T



class JaxBackend:
    """The image's arithmetic on JAX (its default device): :func:`_sums_jax`, :func:`_voxel_sums_jax`, 64-bit types
    for the duration of a call (:class:`_x64`)."""
    name = "jax"
    torch = False

    def device(self, device):
        return device

    def put(self, device):
        import jax.numpy as jnp
        return lambda a, name: jnp.asarray(a)

    def context(self):
        return _x64("jax")

    def field_spec(self, fspec, device):
        return _device_field_spec(fspec, "jax", device)

    def sums(self, chunks, scale, layout, levels, per_tile, tier, bg, n_halves, device):
        return _sums_jax(chunks, scale, layout, levels, per_tile, tier, bg, n_halves, "jax", device)

    voxel_sums = staticmethod(_voxel_sums_jax)


class TorchBackend:
    """The image's arithmetic on torch (``device``; CUDA when torch sees one, else its CPU): :func:`_sums_torch`,
    :func:`_voxel_sums_torch`; tile and slot indices as int64."""
    name = "torch"
    torch = True

    def device(self, device):
        if device is None:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def put(self, device):
        import torch
        return lambda a, name: torch.as_tensor(a.astype(np.int64) if name in ("tiles", "slot") else a, device=device)

    def context(self):
        import contextlib
        return contextlib.nullcontext()

    def field_spec(self, fspec, device):
        return _device_field_spec(fspec, "torch", device)

    def sums(self, chunks, scale, layout, levels, per_tile, tier, bg, n_halves, device):
        return _sums_torch(chunks, scale, layout, levels, per_tile, tier, bg, n_halves, "torch", device)

    voxel_sums = staticmethod(_voxel_sums_torch)
