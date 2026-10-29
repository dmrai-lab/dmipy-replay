"""The field tier (C3) on the reader's side: the contraction of the 12-channel field basis under a main-field
direction (RPK.md 6.4.1, the one contraction of every route) and the path channel a pack stores -- the cosine
modes of the field each walker read along its own path -- decoded, gated and contracted."""


from __future__ import annotations

import numpy as np

from . import compression as _cx


CHANNEL_NAMES = ("iso_local",) + tuple(f"iso_P_{c}" for c in ("xx", "yy", "xy", "xz", "yz")) \
    + tuple(f"aniso_G_{c}" for c in ("xx", "yy", "zz", "xy", "xz", "yz"))

ISO_CHANNEL_NAMES = CHANNEL_NAMES[:6]


def q_of_H(b0_dir):
    """``Q(H) = (Hx^2, Hy^2, Hz^2, 2 Hx Hy, 2 Hx Hz, 2 Hy Hz)``: the contraction weights of the six symmetric
    components, so that ``Q . sym6(M) = H . M . H``."""
    h = np.asarray(b0_dir, float).ravel(); h = h / np.linalg.norm(h)
    return np.array([h[0] ** 2, h[1] ** 2, h[2] ** 2, 2 * h[0] * h[1], 2 * h[0] * h[2], 2 * h[1] * h[2]])


def iso_P6(channels):
    """``(..., 6)`` ``iso_P`` in the (xx, yy, zz, xy, xz, yz) order from the channels ``(..., 6 | 12)``, ``iso_P_zz`` formed
    as ``3 iso_local - iso_P_xx - iso_P_yy``."""
    c = np.asarray(channels)
    return np.concatenate([c[..., 1:3], (3.0 * c[..., 0] - c[..., 1] - c[..., 2])[..., None], c[..., 3:6]], axis=-1)


def iso_weights(b0_dir):
    """``(6,)``: the weights of ``(iso_local, iso_P_xx, iso_P_yy, iso_P_xy, iso_P_xz, iso_P_yz)`` in the isotropic
    term ``iso_local - Q . iso_P`` once ``iso_P_zz = 3 iso_local - iso_P_xx - iso_P_yy`` is formed."""
    q = q_of_H(b0_dir)
    return np.array([1.0 - 3.0 * q[2], -(q[0] - q[2]), -(q[1] - q[2]), -q[3], -q[4], -q[5]])


def field_terms(channels, b0_dir, axis=-1):
    """The two susceptibility terms of the channels along ``axis`` (``iso_local, iso_P (5), aniso_G (6)`` in that
    order, 6 or 12 of them: ``iso_P_zz`` is formed from the trace identity) for a field direction: ``(iso, aniso)``
    with ``iso = iso_local - Q . iso_P`` and ``aniso = Q . aniso_G`` (``None`` when the channels carry no
    anisotropic basis). ``dB = B0 (chi_iso iso + chi_aniso aniso)``: the one contraction of every route, grid, path
    and pack. A torch tensor is contracted on its own device in float64 (a float32 matmul on a GPU may round its inputs to TF32) and stays
    there; anything else on the host."""
    q, w = q_of_H(b0_dir), iso_weights(b0_dir)
    if type(channels).__module__.split(".")[0] == "torch":
        import torch
        c = torch.movedim(channels, axis, -1).to(torch.float64); qt = torch.as_tensor(q, device=c.device); wt = torch.as_tensor(w, device=c.device)
        return c[..., :6] @ wt, (c[..., 6:12] @ qt if c.shape[-1] >= 12 else None)
    c = np.moveaxis(np.asarray(channels, float), axis, -1)
    iso = c[..., :6] @ w
    aniso = c[..., 6:12] @ q if c.shape[-1] >= 12 else None
    return iso, aniso


def contract(channels, b0_dir, *, B0, chi_iso=0.0, chi_aniso=0.0, axis=-1):
    """``dB`` (Tesla) at each point from its channels for one configuration (:func:`field_terms`)."""
    iso, aniso = field_terms(channels, b0_dir, axis=axis)
    dB = np.zeros(iso.shape, float)
    if chi_iso:
        dB = dB + float(chi_iso) * iso
    if chi_aniso:
        if aniso is None:
            raise ValueError("these channels carry no anisotropic basis (6 of 12); chi_aniso must be 0")
        dB = dB + float(chi_aniso) * aniso
    return dB * float(B0)

CELL_IMAGES = 6


def frame_from_axis(axis, *, in_plane=None):
    """Deterministic orthonormal substrate frame R (3x3, columns [x, y, z]) with z = `axis`
    (the primary fibre direction) and a FIXED perpendicular x/y basis (Gram-Schmidt seeded
    from the global axis least aligned with z). A single fibre vector leaves a free rotation
    about itself; this pins x/y so directions are reproducible run-to-run and gradient schemes
    are oriented unambiguously. For an isotropic substrate any axis works — the frame is still
    fixed, giving uniform behaviour across packs. ``in_plane`` (a spec's ``frame.in_plane``) pins
    ``y`` instead: its component perpendicular to ``z``, the direction a secondary bundle opens
    into (RPK.md 4.2), and ``x = y × z``."""
    z = np.asarray(axis, float); z = z / np.linalg.norm(z)
    if in_plane is not None:
        y = np.asarray(in_plane, float); y = y - z * float(y @ z)
        if np.linalg.norm(y) < 1e-9:
            raise ValueError("in_plane is parallel to the axis")
        y /= np.linalg.norm(y)
        return np.column_stack([np.cross(y, z), y, z])
    seed = np.eye(3)[int(np.argmin(np.abs(z)))]        # global axis least aligned with z
    x = seed - z * float(seed @ z); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.column_stack([x, y, z])


def _refuse_trace_component(names):
    """``iso_P_zz`` is never a stored or accumulated component (RPK.md 5.3): a reader forms it from ``3 iso_local -
    iso_P_xx - iso_P_yy``. A producer that hands one over is refused by name."""
    if "iso_P_zz" in names:
        raise ValueError("the field channel carries twelve components: iso_P_zz is never stored (RPK.md 5.3), a reader forms it "
                         "as 3 iso_local - iso_P_xx - iso_P_yy; this field offers it")


def has_susc_path(arrays):
    """Whether the arrays carry a path channel, in the single-depth tensor or in a container's ranges."""
    return any(k.startswith("susc_path_dct") for k in arrays)


def susc_path_coeffs(arrays, meta, modes=None):
    """Dequantised C3 coefficients in the canonical channel order -- ``(n_w, n_ch_full, K)``, twelve channels
    (``iso_local``, five ``iso_P``, six ``aniso_G``) or the six isotropic ones: a component the pack leaves out
    (``zero_components``, an absent ``aniso_G`` block) is zero. ``iso_P_zz`` is not a channel here: the contraction
    forms it from the trace (:func:`~dmipy_replay.fields.hollow_cylinder.field_terms`), and a descriptor that lists it is
    refused.

    The ONE place that undoes the storage container. ``susc_path_dct`` may be an integer array whose
    scale lives in ``susc_path_scale`` (see :func:`susc_path_encode`), so reading it raw yields
    integers ~1e3 off with no error -- every consumer goes through here.

    ``modes`` reads the lowest ``modes`` field modes only, ``(n_w, n_ch_full, modes)``: bit for bit the leading modes of
    the full decode, and only those modes are dequantised (a prefix is held and extended as
    :func:`~dmipy_replay.replay.compression.read_position_coeffs`' bands are).

    Of a pack's ``arrays`` (:class:`~dmipy_replay.replay.decoded.PackArrays`) the result is decoded once per descriptor
    and returned read-only, with a fresh ``names`` list (:func:`~dmipy_replay.replay.compression.read_position_coeffs`)."""
    import json
    K_f = path_mode_count(arrays, meta)
    if modes is not None and not 1 <= int(modes) <= K_f:
        raise ValueError(f"modes = {modes} is outside 1 .. {K_f}, the field modes the pack stores (no modes is no read)")
    desc = json.dumps(meta, sort_keys=True, default=str)
    if modes is None or int(modes) == K_f:
        C, names = _cx._decoded_once(arrays, "susc_path_coeffs", desc, ("susc_path",),
                                     lambda: _susc_path_coeffs(arrays, meta))
        return C, list(names)
    cache = getattr(arrays, "decoded", None)
    if cache is not None:
        full = cache.peek("susc_path_coeffs", desc, {k: v for k, v in arrays.items() if k.startswith("susc_path")})
        if full is not None:
            return full[0][:, :, :int(modes)], list(full[1])
    C = _cx._decoded_prefix(arrays, "susc_path_prefix", desc, ("susc_path",), int(modes),
                            lambda lo, hi: _susc_path_coeffs(arrays, meta, lo, hi)[0], axis=2)
    return C, list(_susc_path_expanded_names(meta))


def path_mode_count(arrays, meta):
    """The field modes the path channel stores per walker and channel: its container's last mode, else the tensor's."""
    if meta.get("container") is not None:
        return int(meta["container"][-1]["modes"][1])
    return int(np.shape(arrays["susc_path_dct"])[2])


def _susc_path_names(meta):
    names = list(meta["channels"])
    _refuse_trace_component(names)
    return names


def _susc_path_expanded_names(meta):
    """The channel names of the decoded array: the canonical twelve or six when the pack stores a subset of them."""
    names = _susc_path_names(meta)
    if names and all(n in CHANNEL_NAMES for n in names):
        return list(CHANNEL_NAMES if any(n.startswith("aniso_G") for n in names) else ISO_CHANNEL_NAMES)
    return names


def _susc_path_coeffs(arrays, meta, lo=0, hi=None):
    """The decode of modes ``lo .. hi - 1`` (all of them by default) and the channel names."""
    K_f = path_mode_count(arrays, meta)
    hi = K_f if hi is None else int(hi)
    lo = int(lo)
    if meta.get("container") is not None:
        parts = []
        for i, r in enumerate(meta["container"]):
            k0, k1 = int(r["modes"][0]), int(r["modes"][1])
            a0, a1 = max(k0, lo), min(k1, hi)
            if a1 <= a0:
                continue
            t = np.swapaxes(np.asarray(arrays[_cx.range_name("susc_path_dct", i, r["bits"])]), 1, 2)
            parts.append(np.swapaxes(np.asarray(_cx._load_range(t, int(r["bits"]), a1 - a0, a0 - k0), np.float64), 1, 2))
        C = np.concatenate(parts, axis=2)
    else:
        C = np.asarray(arrays["susc_path_dct"], np.float64)[:, :, lo:hi]
    if "susc_path_scale" in arrays:
        S = np.asarray(arrays["susc_path_scale"], np.float64)[..., lo:hi]
        if S.ndim == 3:                                            # a merged pack: one scale table per shard, walkers by block
            blk = np.asarray(arrays["band_block"], np.int64)
            for b in np.unique(blk):
                C[blk == b] *= S[b][None]
        else:
            C = C * S[None]
    names = _susc_path_names(meta)
    if names and all(n in CHANNEL_NAMES for n in names):
        full = list(CHANNEL_NAMES if any(n.startswith("aniso_G") for n in names) else ISO_CHANNEL_NAMES)
        if names != full:
            out = np.zeros((C.shape[0], len(full), C.shape[2]), C.dtype)
            out[:, [full.index(n) for n in names]] = C
            C, names = out, full
    return C, names


def susc_path_field(b, b0_dir, *, B0, chi_iso, chi_aniso=0.0, has_aniso=False):
    """``dB(t)`` per walker ``(n_w, n_t)`` from the decoded path channels ``(n_w, n_ch, n_t)`` for one
    ``(B0, direction, chi)``: :func:`~dmipy_sim.fields.hollow_cylinder.contract` along the channel axis."""
    aniso = chi_aniso if (has_aniso and chi_aniso and b.shape[1] >= 12) else 0.0
    return contract(b, b0_dir, B0=B0, chi_iso=chi_iso, chi_aniso=aniso, axis=1)


def _mode_sample_matrix(K, T, n_t, start=0, stop=None, edges=None):
    """``(K, n)``: the orthonormal cosine modes of a window of ``T`` seconds read as the samples ``start .. stop - 1``
    of ``n_t`` -- sample 0 the series at ``t = 0``, sample ``j >= 1`` its mean over the block ``[(j - 1) h, j h]``,
    ``h = T / (n_t - 1)`` (the block-mean reading a raw subset's block means are compared with), or over
    ``[edges[j - 1], edges[j]]`` with ``edges`` ``(n_t,)`` the instants of a series on intervals of their own lengths
    (a walk's read instants)."""
    n_t = int(n_t); stop = n_t if stop is None else int(stop); start = int(start)
    h = float(T) / (n_t - 1)
    E = (np.arange(n_t) * h) if edges is None else np.asarray(edges, np.float64)
    j = np.arange(start, stop)
    k = np.arange(int(K))[:, None]
    om = np.pi * k / float(T)
    D = np.zeros((int(K), j.size))
    e_k0 = np.sqrt(2.0 / float(T)) * np.ones((int(K), 1)); e_k0[0] = 1.0 / np.sqrt(float(T))
    at0 = j == 0
    if at0.any():
        D[:, at0] = e_k0
    jb = j[~at0]
    if jb.size:
        a, b = E[jb - 1], E[jb]
        with np.errstate(divide="ignore", invalid="ignore"):
            m = np.sqrt(2.0 / float(T)) * (np.sin(om * b[None, :]) - np.sin(om * a[None, :])) / om / (b - a)[None, :]
        m[0] = 1.0 / np.sqrt(float(T))
        D[:, ~at0] = m
    return D


def susc_path_decode(arrays, meta, *, n_t, n_w=None, n_cut=None, start=0, edges=None):
    """The field's channels per walker on ``n_t`` samples of the window ``meta["T"]`` -- sample 0 the series at ``t =
    0``, sample ``j`` its mean over block ``j`` (:func:`_mode_sample_matrix`) -- from the stored modes, the omitted components
    zero: ``(field (n_w, n_ch_full, n_samples), names)`` in the canonical channel order. ``start`` / ``n_cut``
    read the samples ``start .. n_cut - 1`` only. A reader's sampling of the stored function (RPK.md 9.5.4), what
    the Bloch route and the dense oracles read; the scalar route contracts the modes (:func:`path_field_integral`)."""
    C, names = susc_path_coeffs(arrays, meta)
    if n_w is not None:
        C = C[:int(n_w)]
    D = _mode_sample_matrix(C.shape[2], float(meta["T"]), int(n_t), start, n_cut, edges=edges)
    return C @ D, names


def path_field_integral(arrays, meta, waveform, T, *, t0=None, n_w=None, modes=None):
    """The gated path integral of every field channel per walker, ``(Psi (n_w, n_ch), names)``: ``Psi = gamma int
    gate(t) b(t) dt = gamma sum_k g_k c_k`` with ``g_k`` the gate's cosine projection on the window ``[t0, t0 + T]``
    (:func:`~dmipy_replay.replay.compression.cosine_projection`; Parseval, RPK.md 9.5.4) and ``c_k`` the stored modes.
    The names are in the canonical order, what :func:`~dmipy_replay.fields.hollow_cylinder.field_terms` reads. ``modes``
    sums the lowest ``modes`` field modes only (the plan's, :meth:`~dmipy_replay.replay.replay.ReplayPack.plan`)."""
    from ..constants import GAMMA
    Cs, names = susc_path_coeffs(arrays, meta, modes)
    if tuple(names) not in (tuple(CHANNEL_NAMES), tuple(ISO_CHANNEL_NAMES)):
        raise ValueError(f"the path channels are stored as {names}, not in the canonical order {tuple(CHANNEL_NAMES)}")
    if n_w is not None:
        Cs = Cs[:n_w]
    gate = waveform.effective_gate if waveform.gate is None else waveform.gate
    gate = np.asarray(gate, np.float64).reshape(-1)[:-1]                       # the readout sample is held over nothing (#225)
    g = _cx.cosine_projection(gate.reshape(1, -1, 1), float(waveform.dt), float(T), Cs.shape[2], t0=t0)[0, :, 0]
    return GAMMA * np.einsum("k,wck->wc", g, Cs), names
