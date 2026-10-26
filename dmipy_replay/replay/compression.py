"""The codecs a replay pack stores its channels in, and the readers of each.

Every channel of a running integral is stored as bands on ONE basis, the continuous one of the storage window
(RPK.md 9.5), accumulated by the walk while it runs (:mod:`dmipy_replay.engine.accumulate`); nothing is encoded from
samples on a grid:

  * positions (C0) -- ``bridge_dst``, the one representation: per axis the path's two exact endpoints and the lowest
    ``K`` sine bands of the residual pinned at both, ``beta_k = int u(t) s_k(t) dt`` with ``s_k = sqrt(2 / T) sin(pi
    k t / T)``; a deliverable gradient is band-limited, so the replay is exact for every acquisition inside the band.
    Any other codec name is refused generically (:func:`require_position_method`). The readers:
    :func:`band_projection` (the waveform side, exact per sample), :func:`bridge_at` / :func:`positions_at` (the
    stored function at instants a reader names), the free band variances (:func:`band_variances`), the block table the walk
    accumulates against (:func:`block_table`, :func:`cosine_band_scale`) and its quadrature (:func:`quadrature_rel`).
  * boundary local time (C2, rho2/D = 1) -- the cumulative local time ``B(t)`` in the same bridge form as the
    positions (:func:`encode_boundary_bands`): its two endpoints exact, its residual as sine bands; the gated
    contact read on the bridge itself (:func:`surface_logweight_bridge`), ``B(t)`` at instants by
    :func:`local_time_at`.
  * the field (C3) -- the cosine modes ``c_k = int e_k b dt`` of the field read along the path
    (:func:`cosine_mode_scale`, :func:`cosine_projection` the gate's side; the tensors in :mod:`dmipy_replay.replay.bank`).
  * compartment occupancy (C1) -- named columns, piecewise constant per walker, stored EXACTLY IN TIME: one static
    label per walker, or the events ``(t, label)`` of every change (:func:`encode_events_occupancy`, RPK.md 9.5.7);
    relaxation is the sum over the runs between events of the rate times the gate's on-time, no grid
    (:func:`relaxation_logweight_runs`). The MT bound pool is one of these columns, not a channel of its own.

The integer band containers (:func:`quantise_bands`, :func:`derive_container`) are a choice of bytes, not of meaning.
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)

import re



import numpy as np


from ..constants import GAMMA

POSITION_METHOD = "bridge_dst"          # the only C0 representation: start, displacement and sine bands (RPK.md 9.5.3)

ALL_METHODS = (POSITION_METHOD,)

def require_position_method(method):
    """Raise unless ``method`` is the one C0 representation this build reads.

    There is deliberately no fallback: any other codec stores different quantities under the same
    tensor names -- ``bridge_dst`` puts the two endpoints where a band codec puts its two lowest
    bands -- so decoding one as the other yields plausible numbers rather than an error. Refusing
    is the only safe response, by name, for whatever codec a pack declares; the pack must be
    re-encoded from its master with ``build_replay_pack()``.
    """
    if method == POSITION_METHOD:
        return method
    if method is None:
        raise ValueError(
            f"pack declares no position codec. It must declare "
            f"compression.method = {POSITION_METHOD!r}; refusing to assume it, since a pack "
            f"written by an older build stores different quantities under the same names.")
    raise ValueError(
        f"position codec {method!r} is not read by this build. Positions are stored as "
        f"{POSITION_METHOD!r}: two exact endpoints per axis followed by sine bands of the pinned "
        f"residual. Re-encode the pack from its master with build_replay_pack().")

_F16, _F32 = 2, 4

def cosine_band_scale(K, T):
    """Per band ``k = 1..K`` the factor taking the cosine projection of a path's increments, ``c_k = int cos(pi k t /
    T) dx(t)``, to the bridge's band on the continuous basis, ``beta_k = sqrt(2 T) / (pi k) c_k``, ``(K,)`` float64.

    The bridge of a path is ``u(t) = x(t) - x(0) - (t / T) (x(T) - x(0))``, pinned at both ends, so by parts
    ``int u s_k dt = -int S_k du`` with ``S_k = -sqrt(2 / T) (T / pi k) cos(pi k t / T)`` an antiderivative of ``s_k``,
    and ``du = dx - (x(T) - x(0)) dt / T`` adds nothing because the cosine integrates to zero over the window (the
    block table's midpoint sum of it is zero too, ``sum_b cos(pi k (2b + 1) / 2N) = 0``)."""
    k = np.arange(1, int(K) + 1, dtype=np.float64)
    return np.sqrt(2.0 * float(T)) / (np.pi * k)

def cosine_mode_scale(K, T):
    """Per mode ``k = 0..K`` the factor taking the accumulated projection ``sum cos(pi k t / T) b dt`` to the mode on the
    orthonormal cosine basis of the window, ``c_k = int e_k b dt`` with ``e_0 = T^-1/2`` and ``e_k = sqrt(2 / T)
    cos(pi k t / T)`` (RPK.md 9.5.3): ``(K + 1,)`` float64, ``[1 / sqrt(T), sqrt(2 / T), ...]``."""
    out = np.full(int(K) + 1, np.sqrt(2.0 / float(T)))
    out[0] = 1.0 / np.sqrt(float(T))
    return out

def cosine_projection(G, dt_wf, T, K, *, t0=None):
    """THE projection of a sample-and-hold course on a window's orthonormal cosine modes, ``(n_meas, K, n_c)``: ``g_k =
    int G(t) e_k(t - t0) dt`` over the window ``[t0, t0 + T]``, ``k = 0..K-1``, piece by piece in closed form (``int_a^b
    e_k dt = sqrt(2 / T) (sin(omega b) - sin(omega a)) / omega``, ``(b - a) / sqrt(T)`` at ``k = 0``). The field phase
    of a walk stored as the modes ``c_k`` of ``b(t)`` is then ``gamma sum_k g_k c_k`` (Parseval, RPK.md 9.5.4), the
    cosine twin of :func:`band_projection`."""
    T = float(T)
    Gl, a, b, _ = _window_pieces(G, dt_wf, T, t0)
    n_meas, _, n_c = Gl.shape
    out = np.zeros((n_meas, int(K), n_c))
    if int(K) == 0 or a.size == 0:
        return out
    out[:, 0, :] = np.einsum("mnc,n->mc", Gl, b - a) / np.sqrt(T)
    if int(K) > 1:
        k = np.arange(1, int(K))
        om = np.pi * k / T
        W = np.sqrt(2.0 / T) * (np.sin(np.outer(b, om)) - np.sin(np.outer(a, om))) / om[None, :]        # (n_pieces, K-1)
        out[:, 1:, :] = np.einsum("mnc,nk->mkc", Gl, W)
    return out

def band_variances(K, T, D):
    """The free bridge's band variances on the continuous basis, ``(K,)``: the eigenvalues of the bridge covariance
    ``2 D (min(s, t) - s t / T)`` per axis, ``2 D T^2 / (pi k)^2`` -- the limit as ``dt -> 0`` of the discrete
    sine transform's ``D dt / (2 sin^2(pi k / 2N))`` on ``N`` intervals of ``dt``, read in the continuous
    normalisation."""
    k = np.arange(1, int(K) + 1, dtype=np.float64)
    return 2.0 * float(D) * float(T) ** 2 / (np.pi * k) ** 2

_PROJECTION_CHUNK = 1 << 25

def _window_pieces(G, dt_wf, T, t0):
    """The sample-and-hold pieces of ``G`` ``(n_meas, n_wf, n_c)`` within the window ``[t0, t0 + T]``: ``(G_live,
    a, b, live)`` with the pieces' ends on the window's clock, clipped to it, the pieces outside dropped (``live`` the
    samples kept). Without ``t0`` a non-zero sample starting beyond ``T`` is refused (the stored path ends there)."""
    G = np.asarray(G, np.float64)
    e = np.arange(G.shape[1] + 1) * float(dt_wf)
    start = 0.0 if t0 is None else float(t0)
    if t0 is None:
        beyond = e[:-1] > float(T) * (1.0 + 1e-9)
        if beyond.any() and np.any(G[:, beyond, :] != 0.0):
            raise ValueError(f"the waveform has a non-zero gradient beyond the pack's T_max = {float(T):.6g} s "
                             f"(its own extent is {e[-1]:.6g} s); the stored path ends there")
    a = np.clip(e[:-1] - start, 0.0, float(T)); b = np.clip(e[1:] - start, 0.0, float(T))
    live = b > a
    return G[:, live, :], a[live], b[live], live

def _piece_weights(a, b, T, k):
    """``(n_pieces, len(k))``: the integral of ``s_k(t)`` over each piece ``[a, b]`` of the window, ``sqrt(2 / T) (cos(omega
    a) - cos(omega b)) / omega``, written ``2 sin(omega (a + b) / 2) sin(omega (b - a) / 2) / omega`` (no cancellation)."""
    om = np.pi * np.asarray(k, np.float64) / float(T)
    mid, half = 0.5 * (a + b), 0.5 * (b - a)
    return np.sqrt(2.0 / float(T)) * 2.0 * np.sin(np.outer(mid, om)) * np.sin(np.outer(half, om)) / om[None, :]

def band_projection(G, dt_wf, T, K, *, t0=None):
    """THE projection of a waveform on a pack's bands, ``(n_meas, K + 2, n_c)``: ``M0 = int G``, ``M1 =
    int G (t - t0) / T`` and ``W_k = int G(t) s_k(t - t0) dt`` over the window ``[t0, t0 + T]``, exactly for the
    sample-and-hold waveform ``G`` ``(n_meas, n_wf, n_c)`` on its own grid ``dt_wf`` (each sample held over its
    interval, as :func:`~dmipy_replay.replay._replay_kernel.effective_gradient` reads it), piece by piece
    (:func:`_piece_weights`). The gradient phase of a walk stored as ``C = [x(0), x(T) - x(0), beta_1..beta_K]`` is
    ``gamma sum C W``; :func:`band_projection_matrix` is the same map as a matrix on the samples."""
    T = float(T)
    Gl, a, b, _ = _window_pieces(G, dt_wf, T, t0)
    n_meas, _, n_c = Gl.shape
    out = np.zeros((n_meas, int(K) + 2, n_c))
    out[:, 0, :] = np.einsum("mnc,n->mc", Gl, b - a)
    out[:, 1, :] = np.einsum("mnc,n->mc", Gl, (b * b - a * a) / (2.0 * T))
    if int(K) == 0 or a.size == 0:
        return out
    step = max(1, int(_PROJECTION_CHUNK // max(8 * a.size, 1)))
    for k0 in range(1, int(K) + 1, step):
        k = np.arange(k0, min(k0 + step, int(K) + 1))
        out[:, 1 + k0:1 + k0 + k.size, :] = np.einsum("mnc,nk->mkc", Gl, _piece_weights(a, b, T, k))
    return out

def segment_projection(segments, T, K, *, t0=None):
    """THE projection of time segments on a pack's bands, ``(n_seg, K + 2)``: per segment ``[a, b]`` (s, on the walk's
    clock) its indicator's ``M0``, ``M1`` and ``W_k`` over the window ``[t0, t0 + T]`` (the segment clipped to it), in
    the closed form :func:`band_projection` sums piece by piece (:func:`_piece_weights`) -- what that projection gives a
    waveform equal to 1 on the segment and 0 elsewhere, wherever the segment's edges fall. A walk stored as ``C =
    [x(0), x(T) - x(0), beta_1..beta_K]`` has ``int_a^b x dt = sum C W`` per axis."""
    T = float(T)
    start = 0.0 if t0 is None else float(t0)
    S = np.asarray(segments, np.float64).reshape(-1, 2)
    a = np.clip(S[:, 0] - start, 0.0, T); b = np.clip(S[:, 1] - start, 0.0, T)
    b = np.maximum(a, b)
    out = np.zeros((S.shape[0], int(K) + 2))
    out[:, 0] = b - a
    out[:, 1] = (b * b - a * a) / (2.0 * T)
    if int(K):
        out[:, 2:] = _piece_weights(a, b, T, np.arange(1, int(K) + 1))
    return out

def band_projection_matrix(n_wf, dt_wf, T, K, *, t0=None):
    """``(K + 2, n_wf)``: :func:`band_projection` as a matrix on a waveform's ``n_wf`` samples (one axis) -- column ``n``
    the projection of the unit sample ``n`` held over its interval, zero for a sample outside the window."""
    T = float(T)
    e = np.arange(int(n_wf) + 1) * float(dt_wf)
    start = 0.0 if t0 is None else float(t0)
    a = np.clip(e[:-1] - start, 0.0, T); b = np.clip(e[1:] - start, 0.0, T)
    out = np.zeros((int(K) + 2, int(n_wf)))
    out[0] = b - a
    out[1] = (b * b - a * a) / (2.0 * T)
    if int(K):
        out[2:] = _piece_weights(a, b, T, np.arange(1, int(K) + 1)).T
    return out

def bridge_phase_variance(G, dt_wf, T, D, *, t0=None):
    """``(n_meas,)``: the variance of ``int G . u dt`` over the window ``[t0, t0 + T]`` for the free bridge ``u`` of a
    walk at diffusivity ``D`` (``gamma^2`` not applied), every band at once -- what the continuous bands carry
    together, so that the phase the bands beyond ``K`` would carry is this less the retained bands' sum. With
    ``q(r) = int_r^T G`` and the bridge covariance ``2 D (min(s, t) - s t / T)`` per axis it is ``2 D (int q^2 - (int
    q)^2 / T)``, ``q`` piecewise linear for a sample-and-hold ``G``, so both integrals are exact."""
    from ._replay_kernel import _cumulative_at
    T = float(T)
    start = 0.0 if t0 is None else float(t0)
    G = np.asarray(G, np.float64)
    e = np.arange(G.shape[1] + 1) * float(dt_wf) - start
    p = np.unique(np.concatenate([[0.0, T], e[(e > 0.0) & (e < T)]]))
    Q0, _ = _cumulative_at(G, dt_wf, start + p)                                  # (n_meas, n_p, n_c)
    q = Q0[:, -1:, :] - Q0                                                        # int_r^T G at the breakpoints
    L = np.diff(p)[None, :, None]
    qa, qb = q[:, :-1, :], q[:, 1:, :]
    int_q2 = (L * (qa * qa + qa * qb + qb * qb) / 3.0).sum(axis=1)
    int_q = (L * (qa + qb) / 2.0).sum(axis=1)
    return 2.0 * float(D) * (int_q2 - int_q * int_q / T).sum(axis=1)

def band_phase_variances(G, dt_wf, T, K, D, *, t0=None):
    """``(per_band (n_meas, K), total (n_meas,))``: the phase variance (rad^2) each of the lowest ``K`` bands carries for
    the waveform ``G`` on a free walk at ``D`` over the window ``T`` -- ``gamma^2 sum_c W_kc^2 var_k`` with
    :func:`band_projection` and :func:`band_variances` -- and what every band of the bridge carries together, the
    closed form :func:`bridge_phase_variance`. The phase the bands beyond ``j`` would carry is ``total -
    per_band[:, :j].sum(1)``, which a dropped independent phase costs the signal half of: the truncation term of a
    pack's band, walk-free."""
    W = band_projection(G, dt_wf, T, int(K), t0=t0)[:, 2:, :]
    pb = GAMMA ** 2 * (W ** 2).sum(axis=2) * band_variances(int(K), T, D)[None, :]
    return pb, GAMMA ** 2 * bridge_phase_variance(G, dt_wf, T, D, t0=t0)

def rescale_bands(arrays, factor):
    """The arrays with every bridge band (positions and contact, any container, any window) multiplied by
    ``factor`` and the exact endpoints untouched: a float container's band columns, an integer container's scale
    table. The field channel's modes ``c_k = int e_k b dt`` carry the same ``sqrt(2 / T)`` and are scaled alike: the
    float tensor ``susc_path_dct``, or the integer ranges' ``susc_path_scale``. What a continuous-basis pack read on a
    grid divided by ``a`` takes (``factor = a^(-1/2)``: the basis's normalisation ``sqrt(2 / T)`` follows ``T / a``)."""
    out = {}
    pos = set(POSITION_AXES)
    for key, v in arrays.items():
        name = key.rsplit("/", 1)[-1]
        if v is None:
            out[key] = v
        elif name in pos:
            a = np.array(v, np.float64, copy=True); a[:, 2:] *= float(factor); out[key] = a
        elif name in ("pos_band_scale", "blt_band_scale", "blt_bridge_dst"):
            out[key] = np.asarray(v, np.float64) * float(factor)
        elif name == "susc_path_scale":
            out[key] = np.asarray(v, np.float64) * float(factor)
        elif name == "susc_path_dct" and np.asarray(v).dtype.kind == "f":      # the float container: no scale table
            out[key] = (np.asarray(v, np.float64) * float(factor)).astype(np.asarray(v).dtype)
        else:
            out[key] = v
    return out

NATIVE_BITS = (8, 16)

MIN_BITS = 3

def _band_ranges(container, K):
    """``[(k0, k1, bits), ...]`` covering bands ``0 .. K-1`` from a container: ``((upto, bits), ...)`` with ``upto``
    None the end, or the ``[{"bands": [k0, k1], "bits": b}, ...]`` a pack's metadata records and :func:`band_curve`
    returns (the path channel's ranges name ``"modes"`` for ``"bands"``). Refuses a gap, an overlap, or a depth outside ``MIN_BITS .. 16``."""
    out, k0 = [], 0
    for item in container:
        if isinstance(item, dict):
            upto, bits = int(item["bands"][1] if "bands" in item else item["modes"][1]), int(item["bits"])
        else:
            upto, bits = item
        k1 = int(K) if upto is None else min(int(upto), int(K))
        if not MIN_BITS <= int(bits) <= 16:
            raise ValueError(f"band container bits must be {MIN_BITS}..16 (got {bits})")
        if k1 > k0:
            out.append((k0, k1, int(bits)))
        k0 = max(k0, k1)
    if k0 < int(K):
        raise ValueError(f"band container covers bands up to {k0} of {K}; end it with (None, bits)")
    return out

def range_name(key, i, bits):
    """The tensor name of band range ``i`` at ``bits``: ``{key}_b{i}`` for a native depth, ``{key}_b{i}u{bits}`` for a
    bit-packed one -- the depth is part of the name so a reader without the metadata can unpack."""
    return f"{key}_b{i}" if int(bits) in NATIVE_BITS else f"{key}_b{i}u{int(bits)}"

_RANGE_RE = r"_b(\d+)(?:u(\d+))?"

_PACK_ROWS = 1 << 15

def unpack_bits(packed, bits, m):
    """The ``m`` unsigned integers of ``bits`` each per row of ``packed`` ``(N, nbytes)``: the inverse of
    :func:`pack_bits`, ``(N, m)`` ``uint32``."""
    packed = np.asarray(packed)
    w = (1 << np.arange(bits - 1, -1, -1)).astype(np.uint32)
    out = np.empty((packed.shape[0], m), np.uint32)
    for i in range(0, packed.shape[0], _PACK_ROWS):
        b = np.unpackbits(packed[i:i + _PACK_ROWS], axis=1, count=m * bits).reshape(-1, m, bits)
        out[i:i + _PACK_ROWS] = (b.astype(np.uint32) * w).sum(axis=2, dtype=np.uint32)
    return out

def _load_range(t, bits, m):
    """The integer band values ``(N, m[, n_axes])`` of a stored tensor: :func:`_store_range` inverted."""
    t = np.asarray(t)
    if bits in NATIVE_BITS:
        return t
    off = 1 << (bits - 1)
    if t.ndim == 2:
        return unpack_bits(t, bits, m).astype(np.int32) - off
    return np.stack([unpack_bits(t[..., a], bits, m).astype(np.int32) - off for a in range(t.shape[2])], axis=2)

def band_scales(arrays, scale_key, n_w):
    """The per-walker scale of every band, ``(N_w, K)`` or ``(N_w, n_axes, K)``: the block table indexed by each
    walker's block (``band_block``, absent: one block)."""
    table = np.asarray(arrays[scale_key], np.float64)
    if "band_block" in arrays:
        return table[np.asarray(arrays["band_block"], np.int64)]
    if table.shape[0] != 1:
        raise KeyError(f"{scale_key} holds {table.shape[0]} blocks but the pack carries no band_block")
    return np.broadcast_to(table[0], (int(n_w),) + table.shape[1:])

def dequantise_bands(arrays, container, key, scale_key, dtype=np.float64):
    """The bands back as ``(N_w, K)`` or ``(N_w, K, n_axes)`` from :func:`quantise_bands`' tensors."""
    r0 = container[0]
    t0 = np.asarray(arrays[range_name(key, 0, r0["bits"])])
    scale = band_scales(arrays, scale_key, t0.shape[0])               # (N_w, K) or (N_w, n_axes, K)
    scale = np.swapaxes(scale, 1, 2) if scale.ndim == 3 else scale    # -> (N_w, K[, n_axes])
    parts = []
    for i, r in enumerate(container):
        k0, k1 = r["bands"]
        q = _load_range(arrays[range_name(key, i, r["bits"])], int(r["bits"]), int(k1 - k0))
        parts.append(np.asarray(q, np.float64) * scale[:, k0:k1])
    return np.concatenate(parts, axis=1).astype(dtype)

DEVICES = ("auto", "numpy", "jax")

CHUNK_BYTES = 1 << 30

def bridge_basis(times, K, T):
    """The bridge's basis at the instants ``times`` (s from the window's start) of a window of ``T`` seconds:
    ``(S (K, n), tau (n,))``, ``S[k, n] = s_k(t_n) = sqrt(2 / T) sin(pi k t_n / T)`` and the chord's fraction ``tau =
    t / T``; ``s_k`` is zero at both ends of the window, where the bridge is pinned."""
    t = np.asarray(times, np.float64).reshape(-1)
    tau = t / float(T)
    k = np.arange(1, int(K) + 1, dtype=np.float64)
    return np.sqrt(2.0 / float(T)) * np.sin(np.pi * np.outer(k, tau)), tau

def bridge_at(C, times, T):
    """The function a stored bridge vector holds, at the instants ``times`` (s from the window's start): ``C`` is
    ``(n_w, K + 2, ...)``, the start, the displacement and the ``K`` bands of each walker (per axis for the positions,
    one channel for the contact), and the value is the chord plus the bands' series, ``(n_w, n, ...)`` float64."""
    C = np.asarray(C, np.float64)
    S, tau = bridge_basis(times, C.shape[1] - 2, T)
    tail = (1,) * (C.ndim - 2)
    out = C[:, None, 0] + C[:, None, 1] * tau.reshape((1, -1) + tail)
    if C.shape[1] > 2:
        out = out + np.einsum("wk...,kn->wn...", C[:, 2:], S)
    return out

def positions_at(arrays, meta, times, walkers=None):
    """Every walker's position at the instants ``times`` (s from the window's start) of a window of ``meta["T"]``
    seconds, ``(n_w, n, 3)`` float64, from the stored endpoints and bands (of the slice ``walkers`` when given)."""
    C = read_position_coeffs(arrays, dtype=np.float64)
    return bridge_at(C if walkers is None else C[walkers], times, float(meta["T"]))

def local_time_at(arrays, meta, times, walkers=None):
    """The cumulative boundary local time ``B(t)`` (at ``rho2 / D = 1``, non-positive) of every walker at the instants
    ``times`` (s from the window's start), ``(n_w, n)`` float64, from the contact channel's exact endpoints and its
    bands (:func:`bridge_bands`; of the slice ``walkers`` when given). The contact over an interval is the
    difference of two of these."""
    sl = slice(None) if walkers is None else walkers
    a = np.asarray(arrays["blt_start"], np.float64)[sl]
    e = np.asarray(arrays["blt_endpoint"], np.float64)[sl]
    vec = np.concatenate([a[:, None], (e - a)[:, None], bridge_bands(arrays, meta)[sl]], axis=1)
    return bridge_at(vec, times, float(meta["T"]))

def has_c2(arrays):
    """Whether the arrays carry the C2 bridge in either container."""
    return "blt_bridge_dst" in arrays or any(k.startswith("blt_b0") for k in arrays)

def c2_bands_K(arrays, meta):
    """The number of C2 bands stored, from the metadata or the tensors."""
    if meta and meta.get("K") is not None:
        return int(meta["K"])
    if "blt_bridge_dst" in arrays:
        return int(np.asarray(arrays["blt_bridge_dst"]).shape[1])
    return int(np.asarray(arrays["blt_band_scale"]).shape[-1])

def bridge_bands(arrays, meta, key="blt", scale_key="blt_band_scale", dtype=np.float64):
    """The C2 bands ``(N_w, K)`` from either container: the float tensor ``blt_bridge_dst`` or the integer ranges
    ``blt_b<i>`` with ``blt_band_scale`` (``meta["container"]``)."""
    if "blt_bridge_dst" in arrays:
        return np.asarray(arrays["blt_bridge_dst"], dtype)
    if meta.get("container") is None:
        raise KeyError("this pack's C2 channel is in the integer container but its metadata carries no 'container'")
    return dequantise_bands(arrays, meta["container"], key, scale_key, dtype)

_EXCLUSIVE_COLUMN = "comp"          # the geometric axis: mutually exclusive, an int label

OCCUPANCY_COLUMNS = ("comp", "bound")

def occupancy_events(arrays, column):
    """A column's stored events as ``(start (N_w,), EventList)``: the label every walker holds at its first event
    (the window's start) and its changes after it (:class:`~dmipy_replay.coefficient_walk.EventList`); a static column
    as ``(labels, None)``."""
    from .events import EventList
    name = column["name"]
    if column["kind"] == "static":
        return np.asarray(arrays[f"{name}_static"]).astype(np.int64), None
    if column["kind"] != "events":
        raise ValueError(f"C1 column {name!r} is stored as {column['kind']!r}, which this build does not read: a column is "
                         "'static' or 'events' (RPK.md 9.5.7)")
    off = np.asarray(arrays[f"{name}_events_offsets"], np.int64)
    t = np.asarray(arrays[f"{name}_events_t"], np.float64)
    lab = np.asarray(arrays[f"{name}_events_label"]).astype(np.int64)
    if np.any(np.diff(off) < 1):
        raise ValueError(f"C1 column {name!r}: every walker's list opens with its label at the window's start; a walker has none")
    heads = off[:-1]
    mask = np.ones(t.shape[0], bool); mask[heads] = False
    counts = np.diff(off) - 1
    return lab[heads], EventList(np.concatenate([[0], np.cumsum(counts)]), t[mask], lab[mask])

def decode_occupancy(arrays, meta):
    """-> ``{name: (start, events)}`` per column (:func:`occupancy_events`): the labels at the window's start and the
    changes after it; a static column's events are ``None``. A pack whose C1 metadata predates the column layout is
    refused by name."""
    if "columns" not in meta:
        raise ValueError(
            "this pack's C1 metadata predates the occupancy-column layout (no 'columns' key); the pack is read by no "
            "revision of this reader. A pack is never converted: re-walk it with a conformant producer.")
    return {d["name"]: occupancy_events(arrays, d) for d in meta["columns"]}

def occupancy_at(arrays, column, times):
    """The labels of every walker at each of ``times`` (s, on the walk's clock), ``(N_w, len(times))``: a column
    evaluated on a reader's instants (the dense oracle, the Bloch route's per-block reading) -- never a stored form."""
    from .events import events_at_instants
    start, ev = occupancy_events(arrays, column)
    return events_at_instants(ev, start, times)

def is_current_c1(meta):
    """True if a pack's C1 channel metadata is in the occupancy-column layout."""
    return isinstance(meta, dict) and "columns" in meta

def has_c1(arrays):
    """Whether the arrays carry the C1 ``comp`` column in either form (events or the static label)."""
    return "comp_events_offsets" in arrays or "comp_static" in arrays

POSITION_AXES = ("pos_x", "pos_y", "pos_z")

def position_ranges(arrays, name):
    """``[(i, bits), ...]`` the band ranges stored under ``{name}_b<i>[u<bits>]``, in band order: the depth of a
    native tensor from its dtype, of a bit-packed one from its name."""
    out = []
    for k in arrays:
        m = re.fullmatch(re.escape(name) + _RANGE_RE, k)
        if m:
            out.append((int(m.group(1)), int(m.group(2)) if m.group(2) else 8 * np.asarray(arrays[k]).dtype.itemsize))
    return sorted(out)

def position_container(arrays):
    """The container the position tensors are in: ``"float"`` (``pos_x`` ...) or ``"bands"`` (``pos_x_ends`` ...)."""
    if POSITION_AXES[0] in arrays:
        return "float"
    if f"{POSITION_AXES[0]}_ends" in arrays:
        return "bands"
    return None

def has_axis_layout(arrays):
    return position_container(arrays) is not None

def read_position_coeffs(arrays, axes=None, dtype=np.float64):
    """Coefficients as (n_w, K, n_axes) from either layout.

    ``axes`` selects spatial components by index (default all present). Reading a subset is the point of
    the layout: pass e.g. ``axes=(0,)`` for a slab or ``(0, 1)`` for a cylinder's transverse plane.
    """
    kind = position_container(arrays)
    if kind is not None:
        suffix = "" if kind == "float" else "_ends"
        present = [i for i, k in enumerate(POSITION_AXES) if f"{k}{suffix}" in arrays]
        want = list(present if axes is None else axes)
        missing = [i for i in want if i not in present]
        if missing:
            raise KeyError(f"pack does not carry position axes {missing} "
                           f"(has {[POSITION_AXES[i] for i in present]}); it was written with a reduced "
                           f"axis set and cannot serve this encoding")
        if kind == "float":
            return np.stack([np.asarray(arrays[POSITION_AXES[i]], dtype) for i in want], axis=2)
        n_w = int(np.asarray(arrays[f"{POSITION_AXES[present[0]]}_ends"]).shape[0])
        scale = band_scales(arrays, "pos_band_scale", n_w)                          # (N_w, n_axes, K)
        ranges = position_ranges(arrays, POSITION_AXES[present[0]])
        K_b = int(scale.shape[2])
        cols = []
        for i in want:
            ends = np.asarray(arrays[f"{POSITION_AXES[i]}_ends"], np.float64)
            bands, k0 = [], 0
            for r, bits in ranges:
                t = arrays[range_name(POSITION_AXES[i], r, bits)]
                m = int(t.shape[1]) if bits in NATIVE_BITS else min(int(t.shape[1]) * 8 // bits, K_b - k0)
                q = np.asarray(_load_range(t, bits, m), np.float64)
                bands.append(q * scale[:, i, k0:k0 + m]); k0 += m
            cols.append(np.concatenate([ends] + bands, axis=1))
        return np.stack(cols, axis=2).astype(dtype)
    # No legacy (n_w, K, 3) fallback by design. A dataset with two position layouts forces every
    # consumer to carry a compatibility branch, and that branch is where silent errors live -- a reader
    # that guesses the wrong convention returns plausible numbers. Fail loudly instead; convert the pack.
    raise KeyError(
        f"pack has no position axes {POSITION_AXES}; found {sorted(arrays)}. Packs written before the "
        f"axis-per-tensor layout store a single (n_walkers, K, 3) 'dct_coeffs'; re-encode them with "
        f"pack_position_arrays(). This reader does not accept mixed layouts -- a dataset with two "
        f"position layouts forces every consumer to carry a compatibility branch, and that branch is "
        f"where silent errors live.")

def _rates(times):
    """Per-pool rates ``1 / T`` of a list of times by pool id: ``inf`` is no decay; a time that is not positive is
    refused, since a zero would be an infinite rate and an identically dark signal, not "no decay"."""
    T = np.asarray(times, np.float64).reshape(-1)
    if not (T > 0).all():
        raise ValueError(f"a relaxation time by pool must be positive, float('inf') for no decay; got {T.tolist()}")
    return 1.0 / T

class CumulativeGate:
    """A sample-and-hold gate on its own grid as its running integral: ``gate(t) = int_0^t chi(t') dt'`` for any ``t``
    (s), piecewise linear between the grid's knots, exact for the held samples, and flat beyond the gate's extent
    (its last sample is the readout, held over nothing, RPK.md 6.6) and before ``t = 0``. What a run between two
    events is weighed by: ``gate(t1) - gate(t0)`` is the gate's on-time over the run, with no grid of the pack in
    it."""
    __slots__ = ("knots", "C", "T")

    def __init__(self, chi, dt_wf):
        chi = np.asarray(chi, np.float64).reshape(-1)
        if chi.size < 2:
            raise ValueError(f"a gate needs at least two samples (the last is the readout, held over nothing); got {chi.size}")
        self.knots = np.arange(chi.size) * float(dt_wf)
        self.C = np.concatenate([[0.0], np.cumsum(chi[:-1]) * float(dt_wf)])
        self.T = float(self.knots[-1])

    def __call__(self, t):
        return np.interp(np.asarray(t, np.float64), self.knots, self.C, left=0.0, right=float(self.C[-1]))

    @property
    def total(self):
        return float(self.C[-1])

def relaxation_logweight_runs(arrays, column, T2_per_comp, T1_per_comp, chi, active, *, t0=0.0, t1):
    """The per-walker relaxation log-weight over the window ``[t0, t1]`` from a C1 column's stored form, exact in time
    (RPK.md 9.5.7): a static column is one run per walker, an events column a run per entry from its instant to the
    next entry's (the last to ``t1``), and the log-weight is ``-sum_runs (r2(label) [chi(t_b) - chi(t_a)] +
    r1(label) [lon(t_b) - lon(t_a)])`` with ``chi`` the transverse gate's running integral (:class:`CumulativeGate`)
    and ``lon = active - chi`` the longitudinal periods' (``active`` the acquisition's own extent: a stored period of
    a stimulated echo relaxes at T1, the walk beyond the echo relaxes at nothing). O(events), one run per walker
    when nothing crosses; no grid, no ``dt``."""
    name = column["name"]
    invT2, invT1 = _rates(T2_per_comp), _rates(T1_per_comp)
    lon = lambda t: active(t) - chi(t)
    if column["kind"] == "static":
        lab = np.asarray(arrays[f"{name}_static"]).astype(np.int64)
        on = float(chi(t1) - chi(t0)); off = float(lon(t1) - lon(t0))
        return -(invT2[lab] * on + invT1[lab] * off)
    if column["kind"] != "events":
        raise ValueError(f"C1 column {name!r} is stored as {column['kind']!r}, which this build does not read")
    off_ = np.asarray(arrays[f"{name}_events_offsets"], np.int64)
    t = np.asarray(arrays[f"{name}_events_t"], np.float64)
    lab = np.asarray(arrays[f"{name}_events_label"]).astype(np.int64)
    n_w = off_.shape[0] - 1
    w = np.repeat(np.arange(n_w), np.diff(off_))
    t_a = np.maximum(t, float(t0))
    t_b = np.empty_like(t); t_b[:-1] = t[1:]; t_b[-1] = t1
    t_b[off_[1:] - 1] = float(t1)
    t_b = np.minimum(t_b, float(t1))
    per = invT2[lab] * (chi(t_b) - chi(t_a)) + invT1[lab] * (lon(t_b) - lon(t_a))
    return -np.bincount(w, weights=per, minlength=n_w)

def surface_logweight_bridge(arrays, meta, rho2_over_D, gate=None, dt_gate=None, *, t0=0.0):
    """The gated surface log-weight ``(rho2 / D) int chi(t) dB(t)`` over the window ``[t0, t0 + T]`` from the contact
    channel's continuous bridge ``B(t) = a + (e - a)(t - t0) / T + sum_k beta_k s_k(t - t0)``: the gate ``chi`` held
    sample by sample on its own grid ``dt_gate`` (its last sample the readout, held over nothing), so

        int chi dB = (e - a) <chi>_T + sum_k beta_k sqrt(2 / T) omega_k int chi cos(omega_k (t - t0)) dt,

    each piece's cosine integral in closed form, no series decoded. Without a gate the exact total ``(rho2 / D) (e -
    a)``."""
    a = np.asarray(arrays["blt_start"], np.float64); e = np.asarray(arrays["blt_endpoint"], np.float64)
    if gate is None:
        return float(rho2_over_D) * (e - a)
    T = float(meta["T"])
    chi = np.asarray(gate, np.float64).reshape(-1)
    if chi.size < 2:
        raise ValueError(f"a gate needs at least two samples (the last is the readout, held over nothing); got {chi.size}")
    Gc, lo, hi, _ = _window_pieces(chi[None, :-1, None], dt_gate, T, float(t0))
    g = Gc[0, :, 0]
    C = bridge_bands(arrays, meta); K = C.shape[1]
    om = np.pi * np.arange(1, K + 1) / T
    mid, half = 0.5 * (lo + hi), 0.5 * (hi - lo)
    Wc = np.sqrt(2.0 / T) * (g[:, None] * 2.0 * np.cos(np.outer(mid, om)) * np.sin(np.outer(half, om))).sum(axis=0)
    return float(rho2_over_D) * ((e - a) * float((g * (hi - lo)).sum()) / T + C @ Wc)
