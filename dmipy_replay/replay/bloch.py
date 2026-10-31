"""The vector-Bloch propagator: every walker's magnetisation carried through the actual RF and gradient of a
sequence along its path read at the walk's block ends (:meth:`~dmipy_replay.replay.replay.ReplayPack.positions`),
the route of anything the scalar gate cannot express -- a non-nominal flip (B1+), a finite or shaped pulse, a
train's coherence pathways, an MT bound pool (RPK.md 6.5, ACQUISITION.md 7.2).
"""

from typing import NamedTuple

import numpy as np

from ..constants import GAMMA

from ..acquisition.rf import RFSchedule

from ._replay_kernel import piece_phase_weights

def _rf_increment(M, flip, ax, xp=np):
    """``M`` (3, N) rotated by ``flip`` rad about the in-plane B1 axis ``ax`` rad (Rodrigues, axis
    ``(cos ax, sin ax, 0)``): one partial B1 step of a finite pulse, the free precession between increments
    supplying the off-resonance tilt. ``flip`` is a scalar (uniform B1) or per-walker ``(N,)`` (B1+ transmit
    inhomogeneity); ``flip = 0`` is the identity. ``xp`` is numpy or jax.numpy: the one RF rotation of the
    package, on host arrays and on traced operands alike."""
    ux, uy = xp.cos(ax), xp.sin(ax)
    c, s = xp.cos(flip), xp.sin(flip)
    omc = 1.0 - c
    Mx, My, Mz = M[0], M[1], M[2]
    return xp.stack([(c + ux * ux * omc) * Mx + (ux * uy * omc) * My + (uy * s) * Mz,
                     (ux * uy * omc) * Mx + (c + uy * uy * omc) * My + (-ux * s) * Mz,
                     (-uy * s) * Mx + (ux * s) * My + c * Mz])

def finite_180_longitudinal_dwell(phi_pre, tau_180, phi_B1=0.0):
    r"""Per-walker longitudinal dwell during a finite 180° refocusing pulse.

    A 180° rotates M about the B1 axis.  The component ALONG B1 stays transverse the
    whole pulse; only the PERPENDICULAR component swings transverse → longitudinal →
    transverse, spending time at z where T1 (not T2 / surface / susceptibility) acts.
    For a walker arriving with transverse azimuth ``phi_pre`` (its accumulated
    gradient + susceptibility phase relative to the static-spin reference), the
    longitudinal dwell is

        tau_par = sin²(phi_pre − phi_B1) · tau_180 / 2 .

    Meiboom-Gill (``phi_B1 = 0``) keeps a coherent on-resonance spin off z
    (tau_par = 0), while a uniformly dephased ensemble (<sin²> = 1/2) gives
    tau_par → tau_180 / 4 — the per-walker, tissue-blind replacement for the scalar
    ``|cos|`` / ``cos²`` profiles.

    Parameters
    ----------
    phi_pre : array-like   per-walker accumulated transverse phase at the pulse (rad).
    tau_180 : float        refocusing pulse duration (s).
    phi_B1 : float         azimuth of the B1 axis vs the static reference (rad); 0 = MG.

    Returns
    -------
    np.ndarray   per-walker longitudinal dwell ``tau_par`` (s).  Move this much dwell
        from the T2 to the T1 channel.
    """
    return np.sin(np.asarray(phi_pre, dtype=np.float64) - phi_B1) ** 2 * (0.5 * tau_180)

_BLOCH_PHASE_TABLE_BYTES = 512 * 2 ** 20      # device budget for one batch of per-piece phase tables

class _BlochTerms(NamedTuple):
    """The operator of a vector-Bloch replay per PIECE of the walk -- the block grid cut at every RF instant -- in
    the order the propagator applies them: the RF rotations that open the piece, free precession over it,
    transverse wall attenuation, relaxation."""
    k: np.ndarray               # (n_p,) block each piece lies in
    w_lo: np.ndarray            # (n_meas, n_p, 3) precession weight on r[k] (gamma included)
    w_hi: np.ndarray            # (n_meas, n_p, 3) precession weight on r[k + 1]
    flips: np.ndarray           # (n_p, n_ev) RF rotation opening each piece, per event slot, rad (0 = none)
    axes: np.ndarray            # (n_p, n_ev) B1 phase of each slot, rad
    b1: object                  # transmit scale: float, or (n_w,) per walker
    dphi_extra: object          # (n_p, n_w) precession beyond the gradient over the piece, or None
    surf: object                # (n_p, n_w) transverse wall factor over the piece, or None
    E2: np.ndarray              # (n_p, n_w) or (n_p, 1) transverse relaxation factor over the piece
    E1: np.ndarray              # (n_p, n_w) or (n_p, 1) longitudinal relaxation factor over the piece
    weights: object             # (n_w,) normalised ensemble weights, or None (plain mean)
    echo_pieces: object         # tuple of piece indices after which the echoes are read, or None

def _bloch_timeline(rf_events, n_t, h, t0=None):
    """Cut the walk at the block ends and at every RF instant. Returns the edges (sorted, ``(n_p + 1,)``), the
    rotations opening each piece ``[(piece, flip, axis), ...]`` and the pulse windows
    ``[(t0, t1, carrier rad/s), ...]`` over which a carrier offset and slice-select off-resonance act.

    With ``t0`` the block ends are those of the WINDOW of the walk starting at ``t0`` on the schedule's clock (RPK.md
    4.3), every time on that clock: an instant belongs to the window ``(t0, t0 + T]`` (and ``t = 0`` to the first),
    a finite pulse's sub-rotations each to theirs, its carrier window clipped to the walk's window; nothing is
    refused, since what lies outside is another window's (the whole schedule is checked against the whole walk
    once, without ``t0``)."""
    T = (int(n_t) - 1) * float(h)
    tol = 1e-9 * float(h)
    start = 0.0 if t0 is None else float(t0)
    end = start + T
    ends = start + np.arange(int(n_t)) * float(h)
    inside = (lambda t: (t > start + tol or (start == 0.0 and t >= -tol)) and t <= end + tol)
    rots, windows, instants = [], [], []
    for e in RFSchedule(rf_events):
        t_s, dur = e.t_s, e.duration_s
        if t0 is None and (t_s < -tol or t_s > T + tol):
            raise ValueError(f"RF event at t = {t_s:.6g} s lies outside the walk [0, {T:.6g}] s")
        nsub = max(1, int(round(dur / float(h)))) if dur > 0.0 else 1
        ts = (np.clip(t_s - dur / 2.0 + (np.arange(nsub) + 0.5) * dur / nsub, 0.0, None) if dur > 0.0
              else np.array([t_s]))
        dflips, axes = e.flip_split(nsub)                                  # even, or by the pulse's envelope
        for t, f, ax in zip(ts, dflips, axes):
            t = min(float(t), end) if t0 is None else float(t)
            if inside(t):
                rots.append((t, float(f), float(ax)))
                instants.append(t)
        windows.append((max(start, t_s - dur / 2.0), min(end, t_s + dur / 2.0), 2.0 * np.pi * e.offset_hz))   # one per event; empty outside the window
    times = np.sort(np.concatenate([ends, np.asarray(instants, np.float64)]))
    edges = [times[0]]
    for t in times[1:]:
        if t - edges[-1] > tol:
            edges.append(t)
    edges = np.asarray(edges)
    if rots and any(abs(t - edges[-1]) <= tol for t, _, _ in rots):
        edges = np.append(edges, edges[-1])                                  # a rotation at T opens an empty piece
    def piece_of(t):
        i = int(np.argmin(np.abs(edges[:-1] - t)))
        return i
    rotations = [(piece_of(t), f, ax) for t, f, ax in rots]
    return edges, rotations, windows

def _bloch_replay_terms(trajectory, h, G, dt_wf, rf_events, *, T2, T1, comp_traj,
                        T2_per_comp, T1_per_comp, extra_phase_per_step,
                        dlog_boundary_unit, surface_relaxivity, D, b1_scale, slice_offsets,
                        slice_gradient, bound_frac, T2_bound, T1_bound, off_resonance_bound,
                        weights, echo_steps=None, t0=None):
    """Resolve the sequence and substrate inputs of :func:`replay_bloch` into per-piece terms.

    Both Bloch replays consume this; the numpy loop and the JAX scan then differ only in how they iterate the
    identical operator, which is what makes the numpy one the reference.

    The walk is cut at the block ends and at every RF instant (:func:`_bloch_timeline`). Over each piece the gradient
    precession is exact for the piecewise-linear path (:func:`_replay_kernel.piece_phase_weights`), relaxation
    runs for the piece's duration, and the per-block-end wall contact and off-resonance are shared by duration. A
    hard pulse is one rotation at its instant; a finite pulse of ``duration_s`` is ``round(duration/dt)``
    sub-rotations spread over its window with the flip split evenly (or by its envelope); its carrier
    ``offset_hz`` and, with ``slice_offsets``/``slice_gradient``, the slice-select off-resonance act over the
    window (a hard pulse has no window). An MT ``bound_frac`` blends the relaxation rates toward the bound
    pool's per block end and adds the bound pool's off-resonance precession by occupancy.
    """
    G = np.asarray(G, np.float64)
    n_w, n_t, _ = trajectory.shape
    dt = float(h)
    edges, rotations, windows = _bloch_timeline(rf_events, n_t, dt, t0=t0)
    w_lo, w_hi, k = piece_phase_weights(G, dt_wf, edges, n_t, dt, t0=t0)       # (n_meas, n_p, 3), (n_p,)
    n_p = k.size
    start = 0.0 if t0 is None else float(t0)
    ell = np.diff(edges)                                                      # (n_p,) piece durations
    frac = ell / dt

    def rate(per_comp, scalar):
        if per_comp is not None:
            if comp_traj is None:
                raise ValueError("comp_traj is required with per-compartment relaxation.")
            return (1.0 / np.asarray(per_comp, np.float64))[np.asarray(comp_traj)].T    # (n_t, n_w)
        return np.full((n_t, 1), 0.0 if scalar is None else 1.0 / float(scalar))
    invT2 = rate(T2_per_comp, T2)
    invT1 = rate(T1_per_comp, T1)
    # a block end's occupancy, contact and bound fraction are accumulated over the step ENDING at it, so the piece in
    # block [t_k, t_{k+1}] reads them at k + 1; a sampled quantity (a field at the block ends) is read through
    # the path interpolant with the piece's unit moments (lo on block end k, hi on block end k + 1)
    k1 = np.minimum(k + 1, n_t - 1)
    a_rel, b_rel = edges[:-1] - start - k * dt, edges[1:] - start - k * dt
    u_hi = (b_rel ** 2 - a_rel ** 2) / (2.0 * dt)                            # int (t - t_k) dt / dt
    u_lo = ell - u_hi

    dphi_extra = None
    def add(term):
        nonlocal dphi_extra
        dphi_extra = term if dphi_extra is None else dphi_extra + term

    if bound_frac is not None:
        if T2_bound is None or T1_bound is None:
            raise ValueError("T2_bound and T1_bound are required when bound_frac is set.")
        bf = np.asarray(bound_frac, np.float64).T                          # (n_t, n_w)
        invT2 = (1.0 - bf) * invT2 + bf * (1.0 / float(T2_bound))
        invT1 = (1.0 - bf) * invT1 + bf * (1.0 / float(T1_bound))
        if float(off_resonance_bound) != 0.0:
            add(bf[k1] * (2.0 * np.pi * float(off_resonance_bound) * ell)[:, None])
    E2 = np.exp(-ell[:, None] * invT2[k1])                                    # (n_p, n_w | 1)
    E1 = np.exp(-ell[:, None] * invT1[k1])

    if extra_phase_per_step is not None:
        x = np.asarray(extra_phase_per_step, np.float64).T / dt                  # (n_t, n_w): the sampled rate
        add(u_lo[:, None] * x[k] + u_hi[:, None] * x[k1])

    surf = None
    if (surface_relaxivity is not None and dlog_boundary_unit is not None
            and float(surface_relaxivity) != 0.0):
        if D is None:
            raise ValueError("D required when surface_relaxivity is not None.")
        surf = np.exp((float(surface_relaxivity) / float(D)) * np.asarray(dlog_boundary_unit, np.float64).T[k1]
                      * frac[:, None])

    slots = {}
    for pi, f, ax in rotations:
        slots.setdefault(pi, []).append((f, ax))
    n_ev = max([len(v) for v in slots.values()] + [1])
    flips = np.zeros((n_p, n_ev)); axes = np.zeros((n_p, n_ev))
    for pi, lst in slots.items():
        for j, (f, a) in enumerate(lst):
            flips[pi, j], axes[pi, j] = f, a
    if windows:
        overlap = np.zeros(n_p); carrier = np.zeros(n_p)
        for t0, t1, w_off in windows:
            o = np.clip(np.minimum(edges[1:], t1) - np.maximum(edges[:-1], t0), 0.0, None)
            overlap += o; carrier += w_off * o
        if np.any(carrier != 0.0):
            add(carrier[:, None])
        if slice_offsets is not None and float(slice_gradient) != 0.0:
            add(overlap[:, None] * (GAMMA * float(slice_gradient) * np.asarray(slice_offsets, np.float64))[None, :])

    b1 = 1.0 if b1_scale is None else np.asarray(b1_scale, np.float64)
    wn = None
    if weights is not None:
        wn = np.asarray(weights, np.float64)
        wn = wn / wn.sum()
    echo_pieces = None
    if echo_steps is not None:
        ends = edges[1:]
        echo_pieces = tuple(int(np.argmin(np.abs(ends - (start + int(e) * dt)))) for e in echo_steps)
    return _BlochTerms(k, w_lo, w_hi, flips, axes, b1, dphi_extra, surf, E2, E1, wn, echo_pieces)

def _bloch_replay_output(sig_last, M_last, echo_out, echo_steps, return_walker_signals):
    if echo_steps is not None:
        return echo_out
    if return_walker_signals:
        return M_last, sig_last
    return sig_last

def replay_bloch(trajectory, h, G, dt_wf, rf_events, *,
                 T2=None, T1=None, comp_traj=None,
                 T2_per_comp=None, T1_per_comp=None,
                 extra_phase_per_step=None,
                 dlog_boundary_unit=None, surface_relaxivity=None, D=None,
                 echo_steps=None, echo_per_walker=False,
                 weights=None, return_walker_signals=False,
                 b1_scale=None, slice_offsets=None, slice_gradient=0.0,
                 bound_frac=None, T2_bound=None, T1_bound=None,
                 off_resonance_bound=0.0, t0=None, M_init=None, return_state=False):
    """Emergent per-walker vector-Bloch replay on the stored (field-independent) walk.

    Propagates each walker's magnetisation ``M = (Mx, My, Mz)`` through the ACTUAL sequence operators -- RF
    rotations from ``rf_events``, the physical gradient precession ``gamma int G(t) . r_w(t) dt``, per-comp
    T2/T1, surface relaxivity, an optional susceptibility off-resonance field, and an optional MT bound-pool
    blend -- piece by piece, the walk cut at every block end and every RF instant, so the phase accrued before a
    pulse is flipped by it exactly wherever the pulse falls (:func:`_bloch_replay_terms`). The forward-engine
    counterpart is :func:`dmipy_replay.engine.bloch.simulate_bloch`.

    This is the numpy REFERENCE: a Python loop over measurements and pieces written so each operator is one
    readable line.

    Coherence pathways / refocusing are EMERGENT: the 180 conjugates the accumulated phase, so the spin echo
    forms by itself. **Pass the PHYSICAL (same-sign-lobe) gradient** (``ScannerSequence.G``, on its own grid).

    Parameters
    ----------
    rf_events : RFSchedule or sequence of RFEvent
        The pulses (:class:`~dmipy_replay.acquisition.rf.RFEvent`, fields ``t_s``, ``flip_deg``, ``axis_deg``,
        ``duration_s``, ``offset_hz``); ``axis_deg`` is the B1 phase (0 = x, 90 = y); ``duration_s = 0`` is an
        instantaneous hard pulse at ``t_s`` (any instant, not a block end); ``offset_hz`` an off-resonance carrier
        over a finite pulse.
    T2, T1 / T2_per_comp, T1_per_comp, comp_traj : as in :func:`replay`.
    extra_phase_per_step : ``(n_w, n_t)`` the off-resonance phase each walker accrues per block end (a field read
        along the walk, a uniform offset, a crusher's winding), integrated with the path interpolant.
    bound_frac, T2_bound, T1_bound, off_resonance_bound : the MT bound-pool blend per block end.
    b1_scale : transmit (B1+) scale of every flip; per walker for an inhomogeneous field.
    slice_offsets, slice_gradient : slice-select off-resonance ``gamma G_slice z_w`` during finite pulses.
    weights : (n_w,) ensemble weights of the walker mean.
    echo_steps, echo_per_walker : record ``Mxy`` at these SAVE indices, per walker if asked.
    t0, M_init, return_state : ``trajectory`` as ONE WINDOW of a longer walk (RPK.md 4.3): its first block end sits
        at ``t0`` on the sequence's clock, every walker starts the window at ``M_init`` ``(n_meas, 3, n_w)`` (the
        state the previous window left it in; equilibrium when None), and with ``return_state`` the state every
        measurement leaves every walker in at the window's end, ``(n_meas, 3, n_w)``, is returned before the
        signals, for the next window to start from.

    Returns
    -------
    signals : (n_meas,) complex walker-mean ``Mx + i My`` at the end, or ``(n_meas, n_echo)`` (``(n_meas,
        n_echo, n_w)`` with ``echo_per_walker``) when ``echo_steps`` is given, or ``(M_final, signals)`` with the
        per-walker (3, n_w) final magnetisation of the last measurement when ``return_walker_signals``; with
        ``return_state``, ``(M_all, <the above>)``.
    """
    tm = _bloch_replay_terms(trajectory, h, G, dt_wf, rf_events, T2=T2, T1=T1,
                             comp_traj=comp_traj, T2_per_comp=T2_per_comp, T1_per_comp=T1_per_comp,
                             extra_phase_per_step=extra_phase_per_step,
                             dlog_boundary_unit=dlog_boundary_unit,
                             surface_relaxivity=surface_relaxivity, D=D, b1_scale=b1_scale,
                             slice_offsets=slice_offsets, slice_gradient=slice_gradient,
                             bound_frac=bound_frac, T2_bound=T2_bound, T1_bound=T1_bound,
                             off_resonance_bound=off_resonance_bound, weights=weights, echo_steps=echo_steps, t0=t0)
    traj = np.asarray(trajectory, np.float64)
    n_meas = tm.w_lo.shape[0]
    n_w = traj.shape[0]
    n_p = tm.k.size
    wmean = (lambda v: np.mean(v)) if tm.weights is None else (lambda v: (v * tm.weights).sum())
    echo_set = None if tm.echo_pieces is None else set(tm.echo_pieces)
    signals = np.empty(n_meas, np.complex128)
    echo_out = None
    M = None
    M_all = np.empty((n_meas, 3, n_w))
    for m in range(n_meas):
        if M_init is None:
            M = np.zeros((3, n_w))
            M[2] = 1.0                                          # equilibrium along +z
        else:
            M = np.array(np.asarray(M_init, np.float64)[m], copy=True)
        rec = []
        for p in range(n_p):
            for f, a in zip(tm.flips[p], tm.axes[p]):
                if f != 0.0:
                    M = _rf_increment(M, f * tm.b1, a)
            kp = tm.k[p]
            dphi = traj[:, kp, :] @ tm.w_lo[m, p] + traj[:, kp + 1, :] @ tm.w_hi[m, p]     # (n_w,)
            if tm.dphi_extra is not None:
                dphi = dphi + tm.dphi_extra[p]
            c, s_ = np.cos(dphi), np.sin(dphi)
            Mx = c * M[0] - s_ * M[1]
            My = s_ * M[0] + c * M[1]
            if tm.surf is not None:
                Mx, My = Mx * tm.surf[p], My * tm.surf[p]
            M = np.stack([Mx * tm.E2[p], My * tm.E2[p], M[2] * tm.E1[p]])
            if echo_set is not None and p in echo_set:
                mxy = M[0] + 1j * M[1]
                for _ in range(tm.echo_pieces.count(p)):
                    rec.append(mxy.copy() if echo_per_walker else wmean(mxy))
        signals[m] = wmean(M[0] + 1j * M[1])
        M_all[m] = M
        if echo_set is not None:
            if echo_out is None:
                echo_out = np.empty((n_meas, len(rec)) + ((n_w,) if echo_per_walker else ()), np.complex128)
            echo_out[m] = np.asarray(rec)
    out = _bloch_replay_output(signals, M, echo_out, echo_steps, return_walker_signals)
    return (M_all, out) if return_state else out


def _build_crusher(crusher, dt, n_t):
    """Per-step crusher phase rate (rad/step) for a per-walker u in [0,1).

    ``crusher`` = ``{'windows_s': [(t0,t1), ...], 'n_cycles': float | [float, ...]}`` or None.  Over a
    window of ``n_win`` steps the rate is ``2 pi n_cycles / n_win``, so a walker at
    macroscopic coordinate ``u`` accrues ``2 pi n_cycles u`` across the window and the
    ensemble (u ~ U[0,1)) dephases over ``n_cycles`` turns -- a DECLARED voxel-scale spoiler, for a winding the
    waveform does not play in ``G``. A spoiler that is played in ``G`` is not declared here: its winding follows
    from the waveform and the prescription (``ScannerSequence.voxel_factor``, dmipy-sim#375).

    ``n_cycles`` is stated, one number for every window or one per window, and windows ADD where they overlap:
    a spoiler and an unbalanced readout wind different amounts over the same interval and cannot share a
    figure.
    """
    rate = np.zeros(n_t, dtype=np.float64)
    if crusher is None:
        return rate, False
    windows = list(crusher.get('windows_s', []))
    if 'n_cycles' not in crusher:
        raise ValueError("a declared crusher states its winding: crusher = {'windows_s': [...], 'n_cycles': ...}. "
                         "There is no default number of turns; a spoiler played in G needs no declaration, since its "
                         "winding follows from the waveform and the prescription (dmipy-sim#375)")
    cyc = crusher['n_cycles']
    cyc = [float(cyc)] * len(windows) if np.ndim(cyc) == 0 else [float(c) for c in cyc]
    if len(cyc) != len(windows):
        raise ValueError(f"n_cycles is one number or one per window ({len(windows)}); got {len(cyc)}")
    for (t0, t1), n_cycles in zip(windows, cyc):
        i0, i1 = int(round(t0 / dt)), int(round(t1 / dt))
        i0, i1 = max(0, i0), min(n_t, i1)
        if i1 > i0:
            rate[i0:i1] += 2.0 * np.pi * n_cycles / (i1 - i0)
    return rate, True
