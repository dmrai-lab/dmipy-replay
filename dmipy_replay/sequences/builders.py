"""The sequence builders: one thin call per family, one mechanics behind them (:mod:`.assemble`).

Every builder takes the measurement axis -- ``gradient_directions`` and either ``bvalues`` (the exact b to
realise) or ``gradient_strengths`` (the amplitude to play) -- the family's shape parameters in the literature's
own vocabulary, an optional ``TE`` (the grid runs from t = 0 to TE; the smallest that fits when omitted),
``n_t``, the ``slew_rate`` limit and an optional timing budget, and returns a validated
:class:`~dmipy_replay.acquisition.scanner_sequence.ScannerSequence` carrying its ``Encoding`` and ``build_spec``::

    seq = pgse([[1, 0, 0]], delta=0.01, Delta=0.03, bvalues=[1e9])          # exact b
    seq = pgse([[1, 0, 0]], delta=0.01, Delta=0.03, gradient_strengths=0.08)  # the amplitude, b follows
    seq.G, seq.G_eff, seq.rf, seq.encoding.bvalues, seq.b(), seq.btensor()

The families: :func:`pgse`, :func:`pgste` (the 3-pulse stimulated echo), :func:`fexi` (filter-exchange imaging: a
bipolar filter, the magnetisation stored over a mixing time, a bipolar detection; :func:`fexi_protocol` its standard
protocol, :func:`fexi_bvalues` the filter and detection b-values read off a built sequence), :func:`ogse` (cosine or
trapezoidal trains), :func:`cpmg` (a refocusing train at constant or alternating polarity), :func:`gre`, :func:`ste` and
:func:`pte` (b-tensor encodings), and the readers of a played gradient :func:`from_waveform`,
:func:`from_btensor_waveform`, :func:`from_pgste_waveform`. :func:`instantaneous` is a sequence's square limit,
:func:`to_gradient_array` the square PGSE an analytical layer integrates.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from ..acquisition.rf import RFEvent, RFSchedule, role_of
from ..acquisition.scanner_sequence import REFOCUS_ATOL, Encoding, Protocol, ScannerSequence
from ..constants import DEFAULT_SLEW_RATE
from ..math.gradient_conversions import g_from_b, q_from_b
from ..acquisition.waveforms import b_from_gradient
from ._helpers import _resolve_te, unify_length_reference_delta_Delta
from .assemble import (PreparedEchoTrain, EchoTrain, GradientEcho, SpinEcho, StimulatedEcho, assemble, axis_pairs, bipolar, cosine,
                       ramp_of, trapezoid, trapezoid_train, _B_ITER, _B_RTOL, _first_step_at, _steps_before)

__all__ = ["pgse", "pgste", "fexi", "fexi_protocol", "fexi_bvalues", "gre", "cpmg", "ogse", "ste", "pte", "from_waveform",
           "from_btensor_waveform", "from_pgste_waveform", "instantaneous", "to_gradient_array"]

_INT_TOL = 1e-6


def _rows(gradient_directions, *arrays, default_direction=(0.0, 0.0, 1.0), n=None):
    """The measurement axis: ``(n_m, 3)`` directions (a default axis when none are given) and the per-row
    broadcast of every array in ``arrays``."""
    if gradient_directions is None:
        if n is None:
            n = max([1] + [np.size(a) for a in arrays if a is not None])
        dirs = np.tile(np.asarray(default_direction, np.float64), (n, 1))
    else:
        dirs = np.asarray(gradient_directions, dtype=np.float64)
        if dirs.ndim == 1:
            dirs = dirs[None]
    n_m = dirs.shape[0]
    out = [None if a is None else np.broadcast_to(np.asarray(a, np.float64), (n_m,)).copy() for a in arrays]
    return dirs, n_m, out


def _need_amplitude(family, bvalues, gradient_strengths):
    if bvalues is None and gradient_strengths is None:
        raise ValueError(f"{family}: give bvalues= (the b to realise) or gradient_strengths= (the amplitude to play)")


def _whole(x, what):
    """``x`` as the integer it must be, or a refusal naming the nearest valid values."""
    n = np.rint(x)
    bad = np.abs(x - n) > _INT_TOL * np.maximum(1.0, np.abs(x))
    if np.any(bad):
        raise ValueError(f"{what} must be whole, got {np.asarray(x)[bad].tolist()}: choose the frequency or the "
                         f"duration so the block holds a whole number")
    return n.astype(int)


# ── spin echoes ──────────────────────────────────────────────────────────────────────────────────────────────────

def pgse(gradient_directions, delta, Delta, *, bvalues=None, gradient_strengths=None, TE=None, n_t=1000,
         slew_rate=DEFAULT_SLEW_RATE, timing=None):
    """PGSE: two same-sign lobes ``Delta`` apart (centre to centre), the 180 midway between them.

    ``delta`` is the lobe's half-amplitude width, its ramps ``g / slew_rate`` on top (vertical at ``np.inf``, the
    square limit -- same structure). Every row's pair is centred on the one 180 at ``TE/2``, so a row with a
    shorter ``Delta`` sits inside the same echo with the pulse in its own gap; ``TE`` longer than the smallest
    that fits adds dead time symmetrically. With a ``timing`` budget the pulses are finite, the lobes keep out
    of the lead-in, the refocusing window and the readout tail, and a gap narrower than the 180 is refused.
    """
    _need_amplitude("pgse", bvalues, gradient_strengths)
    dirs, n_m, (delta_, Delta_) = _rows(gradient_directions, delta, Delta)
    eps = lambda m, g: ramp_of(g, slew_rate)
    return assemble(
        SpinEcho(gap=lambda m, g: Delta_[m] - delta_[m] - eps(m, g), timing=timing),
        gradient_directions=dirs, bvalues=bvalues, gradient_strengths=gradient_strengths, TE=TE, n_t=n_t,
        timing=timing, family='pgse', q_width=delta_,
        span=lambda m, g: delta_[m] + eps(m, g),
        sample=lambda m, g, dt: trapezoid(delta_[m], eps(m, g), dt),
        encoding=lambda g, te, te_min: dict(delta=delta_, Delta=Delta_, ramp_time=np.array([eps(m, g[m]) for m in range(n_m)])),
        build_spec=('pgse', dict(gradient_directions=gradient_directions, delta=delta, Delta=Delta, bvalues=bvalues,
                                 gradient_strengths=gradient_strengths, TE=TE, n_t=n_t, slew_rate=slew_rate,
                                 timing=timing)))


def ogse(gradient_directions, oscillation_frequency, gradient_duration, *, shape="trapezoid", Delta=None, bvalues=None,
         gradient_strengths=None, TE=None, n_t=1000, slew_rate=DEFAULT_SLEW_RATE, timing=None):
    """OGSE: an oscillating block of ``gradient_duration`` on each side of the 180, the same block twice (the 180
    folds the second one, so ``G_eff`` continues the oscillation and ``q(TE) = 0`` exactly).

    ``shape='trapezoid'`` is the train of alternating trapezoid lobes of Drobnjak et al. (2016) -- a lobe per
    half period, ``2 f sigma`` of them (whole; one lobe is a PGSE lobe), ramps ``g / slew_rate`` at every edge;
    ``shape='cosine'`` is the frequency-selective ``cos(2 pi f t)`` over ``f sigma`` whole periods (DC-free q),
    its own slope ``2 pi f g`` kept under the slew limit and its edges ramped into. The two blocks start ``Delta``
    apart (Drobnjak's block separation; the gap between them, ``Delta - gradient_duration``, must hold the
    refocusing window) or, with no ``Delta``, sit against the refocusing window (the budget's, or none); ``TE``
    beyond the minimum adds dead time symmetrically.
    """
    _need_amplitude("ogse", bvalues, gradient_strengths)
    if shape not in ("trapezoid", "cosine"):
        raise ValueError(f"ogse shape must be 'trapezoid' or 'cosine', got {shape!r}")
    dirs, n_m, (f_, sigma_) = _rows(gradient_directions, oscillation_frequency, gradient_duration)
    # a row's block is shaped by its frequency and duration; a row without one (a b = 0 measurement given no
    # frequency) plays nothing, a b = 0 row WITH one plays its block at zero amplitude (it is that shell's b = 0)
    amp = np.broadcast_to(np.asarray(bvalues if bvalues is not None else gradient_strengths, np.float64), (n_m,))
    shaped = (f_ > 0) & (sigma_ > 0)
    if np.any((amp > 0) & ~shaped):
        raise ValueError("ogse needs a positive oscillation_frequency and gradient_duration on every weighted row")
    eps = lambda m, g: ramp_of(g, slew_rate)
    if shape == "trapezoid":
        n_lobes = np.ones(n_m, dtype=int)
        n_lobes[shaped] = _whole(2.0 * f_[shaped] * sigma_[shaped],
                                 "the number of lobes 2 f sigma of a trapezoidal OGSE block")
        lobe = np.where(shaped, sigma_ / np.maximum(n_lobes, 1), 0.0)

        def sample(m, g, dt):
            return trapezoid_train(n_lobes[m], lobe[m], eps(m, g), dt) if shaped[m] else np.zeros(1)
    else:
        n_cyc = np.ones(n_m, dtype=int)
        n_cyc[shaped] = _whole(f_[shaped] * sigma_[shaped], "the number of periods f sigma of a cosine OGSE block")

        def sample(m, g, dt):
            if not shaped[m]:
                return np.zeros(1)
            slew = float(slew_rate)
            if np.isfinite(slew) and 2.0 * np.pi * f_[m] * g > slew * (1.0 + 1e-9):
                raise ValueError(f"a cosine at {f_[m]:.1f} Hz and {g:.4f} T/m slews at {2*np.pi*f_[m]*g:.1f} T/m/s, "
                                 f"above the {slew:.1f} T/m/s limit: lower the amplitude or use shape='trapezoid'")
            return cosine(n_cyc[m], f_[m], eps(m, g), dt)
    if Delta is None:
        gap_ = np.full(n_m, 0.0 if timing is None else float(timing.t_refocus))
    else:
        gap_ = np.broadcast_to(np.asarray(Delta, np.float64), (n_m,)) - sigma_
    return assemble(
        SpinEcho(gap=lambda m, g: gap_[m], timing=timing),
        gradient_directions=dirs, bvalues=bvalues, gradient_strengths=gradient_strengths, TE=TE, n_t=n_t,
        timing=timing, family='ogse', q_width=sigma_,
        span=lambda m, g: sigma_[m] if shaped[m] else 0.0, sample=sample,
        encoding=lambda g, te, te_min: dict(oscillation_frequency=f_, gradient_duration=sigma_,
                                            n_oscillation_cycles=np.where(shaped, f_ * sigma_, 0.0),
                                            Delta=None if Delta is None else sigma_ + gap_,
                                            gradient_rise_time=np.array([eps(m, g[m]) for m in range(n_m)])),
        build_spec=('ogse', dict(gradient_directions=gradient_directions, oscillation_frequency=oscillation_frequency,
                                 gradient_duration=gradient_duration, shape=shape, Delta=Delta, bvalues=bvalues,
                                 gradient_strengths=gradient_strengths, TE=TE, n_t=n_t, slew_rate=slew_rate,
                                 timing=timing)))


# ── the stimulated echo ──────────────────────────────────────────────────────────────────────────────────────────

def pgste(gradient_directions, delta, TM, *, bvalues=None, gradient_strengths=None, TE=None, n_t=1000,
          slew_rate=DEFAULT_SLEW_RATE, timing=None, ste_flip_angles=(90.0, 90.0, 90.0)):
    """PGSTE (stimulated echo): a dephasing lobe, longitudinal storage over ``TM``, the same lobe rephasing.

    Three pulses and no 180 -- an excitation, a store that tips the encoded magnetisation onto z (only the
    stored half returns: the idealised 0.5 the engine applies), a recall ``TM`` later -- so the diffusion time
    ``Delta = delta + TM`` runs on T1, not T2. The lobes are the same sign; the recall's sign flip folds the
    second into ``G_eff``. The time transverse before the store equals the time after the recall (so the static
    field refocuses at the echo, ``TE = 2 t_store + TM``); rows with shorter ramps end their lobe at the same
    store. ``ste_flip_angles`` are the three flips (deg): the label says the role, whatever the flip.
    """
    _need_amplitude("pgste", bvalues, gradient_strengths)
    dirs, n_m, (delta_,) = _rows(gradient_directions, delta)
    TM = float(TM)
    eps = lambda m, g: ramp_of(g, slew_rate)
    return assemble(
        StimulatedEcho(TM, flips=ste_flip_angles, timing=timing),
        gradient_directions=dirs, bvalues=bvalues, gradient_strengths=gradient_strengths, TE=TE, n_t=n_t,
        timing=timing, family='pgste', q_width=delta_,
        span=lambda m, g: delta_[m] + eps(m, g),
        sample=lambda m, g, dt: trapezoid(delta_[m], eps(m, g), dt),
        encoding=lambda g, te, te_min: dict(delta=delta_, Delta=delta_ + TM, tau_perp_SE=np.full(n_m, te - TM),
                                            ramp_time=np.array([eps(m, g[m]) for m in range(n_m)]),
                                            ste_flip_angles=tuple(float(a) for a in ste_flip_angles)),
        build_spec=('pgste', dict(gradient_directions=gradient_directions, delta=delta, TM=TM, bvalues=bvalues,
                                  gradient_strengths=gradient_strengths, TE=TE, n_t=n_t, slew_rate=slew_rate,
                                  timing=timing, ste_flip_angles=ste_flip_angles)))


# ── filter-exchange imaging ──────────────────────────────────────────────────────────────────────────────────────

def _samples(x, dt, what):
    """``x`` seconds as the whole number of samples of ``dt`` it must be, or a refusal naming the nearest."""
    n = float(x) / float(dt)
    if abs(n - round(n)) > 1e-6 * max(1.0, abs(n)):
        raise ValueError(f"{what} = {float(x)*1e3:.6g} ms is {n:.4f} samples of {float(dt)*1e6:.4g} us: it must be a whole "
                         f"number of them ({int(np.floor(n))} or {int(np.ceil(n))} samples)")
    return int(round(n))


def fexi(gradient_directions, delta, Delta, t_m, *, b_f, bvalues, dt, slew_rate=DEFAULT_SLEW_RATE, timing=None,
         crusher_duration=1e-3, crusher_cycles=16.0, ste_flip_angles=(90.0, 90.0, 90.0)):
    """FEXI (filter-exchange imaging, Lasic et al. 2011): a diffusion filter, the magnetisation stored over the mixing
    time ``t_m``, a detection block; three pulses and no 180.

        90 -[+g_f ... -g_f]- crusher - 90 store - t_m - 90 recall - crusher -[+g_d ... -g_d]- echo
            '--- filter ---'                                                 '- detection -'

    Each block is a BIPOLAR pair (``+g`` then ``-g``, lobes of ``delta`` whose leading edges are ``Delta`` apart), so it
    refocuses on its own (``q = 0`` at the store and at the echo) and the mixing time carries no encoding: what the
    stored magnetisation does over ``t_m`` is exchange and longitudinal relaxation alone. A gap ``Delta - delta``
    between the lobes gives the block a diffusion time; a contiguous pair has too short a one to separate a restricted
    pool. The filter and the detection share ``delta`` and ``Delta``, so the time transverse before the store equals
    the time transverse after the recall and a static field refocuses at the echo (``TE = 2 tau + t_m``).

    The CRUSHERS select the stimulated echo: a voxel-scale winding of ``crusher_cycles`` turns over ``crusher_duration``
    just before the store and the same just after the recall, declared on ``crusher`` (a winding a micron cell cannot
    carry geometrically). The pathway they keep is the one whose phase the recall conjugates -- the effective gradient
    flips sign at the recall (:meth:`~dmipy_replay.acquisition.rf.RFSchedule.sign`) and is zero while stored -- at the
    amplitude :func:`dmipy_analytic.epg.ste_amplitude` of the three flips (1/2 at 90 degrees); the magnetisation that
    relaxes back along B0 during ``t_m`` is crushed with the rest.

    ``b_f`` is the filter's b-value and ``bvalues`` the detection's (s/m^2), per row (broadcast against the directions);
    a row with ``b_f = 0`` plays no filter (the equilibrium reference at the same timing). The amplitudes are found so
    that each block realises its b, to the float32 rounding of the stored gradient (:func:`fexi_bvalues` reads them
    back); a block's ramps are
    ``g / slew_rate`` of the larger of the row's two amplitudes (vertical at ``np.inf``). ``Encoding.bvalues`` is the
    whole effective gradient's b, the filter's and the detection's added. The grid is ``dt``: ``t_m`` and
    ``crusher_duration`` are whole numbers of it. With a ``timing`` budget the three pulses last ``t_excite`` each and
    the blocks keep out of the lead-in and the readout tail. ``ste_flip_angles`` are the three flips; each pulse plays
    the role its label says whatever its flip.
    """
    dt = float(dt)
    if dt <= 0.0:
        raise ValueError(f"dt is the sampling interval, a positive time; got {dt}")
    if float(crusher_duration) <= 0.0:
        raise ValueError("a filter-exchange sequence selects its stimulated echo with crushers: crusher_duration > 0")
    b_f_ = np.atleast_1d(np.asarray(b_f, np.float64))
    b_d_ = np.atleast_1d(np.asarray(bvalues, np.float64))
    n = max(b_f_.size, b_d_.size, 1 if gradient_directions is None else np.atleast_2d(gradient_directions).shape[0])
    if gradient_directions is not None and np.atleast_2d(gradient_directions).shape[0] == 1:
        gradient_directions = np.tile(np.atleast_2d(np.asarray(gradient_directions, np.float64)), (n, 1))
    dirs, n_m, (b_f_, b_d_, delta_, Delta_) = _rows(gradient_directions, b_f_ if b_f_.size > 1 else b_f_[0],
                                                     b_d_ if b_d_.size > 1 else b_d_[0], delta, Delta, n=n)
    dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
    if np.any(b_f_ < 0) or np.any(b_d_ < 0):
        raise ValueError("b_f and bvalues must be non-negative")
    n_m_mix = _samples(t_m, dt, "t_m")
    n_c = _samples(crusher_duration, dt, "crusher_duration")
    if n_m_mix < 1:
        raise ValueError("t_m must hold at least one sample: the store and the recall are distinct pulses")

    # the amplitudes: each block realises its own b; a row's two blocks share the ramp of the larger amplitude
    def unit_block(m, eps):
        return bipolar(delta_[m], Delta_[m], eps, dt)

    def b_unit(blk):
        return float(b_from_gradient(blk[None, :, None] * np.array([1.0, 0.0, 0.0]), dt)[0])

    a_f, a_d, eps_ = np.zeros(n_m), np.zeros(n_m), np.zeros(n_m)
    for m in range(n_m):
        eps = 0.0
        for _ in range(_B_ITER):
            bu = b_unit(unit_block(m, eps))
            af, ad = np.sqrt(b_f_[m] / bu), np.sqrt(b_d_[m] / bu)
            eps_new = ramp_of(max(af, ad), slew_rate)
            done = abs(eps_new - eps) <= _B_RTOL * max(eps_new, dt)
            eps = eps_new
            if done:
                break
        bu = b_unit(unit_block(m, eps))
        a_f[m], a_d[m], eps_[m] = np.sqrt(b_f_[m] / bu), np.sqrt(b_d_[m] / bu), eps
    blocks = [unit_block(m, eps_[m]) for m in range(n_m)]
    n_span = max(len(blk) for blk in blocks)

    # the schedule: tau transverse on either side, the store and the recall t_m apart
    w = 0.0 if timing is None else float(timing.t_excite)
    t_exc = 0.0 if timing is None else float(timing.t_prep)
    lead = 0.0 if timing is None else float(timing.t_lead)
    tail = 0.0 if timing is None else float(timing.t_readout_pre_echo)
    if timing is not None:
        _samples(t_exc, dt, "the timing budget's t_prep")
    n_w = int(np.ceil(w / 2.0 / dt - 1e-9))                  # half a pulse, in whole samples
    side = (n_span + n_c + n_w) * dt
    tau = int(np.ceil(max(lead - t_exc + side, side + tail) / dt - 1e-9)) * dt
    t_store = t_exc + tau
    t_recall = t_store + n_m_mix * dt
    TE = t_recall + tau
    n_t = int(round(TE / dt)) + 1
    a1, a2, a3 = (float(a) for a in ste_flip_angles)
    schedule = RFSchedule([RFEvent(t_exc, a1, 'Mz→Mxy', duration_s=w), RFEvent(t_store, a2, 'store', duration_s=w),
                           RFEvent(t_recall, a3, 'recall', duration_s=w)])
    f_end = t_store - (n_w + n_c) * dt                       # the filter ends where its crusher begins
    d_start = t_recall + (n_w + n_c) * dt                     # the detection starts where its crusher ends
    i_f_end, i_d0 = _steps_before(f_end, dt), _first_step_at(d_start, dt)
    G = np.zeros((n_m, n_t, 3))
    for m, blk in enumerate(blocks):
        G[m, i_f_end - len(blk):i_f_end] += a_f[m] * blk[:, None] * dirs[m]
        G[m, i_d0:i_d0 + len(blk)] += a_d[m] * blk[:, None] * dirs[m]
    crusher = {'windows_s': [(f_end, t_store - w / 2.0), (t_recall + w / 2.0, d_start)],
               'n_cycles': float(crusher_cycles)}
    seq = ScannerSequence(G=G, dt=dt, rf=schedule, timing=timing, crusher=crusher, family='fexi')
    # the declared b IS each block's b of the effective gradient the walk integrates
    bf_num, bd_num = fexi_bvalues(seq)
    sf = np.where((b_f_ > 0) & (bf_num > 0), np.sqrt(b_f_ / np.where(bf_num > 0, bf_num, 1.0)), 1.0)
    sd = np.where((b_d_ > 0) & (bd_num > 0), np.sqrt(b_d_ / np.where(bd_num > 0, bd_num, 1.0)), 1.0)
    i_mid = int(round(t_store / dt))
    G[:, :i_mid] *= sf[:, None, None]
    G[:, i_mid:] *= sd[:, None, None]
    TE_min = TE
    enc = Encoding(bvalues=b_from_gradient(np.asarray(seq.with_gradient(G).G_eff, np.float64), dt),
                   gradient_directions=dirs, TE=np.full(n_m, TE), delta=delta_, Delta=Delta_,
                   tau_perp_SE=np.full(n_m, TE - n_m_mix * dt), ramp_time=eps_, minimum_te=TE_min, te_auto=True,
                   ste_flip_angles=(a1, a2, a3))
    return ScannerSequence(
        G=G, dt=dt, rf=schedule, timing=timing, crusher=crusher, family='fexi', encoding=enc,
        notes="bipolar filter - crusher - store - t_m (exchange) - recall - crusher - bipolar detection",
        build_spec=('fexi', dict(gradient_directions=gradient_directions, delta=delta, Delta=Delta, t_m=t_m, b_f=b_f,
                                 bvalues=bvalues, dt=dt, slew_rate=slew_rate, timing=timing,
                                 crusher_duration=crusher_duration, crusher_cycles=crusher_cycles,
                                 ste_flip_angles=ste_flip_angles))).validate()


def fexi_protocol(gradient_directions, delta, Delta, t_m, *, b_f, bvalues, dt, **kwargs):
    """The standard FEXI protocol: the filter off and on, at every mixing time, at every detection b-value, along every
    direction -- one :func:`fexi` sequence per mixing time (one echo time each), its rows the filter (off, then on at
    ``b_f``) by the detection b-values by the directions, in that order. ``t_m`` lists the mixing times; ``kwargs`` are
    :func:`fexi`'s other keywords. Each row's filter and detection b-values, and the mixing time, are read back from the
    sequences (:func:`fexi_bvalues`, ``ScannerSequence.TM``)."""
    if float(b_f) <= 0.0:
        raise ValueError("b_f is the filter's b-value when it is on: positive (the protocol adds the filter-off rows)")
    dirs = np.atleast_2d(np.asarray(gradient_directions, np.float64))
    b_d = np.atleast_1d(np.asarray(bvalues, np.float64))
    filt, bd, dd = (a.reshape(-1) for a in np.meshgrid([0.0, float(b_f)], b_d, np.arange(len(dirs)), indexing="ij"))
    seqs = [fexi(dirs[dd.astype(int)], delta, Delta, tm, b_f=filt, bvalues=bd, dt=dt, **kwargs)
            for tm in np.atleast_1d(np.asarray(t_m, np.float64))]
    return Protocol(seqs)


def fexi_bvalues(seq):
    """``(b_f, b_d)``, ``(n_meas,)`` each in s/m^2: the b-values of a filter-exchange sequence's filter (the effective
    gradient before the store) and its detection (after the recall), each the package's b integral
    (:func:`~dmipy_replay.acquisition.waveforms.b_from_gradient`) of that block alone.

    Each is a b-value only because each block refocuses on its own: a sequence without a store and a recall, or
    whose gradient leaves a net moment at the store (a PGSTE, whose two lobes encode across the mixing time as one), is
    refused."""
    if not isinstance(seq, ScannerSequence):
        raise TypeError(f"fexi_bvalues reads a ScannerSequence; got {type(seq).__name__}")
    roles = {role_of(e): e for e in seq.rf if role_of(e) in ("store", "recall")}
    if set(roles) != {"store", "recall"}:
        raise ValueError("a filter-exchange sequence stores and recalls its magnetisation; this schedule has no "
                         "store / recall pair")
    i_store = int(round(roles["store"].t_s / seq.dt))
    i_recall = int(round(roles["recall"].t_s / seq.dt))
    G = np.asarray(seq.G_eff, np.float64)
    pre = np.zeros_like(G)
    pre[:, :i_store] = G[:, :i_store]
    post = np.zeros_like(G)
    post[:, i_recall:] = G[:, i_recall:]
    q_store = np.abs(np.sum(pre, axis=1) * seq.dt).max(axis=1)
    q_max = np.abs(np.cumsum(pre, axis=1) * seq.dt).max(axis=(1, 2))
    if np.any(q_store > REFOCUS_ATOL * np.maximum(q_max, 1e-300)):
        raise ValueError("the gradient before the store leaves a net moment at the store: the encoding runs across the "
                         "mixing time (a PGSTE) and has no separate filter b-value")
    return b_from_gradient(pre, seq.dt), b_from_gradient(post, seq.dt)


# ── gradient echoes ──────────────────────────────────────────────────────────────────────────────────────────────

def gre(TE, *, gradient_directions=None, bvalues=None, gradient_strengths=None, delta=None, Delta=None, n_t=1000,
        slew_rate=DEFAULT_SLEW_RATE, timing=None):
    """Gradient echo (no 180): a self-refocusing bipolar pair per measurement, or a pure FID.

    With ``bvalues`` / ``gradient_strengths`` the pair (``delta``, ``Delta``) starts once the excitation lets
    it and the rest of ``TE`` is free precession; with neither the gradient is zero. Physical and effective
    gradients coincide (nothing folds), so a static field is not refocused -- the readout is complex.
    """
    TE = float(TE)
    dirs, n_m, (delta_, Delta_) = _rows(gradient_directions, delta, Delta)
    weighted = bvalues is not None or gradient_strengths is not None
    if weighted and (delta is None or Delta is None):
        raise ValueError("gre: delta and Delta are required for a diffusion-weighted gradient echo")
    eps = lambda m, g: ramp_of(g, slew_rate)
    if weighted:
        span = lambda m, g: Delta_[m] + delta_[m] + eps(m, g)
        sample = lambda m, g, dt: bipolar(delta_[m], Delta_[m], eps(m, g), dt)
        enc = lambda g, te, te_min: dict(delta=delta_, Delta=Delta_, refocused=False,
                                         ramp_time=np.array([eps(m, g[m]) for m in range(n_m)]))
    else:
        gradient_strengths = np.zeros(n_m)
        span = lambda m, g: 0.0
        sample = lambda m, g, dt: np.zeros(1)
        enc = lambda g, te, te_min: dict(refocused=False)
    return assemble(
        GradientEcho(timing=timing), gradient_directions=dirs, bvalues=bvalues, gradient_strengths=gradient_strengths,
        TE=TE, n_t=n_t, timing=timing, family='gre', q_width=None if not weighted else delta_,
        span=span, sample=sample, encoding=enc,
        build_spec=('gre', dict(TE=TE, gradient_directions=gradient_directions, bvalues=bvalues,
                                gradient_strengths=gradient_strengths,
                                delta=delta, Delta=Delta, n_t=n_t, slew_rate=slew_rate, timing=timing)))


def ste(gradient_duration, *, bvalues=None, gradient_strengths=None, TE=None, n_t=1000, slew_rate=DEFAULT_SLEW_RATE,
        timing=None):
    """Spherical tensor encoding (``b_delta = 0``): three sequential self-refocused pairs, one per Cartesian axis,
    back to back over ``gradient_duration`` -- a gradient-echo encoding (an excitation only, nothing folds).
    Each axis's q returns to zero inside its own pair, so B is diagonal and, by symmetry, ``(b/3) I``."""
    _need_amplitude("ste", bvalues, gradient_strengths)
    n_m = max(np.size(bvalues) if bvalues is not None else 1, np.size(gradient_strengths) if gradient_strengths is not None else 1)
    dirs, n_m, (sigma_,) = _rows(None, gradient_duration, n=n_m)
    eps = lambda m, g: ramp_of(g, slew_rate)
    return assemble(
        GradientEcho(timing=timing), gradient_directions=dirs, bvalues=bvalues, gradient_strengths=gradient_strengths,
        TE=TE, n_t=n_t, timing=timing, family='ste',
        span=lambda m, g: sigma_[m],
        sample=lambda m, g, dt: axis_pairs(np.eye(3), sigma_[m], eps(m, g), dt),
        encoding=lambda g, te, te_min: dict(gradient_duration=sigma_, refocused=False,
                                            ramp_time=np.array([eps(m, g[m]) for m in range(n_m)])),
        build_spec=('ste', dict(gradient_duration=gradient_duration, bvalues=bvalues, gradient_strengths=gradient_strengths,
                                TE=TE, n_t=n_t, slew_rate=slew_rate, timing=timing)))


def pte(plane_normal, gradient_duration, *, bvalues=None, gradient_strengths=None, TE=None, n_t=1000,
        slew_rate=DEFAULT_SLEW_RATE, timing=None):
    """Planar tensor encoding (``b_delta = -0.5``): two sequential self-refocused pairs along two orthonormal axes
    of the plane normal to ``plane_normal``, back to back over ``gradient_duration`` -- a gradient-echo encoding.
    ``encoding.gradient_directions`` is the first in-plane axis."""
    _need_amplitude("pte", bvalues, gradient_strengths)
    n = np.asarray(plane_normal, dtype=np.float64)
    n = n / np.linalg.norm(n)
    ref = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = ref - np.dot(ref, n) * n
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    v /= np.linalg.norm(v)
    n_m = max(np.size(bvalues) if bvalues is not None else 1, np.size(gradient_strengths) if gradient_strengths is not None else 1)
    dirs, n_m, (sigma_,) = _rows(None, gradient_duration, default_direction=u, n=n_m)
    eps = lambda m, g: ramp_of(g, slew_rate)
    return assemble(
        GradientEcho(timing=timing), gradient_directions=dirs, bvalues=bvalues, gradient_strengths=gradient_strengths,
        TE=TE, n_t=n_t, timing=timing, family='pte',
        span=lambda m, g: sigma_[m],
        sample=lambda m, g, dt: axis_pairs(np.stack([u, v]), sigma_[m], eps(m, g), dt),
        encoding=lambda g, te, te_min: dict(gradient_duration=sigma_, refocused=False,
                                            ramp_time=np.array([eps(m, g[m]) for m in range(n_m)])),
        build_spec=('pte', dict(plane_normal=plane_normal, gradient_duration=gradient_duration, bvalues=bvalues,
                                gradient_strengths=gradient_strengths, TE=TE, n_t=n_t, slew_rate=slew_rate,
                                timing=timing)))


# ── the echo train ───────────────────────────────────────────────────────────────────────────────────────────────

def cpmg(n_echoes, TE, *, beta_deg=180.0, refocus_axis_deg=90.0, gradient_directions=None, bvalues=None,
         gradient_strengths=None, polarity="constant", n_t_per_echo=100, slew_rate=DEFAULT_SLEW_RATE, timing=None):
    """CPMG: a 90 about x and ``n_echoes`` refocusing pulses of ``beta_deg`` at ``(k + 1/2) TE``, an echo read
    at every ``k TE``; the grid runs to the last echo.

    ``refocus_axis_deg`` is what makes it Meiboom-Gill: the refocusing pulses turn about y (90, the default),
    a quarter turn from the excitation, so a flip-angle error ``beta_deg != 180`` cancels every second echo
    instead of accumulating; ``refocus_axis_deg=0`` is the Carr-Purcell train, which decays with the error. The
    transverse-only engine sees only the sign the pulses impose; ``simulate_bloch`` plays the axes.

    With neither ``bvalues`` nor ``gradient_strengths`` the train carries no gradient: a relaxation (T2)
    train, the usual CPMG. A diffusion-weighted train is opt-in: the gradient is then on wherever the pulses
    and readouts leave room, ``polarity='constant'`` at one sign throughout, ``'alternate'`` flipping every
    echo interval; either way each interval's ``G_eff`` is a bipolar pair and every echo refocuses.
    ``bvalues`` is the b of the whole train (the last echo's).
    """
    n_echoes, TE_echo = int(n_echoes), float(TE)
    if n_echoes < 1:
        raise ValueError("cpmg needs at least one echo")
    dirs, n_m, () = _rows(gradient_directions, n=max(np.size(bvalues) if bvalues is not None else 1,
                                                     np.size(gradient_strengths) if gradient_strengths is not None else 1))
    if bvalues is None and gradient_strengths is None:
        gradient_strengths = np.zeros(n_m)
    eps = lambda m, g: ramp_of(g, slew_rate)
    train = EchoTrain(n_echoes, TE_echo, polarity=polarity, beta_deg=beta_deg, refocus_axis_deg=refocus_axis_deg,
                      timing=timing)
    half = np.full(n_m, TE_echo / 2.0)
    return assemble(
        train, gradient_directions=dirs, bvalues=bvalues, gradient_strengths=gradient_strengths, TE=None,
        n_t=n_t_per_echo, timing=timing, family='cpmg', q_width=half,
        span=lambda m, g: 0.0, sample=None,
        fill=lambda m, g, dt, n: trapezoid(n * dt - eps(m, g), eps(m, g), dt)[:n],
        encoding=lambda g, te, te_min: dict(delta=half, Delta=half, refocused=True, cpmg_n_echoes=n_echoes,
                                            cpmg_TE=TE_echo, cpmg_beta_deg=float(beta_deg),
                                            cpmg_refocus_axis_deg=float(refocus_axis_deg),
                                            n_t_per_echo=int(n_t_per_echo),
                                            ramp_time=np.array([eps(m, g[m]) for m in range(n_m)])),
        build_spec=('cpmg', dict(n_echoes=n_echoes, TE=TE, gradient_directions=gradient_directions, bvalues=bvalues,
                                 gradient_strengths=gradient_strengths, polarity=polarity, beta_deg=beta_deg,
                                 n_t_per_echo=n_t_per_echo, slew_rate=slew_rate, timing=timing)))


def splice(gradient_directions, delta, Delta, n_echoes, TE_echo, *, bvalues=None, gradient_strengths=None,
           TE_prep, beta_deg=180.0, refocus_axis_deg=90.0, n_t_per_echo=40, slew_rate=DEFAULT_SLEW_RATE,
           crusher_cycles=16.0, crusher_width=None, timing=None):
    """A diffusion-prepared refocusing train: the shape an ultra-low-field diffusion FSE plays.

    A 90, a PGSE pair of ``delta`` at ``Delta`` about the preparation's 180, so the preparation's echo falls
    at ``TE_prep``; then ``n_echoes`` refocusing pulses of ``beta_deg`` about ``refocus_axis_deg``, one every
    ``TE_echo``, each with a **crusher pair** straddling it. ``bvalues`` is the b the preparation delivers;
    the train carries no gradient of its own.

    The crusher is what makes a train a train. Without voxel-scale dephasing every coherence pathway stays
    degenerate and recombines, so the echoes barely depend on the refocusing flip angle -- and a real train's
    behaviour at a reduced flip IS that dependence. It is declared here as ``crusher`` on the sequence, in
    windows of ``crusher_width`` (default half a train interval) either side of every refocusing pulse,
    winding ``crusher_cycles`` turns; :meth:`~dmipy_replay.replay.ReplayPack.replay_bloch` and
    dmipy-sim's ``engine.bloch.simulate_bloch`` both read it. The window is balanced about its pulse so the pulse
    rewinds what the window wound, which is what preserves the refocused pathway while spoiling the ones that
    were stored for part of it.

    ``refocus_axis_deg`` is the Meiboom-Gill quarter turn from the excitation (90, the default), where a flip
    error self-corrects every second echo; 0 is Carr-Purcell, where it accumulates. That choice is the CPMG
    condition, and after a diffusion preparation it is violated walker by walker whatever the axis, which is
    the loss this family exists to work around.

    Every echo lands on a sample: ``n_t_per_echo`` samples fill each train interval and the preparation is a
    whole number of them, or the build is refused.
    """
    n_echoes, TE_echo, delta_, Delta_ = int(n_echoes), float(TE_echo), float(delta), float(Delta)
    if n_echoes < 1:
        raise ValueError("splice needs at least one echo after the preparation")
    _need_amplitude("splice", bvalues, gradient_strengths)
    dirs, n_m, (delta_r, Delta_r) = _rows(gradient_directions, delta_, Delta_,
                                          n=max(np.size(bvalues) if bvalues is not None else 1,
                                                np.size(gradient_strengths) if gradient_strengths is not None else 1))
    eps = lambda m, g: ramp_of(g, slew_rate)
    train = PreparedEchoTrain(n_echoes, TE_echo, float(TE_prep),
                              gap=lambda m, g: Delta_r[m] - delta_r[m] - eps(m, g),
                              beta_deg=beta_deg, refocus_axis_deg=refocus_axis_deg, timing=timing)
    TE = float(TE_prep) + n_echoes * TE_echo
    seq = assemble(
        train, gradient_directions=dirs, bvalues=bvalues, gradient_strengths=gradient_strengths, TE=TE,
        n_t=n_t_per_echo, timing=timing, family='splice', q_width=delta_r,
        span=lambda m, g: delta_r[m] + eps(m, g), sample=lambda m, g, dt: trapezoid(delta_r[m], eps(m, g), dt),
        encoding=lambda g, te, te_min: dict(delta=delta_r, Delta=Delta_r, refocused=True,
                                            splice_n_echoes=n_echoes, splice_TE_echo=TE_echo,
                                            splice_TE_prep=float(TE_prep), splice_beta_deg=float(beta_deg),
                                            splice_refocus_axis_deg=float(refocus_axis_deg),
                                            n_t_per_echo=int(n_t_per_echo),
                                            ramp_time=np.array([eps(m, g[m]) for m in range(n_m)])),
        build_spec=('splice', dict(gradient_directions=gradient_directions, delta=delta, Delta=Delta,
                                   n_echoes=n_echoes, TE_echo=TE_echo, bvalues=bvalues,
                                   gradient_strengths=gradient_strengths, TE_prep=TE_prep, beta_deg=beta_deg,
                                   refocus_axis_deg=refocus_axis_deg, n_t_per_echo=n_t_per_echo,
                                   slew_rate=slew_rate, crusher_cycles=crusher_cycles,
                                   crusher_width=crusher_width, timing=timing)))
    half = 0.5 * TE_echo if crusher_width is None else float(crusher_width)
    n_half = max(1, int(round(half / float(seq.dt))))                 # whole samples either side: a balanced pair
    windows = []
    for e in seq.rf:
        if e.label == 'refocus':
            i = int(round(float(e.t_s) / float(seq.dt)))
            windows.append(((i - n_half) * float(seq.dt), (i + n_half) * float(seq.dt)))
    return replace(seq, crusher={'windows_s': windows, 'n_cycles': float(crusher_cycles)})


# ── readers of a played gradient ─────────────────────────────────────────────────────────────────────────────────

def from_waveform(G, dt, gradient_directions, delta=None, Delta=None, TE=None, allow_unrefocused=False):
    """An arbitrary gradient waveform as an acquisition; b numerically from ``G``. No pulses are declared, so
    the gradient must refocus on its own (``allow_unrefocused=True`` turns that refusal into a warning)."""
    from warnings import warn
    G = np.asarray(G, dtype=np.float32)
    gradient_directions = np.asarray(gradient_directions, dtype=np.float64)
    bvalues = b_from_gradient(G, dt)
    delta_, Delta_, TE_ = unify_length_reference_delta_Delta(bvalues, delta, Delta, TE)
    qvalues = gradient_strengths = None
    if delta_ is not None and Delta_ is not None:
        gradient_strengths = g_from_b(bvalues, delta_, Delta_)
        qvalues = q_from_b(bvalues, delta_, Delta_)
    seq = ScannerSequence(G=G, dt=dt, family='waveform',
                          encoding=Encoding(bvalues=bvalues, gradient_directions=gradient_directions, TE=TE_,
                                            qvalues=qvalues, gradient_strengths=gradient_strengths, delta=delta_,
                                            Delta=Delta_))
    try:
        seq.validate()
    except ValueError as e:
        msg = (f"from_waveform: {e}; stationary spins will not rephase at TE. Ensure the waveform "
               f"refocuses, or pass allow_unrefocused=True if intentional.")
        if allow_unrefocused:
            warn(msg)
        else:
            raise ValueError(msg) from None
    return seq


def from_btensor_waveform(G, dt, *, echo_idx=None, TE=None, timing=None):
    """A precomputed b-tensor gradient waveform -- the PHYSICAL gradient, as played -- as a spin echo.

    For an externally designed b-tensor encoding (e.g. a dmipy-design ``design_waveform`` output) rather than
    the canonical pairs of :func:`ste` / :func:`pte`. The b-tensor SHAPE (spherical / planar / linear, i.e.
    b_delta) is whatever the numbers produce: it is computed from ``G_eff``, not declared, so there is no shape
    argument; ``seq.btensor()`` reads the realised shape. The 180 is declared at ``echo_idx`` (default TE/2 --
    the only position at which the static field refocuses at the echo, so a non-TE/2 ``echo_idx`` raises) and
    folds into ``G_eff`` through the schedule's sign, as for every family. With a ``timing`` budget the pulses
    are the finite 90 / 180 it implies and the sequence carries it (a designer's output built to a budget
    arrives with the budget, and ``validate()`` holds the gradient to its windows).
    """
    G = np.asarray(G, dtype=np.float32)
    if G.ndim == 2:
        G = G[None]
    n_m, n_t, _ = G.shape
    dt = float(dt)
    T_total = (n_t - 1) * dt
    gradient_directions = np.tile([0., 0., 1.], (n_m, 1))
    TE_, te_auto = _resolve_te(TE, T_total, n_m)
    te2 = int(round((float(TE_[0]) / 2.0) / dt))      # the spin-echo 180 position
    if echo_idx is None:
        echo_idx = te2
    echo_idx = int(np.clip(echo_idx, 0, n_t - 1))
    if abs(echo_idx - te2) > 1:                       # beyond rounding
        raise ValueError(
            f"from_btensor_waveform: echo_idx={echo_idx} places the 180 at {echo_idx * dt * 1e3:.2f} ms, not "
            f"TE/2={te2 * dt * 1e3:.2f} ms. A spin echo refocuses the static field at 2*t_180="
            f"{2 * echo_idx * dt * 1e3:.2f} ms, not at the echo TE={float(TE_[0]) * 1e3:.2f} ms.")
    if timing is None:
        schedule = RFSchedule([RFEvent(0.0, 90, 'Mz→Mxy'), RFEvent(echo_idx * dt, 180, 'refocus')])
    else:
        schedule = RFSchedule([RFEvent(timing.t_prep, 90, 'Mz→Mxy', duration_s=timing.t_excite),
                               RFEvent(echo_idx * dt, 180, 'refocus', duration_s=timing.t_refocus)])
    sign = schedule.sign(np.arange(n_t) * dt)[None, :, None]
    bvalues_num = b_from_gradient(G * sign, dt)
    return ScannerSequence(
        G=G, dt=dt, rf=schedule, timing=timing, family='btensor',
        encoding=Encoding(bvalues=bvalues_num, gradient_directions=gradient_directions, TE=TE_, minimum_te=T_total,
                          te_auto=te_auto),
        build_spec=('from_btensor_waveform', dict(G=G, dt=dt, echo_idx=echo_idx, TE=TE, timing=timing))).validate()


def from_pgste_waveform(G, dt, *, store_idx, recall_idx, gradient_directions=None, ste_flip_angles=(90., 90., 90.),
                        TE=None, timing=None):
    """A precomputed PGSTE (stimulated-echo) gradient waveform -- the PHYSICAL gradient, as played -- as a
    stimulated echo.

    The stimulated-echo analogue of :func:`from_btensor_waveform`, for an externally designed PGSTE encoding
    (e.g. a dmipy-design ``design_stimulated_echo`` output) rather than the canonical lobes of :func:`pgste`.
    Three pulses (excite / store / recall) and NO 180, so the static field is refocused only when the two
    transverse encoding periods match (``tau1 = tau3``, the PGSTE analogue of a 180 at TE/2); unequal periods
    leave a residual static dephasing at the echo and are refused. The two same-sign lobes straddle a
    gradient-off mixing time ``TM`` on z.

    ``store_idx`` / ``recall_idx`` are the samples of the store and recall pulses; ``delta = store_idx * dt`` and
    ``TM = (recall_idx - store_idx) * dt`` follow. With a ``timing`` budget the three pulses are finite
    (``t_excite`` each) and the sequence carries the budget.
    """
    G = np.asarray(G, dtype=np.float32)
    if G.ndim == 2:
        G = G[None]
    n_m, n_t, _ = G.shape
    dt = float(dt)
    store_idx = int(store_idx); recall_idx = int(recall_idx)
    if not (0 < store_idx < recall_idx < n_t):
        raise ValueError(f"from_pgste_waveform: need 0 < store_idx ({store_idx}) < recall_idx ({recall_idx}) < n_t ({n_t}).")
    T_total = (n_t - 1) * dt
    enc = np.any(np.abs(G[0]) > 1e-9, axis=1)             # (n_t,) encoded samples
    tau1 = int(np.sum(enc[:store_idx])) * dt
    tau3 = int(np.sum(enc[recall_idx:])) * dt
    if abs(tau1 - tau3) > 2.0 * dt:
        raise ValueError(
            f"from_pgste_waveform: tau1 = {tau1*1e3:.2f} ms != tau3 = {tau3*1e3:.2f} ms. A stimulated echo refocuses the "
            f"static field only when the two transverse encoding periods match; unequal periods leave a residual "
            f"static dephasing at the echo.")
    a1, a2, a3 = (float(a) for a in ste_flip_angles)
    w = 0.0 if timing is None else float(timing.t_excite)
    schedule = RFSchedule([RFEvent(0.0 if timing is None else timing.t_prep, a1, 'Mz→Mxy', duration_s=w),
                           RFEvent(store_idx * dt, a2, 'store', duration_s=w),
                           RFEvent(recall_idx * dt, a3, 'recall', duration_s=w)])
    sign = schedule.sign(np.arange(n_t) * dt)[None, :, None]
    delta = float(store_idx * dt)
    TM = float((recall_idx - store_idx) * dt)
    bvalues_num = b_from_gradient(G * sign, dt)
    gradient_directions = (np.tile([0., 0., 1.], (n_m, 1)) if gradient_directions is None
                           else np.asarray(gradient_directions, dtype=np.float64))
    TE_, te_auto = _resolve_te(TE, T_total, n_m)
    delta_ = np.full(n_m, delta)
    return ScannerSequence(
        G=G, dt=dt, rf=schedule, timing=timing, family='pgste',
        encoding=Encoding(bvalues=bvalues_num, gradient_directions=gradient_directions, TE=TE_, delta=delta_,
                          Delta=delta_ + TM, minimum_te=T_total, te_auto=te_auto, ste_flip_angles=(a1, a2, a3)),
        build_spec=('from_pgste_waveform', dict(G=G, dt=dt, store_idx=store_idx, recall_idx=recall_idx,
                                                gradient_directions=gradient_directions,
                                                ste_flip_angles=ste_flip_angles, TE=TE, timing=timing))).validate()


# ── views of a built sequence ────────────────────────────────────────────────────────────────────────────────────

def instantaneous(seq):
    """The idealised instantaneous (infinite-slew / square) limit of a built sequence: the same builder with
    ``slew_rate=np.inf`` where the builder slew-limits; the others rebuild as they are. This is the idealised
    view an analytical model reads."""
    if seq.build_spec is None:
        return seq
    name, kwargs = seq.build_spec
    kwargs = dict(kwargs)
    if 'slew_rate' in kwargs:
        kwargs['slew_rate'] = np.inf
    return globals()[name](**kwargs)


def to_gradient_array(seq, n_t=1000):
    """``(G_eff, dt)`` -- the EFFECTIVE gradient -- of the square PGSE with this sequence's b-values, directions,
    delta and Delta on an ``n_t`` grid: :func:`pgse` at infinite slew, as the analytical layer integrates it."""
    e = seq.encoding
    if e is None or e.delta is None or e.Delta is None or e.gradient_strengths is None:
        raise ValueError("to_gradient_array() requires an encoding with delta, Delta, and gradient_strengths.")
    delta_tol = np.float32(1e-6)
    if (np.max(e.delta) - np.min(e.delta)) > delta_tol:
        raise ValueError("to_gradient_array() requires uniform delta.")
    if (np.max(e.Delta) - np.min(e.Delta)) > delta_tol:
        raise ValueError("to_gradient_array() requires uniform Delta.")
    sq = pgse(e.gradient_directions, float(e.delta[0]), float(e.Delta[0]), bvalues=e.bvalues, n_t=n_t, slew_rate=np.inf)
    return sq.G_eff, sq.dt
