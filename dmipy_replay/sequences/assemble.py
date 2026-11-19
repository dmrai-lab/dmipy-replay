"""Assemblers x shapes: the one mechanics behind every sequence builder.

A builder is a thin call. It names a SHAPE -- what one encoding block looks like as a function of its
amplitude (a trapezoid lobe, a train of them, a cosine, a bipolar pair, the per-axis pairs of a b-tensor
encoding) -- and an ASSEMBLER -- which RF schedule the block is played around and where the block sits
relative to the pulses -- and hands both the measurement axis. Everything a
:class:`~dmipy_replay.acquisition.scanner_sequence.ScannerSequence` needs is produced here once: the grid that
runs from t = 0 to TE, the PHYSICAL gradient, the schedule, the exact b, the ``Encoding``.

The assemblers:

* :class:`SpinEcho` -- a 90, one 180 at TE/2, the echo at TE. Every row's block pair is centred on the 180:
  the same block before and after it (what a scanner plays), which the 180 folds into ``G_eff = B, -B`` so
  ``q(TE) = 0`` sample for sample. A row's own gap between its blocks (``Delta - delta - ramp`` for a PGSE, the
  refocusing window for an OGSE) is kept; a TE longer than the minimum adds dead time symmetrically outside
  the pair.
* :class:`StimulatedEcho` -- a 90, a store, a recall; no 180. The block before the store and the same block
  after the recall, with the time transverse before the store equal to the time transverse after the recall
  (the stimulated echo's analogue of a 180 at TE/2, so the static field refocuses at the echo) and the mixing
  time ``TM`` between them; ``TE = 2 t_store + TM``.
* :class:`GradientEcho` -- a 90 and no refocusing pulse: the block must refocus itself (a bipolar pair, the
  per-axis pairs of an STE / PTE). It starts when the excitation lets it; the rest of TE is free precession.
* :class:`EchoTrain` -- a 90 and ``n_echoes`` refocusing pulses at ``(k + 1/2) TE``, an echo at every ``k TE``.
  The gradient is on wherever the pulses and readouts leave room, at one polarity or alternating per interval
  (Carr-Purcell either way in ``G_eff``); each contiguous stretch is one lobe with its own ramps.
* :class:`PreparedEchoTrain` -- a diffusion preparation whose own echo forms at ``TE_prep``, then ``n_echoes``
  refocusing pulses of ``beta_deg`` every ``TE_echo``. The encoding sits either side of the preparation's 180
  exactly as :class:`SpinEcho` places it; the train carries no gradient of its own, and what separates its
  coherence pathways is the crusher the builder declares. This is the shape a diffusion-weighted fast spin
  echo plays, and what :func:`~dmipy_replay.sequences.builders.splice` is built on.

Slew is a LIMIT, never a fork: a finite ``slew_rate`` gives every lobe ramps of ``g / slew`` (a cosine, whose
own slope must stay under the limit, gets ramped edges), ``np.inf`` gives vertical ones, and the structure is
the same either way. A block that cannot reach its amplitude inside its own span is refused, not clipped. A
sequence built at a finite slew plays within it: no step of its played gradient, on any axis, is steeper than
the slew (:meth:`~dmipy_replay.acquisition.scanner_sequence.ScannerSequence.check_slew`, applied before it is
returned), and each ramped lobe's area is its ideal area to within ``min(1, dt / (4 eps)) g dt``.

A builder passes either ``bvalues`` or ``gradient_strengths``. With ``bvalues`` the amplitude is found by
iteration (the ramps depend on it) and then scaled so ``bvalues == b_from_gradient(G_eff, dt)`` exactly; with
``gradient_strengths`` the b is whatever the played gradient integrates to. A timing budget
(:class:`~dmipy_replay.acquisition.timing.SequenceTiming`) makes the pulses finite and keeps the gradient out of
the lead-in, the refocusing windows and the readout tails; without one the pulses are instantaneous and
constrain nothing.
"""
from __future__ import annotations

import math

import numpy as np

from ..acquisition.rf import RFEvent, RFSchedule
from ..acquisition.scanner_sequence import Encoding, ScannerSequence
from ..constants import resolve_slew
from dmipy_analytic.constants import GAMMA
from ..acquisition.waveforms import b_from_gradient

__all__ = ["trapezoid", "trapezoid_train", "cosine", "bipolar", "axis_pairs",
           "SpinEcho", "StimulatedEcho", "GradientEcho", "EchoTrain", "PreparedEchoTrain",
           "assemble", "ramp_of", "pair_timings"]

_B_ITER = 16          # amplitude iterations (the ramps depend on the amplitude)
_B_RTOL = 1e-10
_EPS_T = 1e-9         # seconds: placement tolerance against floating rounding


# ── shapes: unit-amplitude blocks sampled on the sequence grid ─────────────────────────────────────────────────
#
# Sample k of a block is the value of its ideal shape at the middle of its step, (k + 1/2) dt; the engine applies
# G[k] over [k dt, (k + 1) dt). A RAMPED block (eps > 0) keeps every sample whose mid-step lies inside its span
# (:func:`_inside`), so the waveform it plays is the ideal shape sampled, zero outside it: the ideal shape changes
# by at most dt / eps (unit amplitude) between two points dt apart, hence so does the sampled block, and no step is
# steeper than the slew the ramps were built for. Lobes that follow one another are one sampled waveform (a train),
# or, where a pair must cancel sample for sample (an axis pair, a bipolar pair), the same lobe negated and placed
# no earlier than the first step at or after the first lobe's ideal end. A square block (eps = 0, the
# instantaneous limit) has ``floor(span / dt)`` samples and no slew to hold.

def ramp_of(g, slew_rate):
    """The ramp a lobe of amplitude ``g`` needs at ``slew_rate`` (0 in the instantaneous limit)."""
    slew, square = resolve_slew(slew_rate)
    return 0.0 if square or g <= 0.0 else float(g) / float(slew)


def _inside(span, dt):
    """The number of steps whose mid-step ``(k + 1/2) dt`` lies inside ``(0, span)`` (at least one)."""
    return max(1, int(math.ceil(float(span) / dt - 0.5 - _EPS_T)))


def _square_count(span, dt):
    """The number of whole steps inside ``span`` (at least one): a square block's length."""
    return max(1, int(math.floor(float(span) / dt + _EPS_T)))


def _lobe_at(t, delta, eps):
    """The ideal unit lobe at times ``t`` (its own clock, 0 at its start): a ramp 0 -> 1 over ``eps``, flat until
    ``delta``, a ramp 1 -> 0 over ``eps``, zero outside ``[0, delta + eps]``."""
    a = np.ones_like(t)
    up = t < eps
    a[up] = t[up] / eps
    dn = t >= delta
    a[dn] = np.clip((delta + eps - t[dn]) / eps, 0.0, 1.0)
    a[t < 0.0] = 0.0
    return a


def trapezoid(delta, eps, dt):
    """One unit lobe: a ramp 0 -> 1 over ``eps``, flat until ``delta``, a ramp 1 -> 0 over ``eps`` -- the
    half-amplitude width is ``delta``, the physical span ``delta + eps``; ``eps = 0`` is the square limit.
    Sample ``k`` is the value at the middle of its step, ``(k + 1/2) dt``; a ramped lobe keeps every sample whose
    mid-step lies inside its span, so no step between samples (nor from or to zero) exceeds ``dt / eps``."""
    delta, eps = float(delta), float(eps)
    if eps > delta + _EPS_T:
        raise ValueError(f"a lobe of {delta*1e3:.3f} ms (half-amplitude width) cannot ramp over {eps*1e3:.3f} ms: "
                         f"the ramp is longer than the lobe -- lower the amplitude, lengthen the lobe or raise the slew")
    if eps <= 0.0:
        return np.ones(_square_count(delta, dt))
    t = (np.arange(_inside(delta + eps, dt)) + 0.5) * dt
    return _lobe_at(t, delta, eps)


def trapezoid_train(n_lobes, lobe, eps, dt):
    """``n_lobes`` adjacent unit lobes of span ``lobe`` each, alternating in sign (+, -, +, ...): the trapezoidal
    OGSE block (Drobnjak et al. 2016); one lobe is a PGSE lobe. Each lobe ramps over ``eps`` at both ends, so two
    neighbours pass from +1 to -1 along one straight ramp of ``2 eps``. A ramped train is the one ideal waveform
    sampled at mid-step (every lobe at its own ideal time); a square train is ``n_lobes`` square lobes of
    ``floor(lobe / dt)`` samples each, back to back."""
    if eps > lobe / 2.0 + _EPS_T:
        raise ValueError(f"a lobe of {lobe*1e3:.3f} ms cannot ramp up and down over {eps*1e3:.3f} ms each at this "
                         f"amplitude and slew: lower the amplitude, the frequency, or raise the slew")
    n_lobes, lobe, eps = int(n_lobes), float(lobe), float(eps)
    if eps <= 0.0:
        one = trapezoid(lobe, 0.0, dt)
        n1 = len(one)
        out = np.empty(n_lobes * n1)
        for k in range(n_lobes):
            out[k * n1:(k + 1) * n1] = one if k % 2 == 0 else -one
        return out
    t = (np.arange(_inside(n_lobes * lobe, dt)) + 0.5) * dt
    k = np.minimum(np.floor(t / lobe).astype(int), n_lobes - 1)
    return np.where(k % 2 == 0, 1.0, -1.0) * _lobe_at(t - k * lobe, lobe - eps, eps)


def cosine(n_cycles, frequency, eps, dt):
    """``n_cycles`` whole periods of ``cos(2 pi f t)`` -- the frequency-selective OGSE block, DC-free in q -- with
    its edges ramped over ``eps`` (the cosine starts and ends at full amplitude; a scanner ramps into it). A ramped
    block keeps every sample whose mid-step lies inside its span; a square one (``eps = 0``) has
    ``floor(span / dt)`` samples."""
    sigma = float(n_cycles) / float(frequency)
    n = max(2, _inside(sigma, dt) if eps > 0.0 else _square_count(sigma, dt))
    t = (np.arange(n) + 0.5) * dt                                     # mid-step, as every shape
    a = np.cos(2.0 * np.pi * float(frequency) * t)
    if eps > 0.0:
        a = a * np.clip(t / eps, 0.0, 1.0) * np.clip((sigma - t) / eps, 0.0, 1.0)
    return a


def _second_lobe_at(i1, span, eps, dt):
    """Where the negated copy of a lobe of ``span`` starts in a pair: sample ``i1``, and for a ramped lobe no earlier
    than ``ceil(span / dt)``, the first step at or after the first lobe's ideal end. The step across the junction is
    then the first lobe's last sample plus its first, at most ``dt / eps``."""
    if eps <= 0.0:
        return int(i1)
    return max(int(i1), int(math.ceil(float(span) / dt - _EPS_T)))


def bipolar(delta, Delta, eps, dt):
    """A self-refocusing pair: a unit lobe at 0 and its negative ``Delta`` later (centre to centre), sample for
    sample, so the pair's samples sum to zero exactly."""
    if float(Delta) < float(delta) + float(eps) - _EPS_T:
        raise ValueError("the two lobes of a bipolar pair overlap: Delta is shorter than the lobe with its ramp")
    lobe = trapezoid(delta, eps, dt)
    i1 = _second_lobe_at(round(float(Delta) / dt), float(delta) + float(eps), float(eps), dt)
    out = np.zeros(i1 + len(lobe))
    out[:len(lobe)] += lobe
    out[i1:] -= lobe
    return out


def axis_pairs(axes, duration, eps, dt):
    """Sequential self-refocusing pairs, one per axis in ``axes`` (rows, unit vectors), back to back over
    ``duration``: the canonical spherical (three axes) and planar (two) tensor encodings. Returns ``(n, 3)``.
    Each axis plays a lobe of span ``duration / (2 n_axes)`` and then its exact negative, sample for sample, so its
    q returns to zero inside its own pair and the off-diagonal B is exactly zero. A ramped pair's negative lobe
    starts at the first step at or after the first lobe's ideal end, so each pair is less than one step longer than
    its ideal ``2 duration / (2 n_axes)`` and pair ``i`` starts less than ``i`` steps after its ideal time."""
    axes = np.asarray(axes, dtype=np.float64)
    n_seg = 2 * len(axes)
    seg = float(duration) / n_seg
    lobe = trapezoid(seg - eps, eps, dt)
    n1 = len(lobe)
    c1 = _second_lobe_at(n1, seg, float(eps), dt)                     # the negative lobe's offset in the pair
    out = np.zeros((len(axes) * (c1 + n1), 3))
    for i, ax in enumerate(axes):
        a0 = i * (c1 + n1)
        out[a0:a0 + n1] += lobe[:, None] * ax
        out[a0 + c1:a0 + c1 + n1] -= lobe[:, None] * ax
    return out


# ── assemblers: an RF schedule and where the blocks sit around it ──────────────────────────────────────────────

def _lead(timing):
    return 0.0 if timing is None else float(timing.t_lead)


def _tail(timing):
    return 0.0 if timing is None else float(timing.t_readout_pre_echo)


def _dur(timing, which):
    return 0.0 if timing is None else float(getattr(timing, which))


def _steps_before(t, dt):
    """How many steps ``[k dt, (k + 1) dt)`` end by time ``t``: the exclusive end index of a block that must be
    over by ``t``."""
    return int(math.floor(t / dt + _EPS_T))


def _first_step_at(t, dt):
    """The first step that starts at or after time ``t``."""
    return int(math.ceil(t / dt - _EPS_T))


def _margin_cap(dt, end1_idx, start2_idx, lead, tail_bound):
    """The largest shared length (samples) a :class:`SpinEcho`-style block pair -- one ``end``-anchored at the
    grid index ``end1_idx`` (against the 180's gap), the other ``start``-anchored at ``start2_idx`` (against
    the same gap, on the far side) -- may have without either block's OWN far edge landing inside a budget's
    lead-in (block 1's start, against ``lead``) or readout-tail (block 2's end, against ``tail_bound``) dead
    window once discretised.

    The near edge of each block (the 180's gap) is already grid-safe (``_steps_before`` / ``_first_step_at``
    round towards it); a block's LENGTH is rounded independently by its shape, and on whichever margin binds
    with zero continuous slack (:meth:`SpinEcho.te_min`'s ``2 * max(lead, tail)``) the two roundings do not
    cancel, so the discretised block can land up to about one sample past that edge (dmipy-replay#645) -- a
    genuine overlap with the dead window, not mere rounding. Both blocks are capped to the SAME length here
    (never lengthened, only shortened); :func:`assemble`'s placement loop is the one that then keeps them
    exact mirror images of each other across the 180 (the raw shape need not itself be palindromic at an
    arbitrary (span, dt), so independently slicing each block's far end from it would not leave equal
    moments -- see the loop's own note)."""
    return min(end1_idx - _first_step_at(lead, dt), _steps_before(tail_bound, dt) - start2_idx)


# A placement says where a block goes. Sample k acts over the step [k dt, (k + 1) dt) (the engine's rule), and
# the readout sample n_t - 1 acts over nothing. ("end", t, factor[, cap]): the block's last step ends by time t
# (it ends against a pulse, or at the readout); ("start", t, factor[, cap]): its first step starts at or after
# t (it follows a pulse); ("at", i0, factor, n): exactly n samples from sample i0 (a stretch handed out to
# fill). The optional ``cap`` on "start"/"end" (:func:`_margin_cap`) truncates the block -- from its front for
# "end", its back for "start", i.e. always the half FARTHER from the anchor -- to at most ``cap`` samples; a
# "start" capped the SAME as a preceding same-row "end" is instead the exact time-reverse of that "end"
# block's (already-truncated) array, not a second independent slice (`assemble()`'s loop, dmipy-replay#645).


class SpinEcho:
    """90 at 0, 180 at TE/2, echo at TE; each row's block pair centred on the 180 with its own ``gap``."""

    truncates_to_margin = True            # an explicit TE's block pair is cut to the budget's margin (_margin_cap)

    def __init__(self, gap, timing=None):
        self.gap = gap                    # callable (m, g) -> seconds between the two blocks
        self.timing = timing

    def te_min(self, spans, g):
        gaps = np.array([self.gap(m, g[m]) for m in range(len(spans))])
        if np.any(gaps < -_EPS_T):
            raise ValueError("the two encoding blocks overlap: Delta is shorter than the lobe with its ramp")
        if self.timing is not None and np.any(gaps < self.timing.t_refocus - _EPS_T):
            raise ValueError(f"the gap between the blocks, {float(np.min(gaps))*1e3:.3f} ms, is narrower than the "
                             f"refocusing window {self.timing.t_refocus*1e3:.3f} ms: the 180 does not fit")
        return float(np.max(2.0 * spans + gaps)) + 2.0 * max(_lead(self.timing), _tail(self.timing))

    def layout(self, spans, g, TE, dt, n_t):
        schedule = (self.timing.rf_events(TE) if self.timing is not None else
                    RFSchedule([RFEvent(0.0, 90, 'Mz→Mxy'), RFEvent(TE / 2.0, 180, 'refocus')]))
        placements = []
        for m, span in enumerate(spans):
            gap = self.gap(m, g[m])
            end1, start2 = TE / 2.0 - gap / 2.0, TE / 2.0 + gap / 2.0
            cap = None
            if self.timing is not None:
                cap = _margin_cap(dt, _steps_before(end1, dt), _first_step_at(start2, dt),
                                  _lead(self.timing), TE - _tail(self.timing))
            placements.append([("end", end1, 1.0, cap), ("start", start2, 1.0, cap)])
        return schedule, placements


class StimulatedEcho:
    """90 at 0, store, recall; the block before the store and after the recall, ``TE = 2 t_store + TM``."""

    truncates_to_margin = True

    def __init__(self, TM, flips=(90.0, 90.0, 90.0), timing=None):
        self.TM = float(TM)
        self.flips = tuple(float(a) for a in flips)
        self.timing = timing

    def te_min(self, spans, g):
        w = _dur(self.timing, "t_excite")
        return 2.0 * float(np.max(spans)) + w + self.TM + 2.0 * max(_lead(self.timing), _tail(self.timing))

    def layout(self, spans, g, TE, dt, n_t):
        w = _dur(self.timing, "t_excite")
        s_max = float(np.max(spans))
        d = (TE - self.TM - 2.0 * s_max - w) / 2.0
        t_store = d + s_max + w / 2.0
        t_recall = t_store + self.TM
        a1, a2, a3 = self.flips
        schedule = RFSchedule([RFEvent(_dur(self.timing, "t_prep"), a1, 'Mz→Mxy', duration_s=w),
                               RFEvent(t_store, a2, 'store', duration_s=w),
                               RFEvent(t_recall, a3, 'recall', duration_s=w)])
        end1, start2 = t_store - w / 2.0, t_recall + w / 2.0
        cap = None
        if self.timing is not None:
            cap = _margin_cap(dt, _steps_before(end1, dt), _first_step_at(start2, dt),
                              _lead(self.timing), TE - _tail(self.timing))
        return schedule, [[("end", end1, 1.0, cap), ("start", start2, 1.0, cap)] for _ in spans]


class GradientEcho:
    """90 at 0 and no refocusing pulse; the (self-refocusing) block starts when the excitation lets it and must end
    before the budget's readout tail: a block that refocuses itself is never cut, so one longer than that room is
    refused."""

    truncates_to_margin = False

    def __init__(self, timing=None):
        self.timing = timing

    def te_min(self, spans, g):
        return _lead(self.timing) + float(np.max(spans)) + _tail(self.timing)

    def layout(self, spans, g, TE, dt, n_t):
        schedule = RFSchedule([RFEvent(_dur(self.timing, "t_prep"), 90, 'Mz→Mxy',
                                       duration_s=_dur(self.timing, "t_excite"))])
        cap = None
        if self.timing is not None:
            lead = _lead(self.timing)
            cap = _steps_before(TE - _tail(self.timing), dt) - _first_step_at(lead, dt)
        return schedule, [[("start", _lead(self.timing), 1.0, cap)] for _ in spans]


class PreparedEchoTrain:
    """A diffusion preparation, then a refocusing train: what a diffusion-weighted fast spin echo plays.

    90 at 0 and the preparation's 180 at ``te_prep / 2``, with an encoding block either side of it exactly as
    :class:`SpinEcho` places them, so the preparation's own echo forms at ``te_prep``; then ``n_echoes``
    refocusing pulses of ``beta_deg`` about ``refocus_axis_deg``, one every ``TE_echo``, so an echo forms
    halfway between consecutive pulses. ``TE_prep`` is the preparation's own echo, as a
    :class:`StimulatedEcho`'s ``TM`` is its own mixing time, and the last echo is then
    ``TE_prep + n_echoes * TE_echo``.

    The train carries no gradient of its own. What separates its coherence pathways is the crusher the
    builder declares -- the voxel-scale spoiler, which is what a train really uses and what a micron cell
    cannot produce geometrically.

    Every echo lands on a sample by construction (:meth:`n_t_of`), because a refocusing pulse half a sample
    off centre leaves its crusher unbalanced and dephases the echo the crusher exists to keep.
    """

    def __init__(self, n_echoes, TE_echo, TE_prep, gap, prep_flip_deg=180.0, beta_deg=180.0,
                 refocus_axis_deg=90.0, timing=None):
        self.n_echoes, self.TE_echo, self.TE_prep = int(n_echoes), float(TE_echo), float(TE_prep)
        self.gap = gap                    # callable (m, g) -> seconds between the preparation's two blocks
        self.prep_flip, self.beta = float(prep_flip_deg), float(beta_deg)
        self.refocus_axis = float(refocus_axis_deg)
        self.timing = timing

    def te_min(self, spans, g):
        prep = SpinEcho(self.gap, self.timing).te_min(spans, g)
        if self.TE_prep < prep - _EPS_T:
            raise ValueError(f"TE_prep = {self.TE_prep*1e3:.3f} ms is below the {prep*1e3:.3f} ms the "
                             "preparation's blocks, their gap and its pulse windows need")
        return self.TE_prep + self.n_echoes * self.TE_echo

    def n_t_of(self, n_t_per_echo):
        """``n_t_per_echo`` samples fill each train interval and the preparation is a whole number of them, so
        every echo lands on a sample. A preparation this grid cannot express is refused, not rounded."""
        dt = self.TE_echo / int(n_t_per_echo)
        n_prep = self.TE_prep / dt
        if abs(n_prep - round(n_prep)) > 1e-6:
            raise ValueError(
                f"the preparation's echo at {self.TE_prep*1e3:.4f} ms is {n_prep:.4f} samples on this grid "
                f"({dt*1e6:.3f} us, {int(n_t_per_echo)} per {self.TE_echo*1e3:.3f} ms interval), and an echo "
                "must fall on a sample. Choose n_t_per_echo so the preparation is a whole number of them.")
        return int(round(n_prep)) + self.n_echoes * int(n_t_per_echo) + 1

    def windows(self):
        return EchoTrain(self.n_echoes, self.TE_echo, beta_deg=self.beta,
                         refocus_axis_deg=self.refocus_axis, timing=self.timing).windows()

    def layout(self, spans, g, TE, dt, n_t):
        te_prep = self.TE_prep
        w = _dur(self.timing, "t_refocus")
        ev = [RFEvent(_dur(self.timing, "t_prep"), 90, 'Mz→Mxy', duration_s=_dur(self.timing, "t_excite")),
              RFEvent(te_prep / 2.0, self.prep_flip, 'refocus', axis_deg=self.refocus_axis, duration_s=w)]
        ev += [RFEvent(te_prep + (k + 0.5) * self.TE_echo, self.beta, 'refocus', axis_deg=self.refocus_axis,
                       duration_s=w) for k in range(self.n_echoes)]
        placements = []
        for m, span in enumerate(spans):
            gap = self.gap(m, g[m])
            placements.append([("end", te_prep / 2.0 - gap / 2.0, 1.0), ("start", te_prep / 2.0 + gap / 2.0, 1.0)])
        return RFSchedule(ev), placements


class EchoTrain:
    """90 about x at 0 and ``n_echoes`` refocusing pulses of ``beta_deg`` at ``(k + 1/2) TE`` about the axis
    ``refocus_axis_deg`` -- 90 (y) is Meiboom-Gill, where a flip-angle error self-corrects every second echo;
    0 (x) is Carr-Purcell, where it accumulates. A gradient, if any, fills every stretch the pulses and readouts
    leave, at one polarity or alternating per echo interval."""

    def __init__(self, n_echoes, TE, polarity="constant", beta_deg=180.0, refocus_axis_deg=90.0, timing=None):
        if polarity not in ("constant", "alternate"):
            raise ValueError(f"polarity must be 'constant' or 'alternate', got {polarity!r}")
        self.n_echoes, self.TE_echo = int(n_echoes), float(TE)
        self.polarity, self.beta, self.refocus_axis = polarity, float(beta_deg), float(refocus_axis_deg)
        self.timing = timing

    def te_min(self, spans, g):
        return self.n_echoes * self.TE_echo

    def n_t_of(self, n_t_per_echo):
        """The grid: ``n_t_per_echo`` samples per interval, every echo on a sample."""
        return self.n_echoes * int(n_t_per_echo) + 1

    def windows(self):
        """Gradient-off intervals over the train: each refocusing window, and the same dead time -- the larger of
        the lead-in and the readout tail -- on both sides of every echo, so the two halves of every interval
        carry the same gradient and every echo refocuses."""
        tm = self.timing
        if tm is None:
            return []
        d = max(tm.t_lead, tm.t_readout_pre_echo)
        w = []
        for k in range(self.n_echoes):
            c = (k + 0.5) * self.TE_echo
            w += [(k * self.TE_echo, k * self.TE_echo + d), (c - tm.t_refocus / 2.0, c + tm.t_refocus / 2.0),
                  ((k + 1) * self.TE_echo - d, (k + 1) * self.TE_echo)]
        return w

    def layout(self, spans, g, TE, dt, n_t):
        tm = self.timing
        schedule = RFSchedule([RFEvent(_dur(tm, "t_prep"), 90, 'Mz→Mxy', duration_s=_dur(tm, "t_excite"))] +
                              [RFEvent((k + 0.5) * self.TE_echo, self.beta, 'refocus', axis_deg=self.refocus_axis,
                                       duration_s=_dur(tm, "t_refocus"))
                               for k in range(self.n_echoes)])
        t = np.arange(n_t) * dt
        on = np.ones(n_t, bool)
        on[-1] = False                                                # the readout sample acts over nothing
        for a, b in self.windows():
            on &= ~((t >= a - _EPS_T) & (t <= b + _EPS_T))            # samples inside a window, as validate() reads it
        interval = np.minimum(np.floor(t / self.TE_echo + _EPS_T).astype(int), self.n_echoes - 1)
        pol = np.where(interval % 2 == 0, 1.0, -1.0) if self.polarity == "alternate" else np.ones(n_t)
        # one lobe per stretch, and a stretch never crosses an echo: each interval refocuses on its own
        key = np.where(on, pol * (interval + 1), 0.0)
        runs = []                                                     # (start, length, polarity) of each stretch
        i = 0
        while i < n_t:
            if key[i] == 0.0:
                i += 1
                continue
            j = i
            while j < n_t and key[j] == key[i]:
                j += 1
            runs.append(("at", i, float(np.sign(key[i])), j - i))
            i = j
        return schedule, [list(runs) for _ in spans]


# ── the driver ──────────────────────────────────────────────────────────────────────────────────────────────────

class _NoRoom(ValueError):
    """A block that does not fit whole where its layout puts it on this grid."""


def _place(G, m, i0, block):
    n = block.shape[0]
    if i0 < 0 or i0 + n > G.shape[1] - 1:                             # the last sample is the readout: not a step
        raise _NoRoom(f"an encoding block of {n} samples at sample {i0} runs off the {G.shape[1]}-sample grid")
    G[m, i0:i0 + n] += block


def _peak_q(G_eff, dirs, dt):
    """``(n_m,)`` 1/m: each row's q, the largest ``|q(t)|`` its built effective gradient reaches along the row's unit
    direction, ``q(t) = gamma / (2 pi) * sum_{k <= t / dt} G_eff[k] . d * dt`` -- a pulsed pair's lobe area, one lobe's
    of an oscillating train, a quarter period's of a cosine, each as built."""
    q = np.zeros(G_eff.shape[0])
    for m in range(G_eff.shape[0]):
        norm = float(np.linalg.norm(dirs[m]))
        if norm == 0.0:
            continue
        run = np.cumsum(G_eff[m] @ (dirs[m] / norm)) * dt
        q[m] = float(np.abs(run).max()) * GAMMA / (2.0 * np.pi)
    return q


#: The most whole steps an automatic ``TE`` adds to the floor the blocks set, so that every block fits whole.
_ROOM_STEPS = 8


def _lobes_along(g):
    """The lobes of one row's gradient along its direction, ``g`` ``(n_t,)``: the maximal runs of non-zero samples of
    one sign, as ``[(i0, i1), ...]`` sample indices (``g[i0:i1]``)."""
    s = np.sign(g)
    out, i, n = [], 0, len(g)
    while i < n:
        if s[i] == 0:
            i += 1; continue
        j = i + 1
        while j < n and s[j] == s[i]:
            j += 1
        out.append((i, j)); i = j
    return out


def pair_timings(G_eff, dirs, dt):
    """``(delta_built, Delta_built)``, each ``(n_m,)`` seconds: each row's pulsed pair as built, read off the samples of
    its effective gradient along the row's direction, ``g[k] = G_eff[k] . d`` held over ``[k dt, (k + 1) dt)`` and
    taken as the value at ``k dt`` between samples. A lobe is a maximal run of non-zero samples of one sign; the pair is
    the row's FIRST TWO lobes; each lobe's edges are where ``|g|`` crosses half that lobe's peak, by linear
    interpolation of ``g`` between the two samples about the crossing (a neighbouring sample of the opposite sign
    included: the line through zero between them). ``delta_built`` is the mean of the two lobes'
    half-amplitude widths, ``Delta_built`` the separation of their centres (centre to centre). NaN on a row with fewer
    than two lobes (a b = 0 row, a zero direction)."""
    G_eff = np.asarray(G_eff, np.float64)
    n_m = G_eff.shape[0]
    width, sep = np.full(n_m, np.nan), np.full(n_m, np.nan)
    for m in range(n_m):
        norm = float(np.linalg.norm(dirs[m]))
        if norm == 0.0:
            continue
        g = G_eff[m] @ (np.asarray(dirs[m], np.float64) / norm)
        lobes = _lobes_along(g)
        if len(lobes) < 2:
            continue
        edges = []
        for i0, i1 in lobes[:2]:
            # g in the lobe's own sign, off before and after the grid: a neighbour of the opposite sign is negative
            a = np.concatenate([[0.0], np.sign(g[i0]) * g, [0.0]])
            run = a[i0 + 1:i1 + 1]
            h = 0.5 * run.max()
            up = i0 + int(np.argmax(run >= h))                            # first sample at or above half: index into g
            dn = i0 + len(run) - 1 - int(np.argmax(run[::-1] >= h))       # last one
            t_up = (up - 1 + (h - a[up]) / (a[up + 1] - a[up])) * dt      # between g[up - 1] (a[up]) and g[up]
            t_dn = (dn + (a[dn + 1] - h) / (a[dn + 1] - a[dn + 2])) * dt  # between g[dn] (a[dn + 1]) and g[dn + 1]
            edges.append((t_up, t_dn))
        (u1, d1), (u2, d2) = edges
        width[m] = 0.5 * ((d1 - u1) + (d2 - u2))
        sep[m] = 0.5 * (u2 + d2) - 0.5 * (u1 + d1)
    return width, sep


def assemble(assembler, *, gradient_directions, span, bvalues=None, gradient_strengths=None, sample=None,
             fill=None, TE=None, n_t=1000, slew_rate=np.inf, family, build_spec, encoding=None, state_q=False,
             state_pair=False, timing=None):
    """Build a validated :class:`ScannerSequence` from an assembler and a shape.

    ``span(m, g)`` is row ``m``'s block duration at amplitude ``g``; ``sample(m, g, dt)`` its unit block, ``(n,)``
    along the row's direction or ``(n, 3)``; ``fill(m, g, dt, n)`` a unit block of ``n`` samples for an assembler
    that hands out stretches to fill (the echo train). Exactly one of ``bvalues`` / ``gradient_strengths`` sets
    the amplitude; ``encoding(g, TE, te_min)`` returns the family's extra ``Encoding`` fields, ``g`` the amplitude the ramps were
    built for; ``state_q`` states
    ``Encoding.qvalues``, the largest ``|q(t)|`` each row's built effective gradient reaches along its direction;
    ``state_pair`` states ``Encoding.delta_built`` and ``Encoding.Delta_built``, the pulsed pair as built
    (:func:`pair_timings`), for a family whose encoding is one pair of lobes.

    ``slew_rate`` is the limit the shapes' ramps were built for. When it is finite the returned sequence is
    checked against it (:meth:`ScannerSequence.check_slew`), and an automatic ``TE`` (``TE=None``, on an assembler
    whose grid is not fixed by its echoes) is the floor the blocks set extended by the fewest whole steps of the
    grid, ``TE = te_min (n_t - 1) / (n_t - 1 - k)``, at which every block is placed whole (a ramped block keeps
    every sample inside its span, so it can be one step longer than the floor's room); ``Encoding.minimum_te``
    is that floor.

    With ``bvalues`` the amplitude is iterated (up to ``_B_ITER`` builds, until the b-scale is within ``_B_RTOL`` of 1)
    and the sequence returned is the last build whose final scale to the target b does not exceed ``1 + _B_RTOL`` on
    any row, scaled: its ramps were made for an amplitude at least the one it plays, so it holds the slew, and its b
    is the target. ``Encoding`` states that build's ramps.
    """
    dirs = np.asarray(gradient_directions, dtype=np.float64)
    if dirs.ndim == 1:
        dirs = dirs[None]
    n_m = dirs.shape[0]
    if bvalues is not None and gradient_strengths is not None:
        raise ValueError("give bvalues= or gradient_strengths=, not both")
    slew, square = resolve_slew(slew_rate)
    b_target = None if bvalues is None else np.broadcast_to(np.asarray(bvalues, np.float64), (n_m,)).copy()
    if b_target is not None and np.any(b_target < 0):
        raise ValueError("bvalues must be non-negative")
    if gradient_strengths is not None:
        g = np.broadcast_to(np.asarray(gradient_strengths, np.float64), (n_m,)).copy()
        if np.any(g < 0):
            raise ValueError("gradient_strengths must be non-negative")
    else:
        g = np.zeros(n_m)                                             # the square unit shape seeds the iteration

    def _lay(g, amp, te, n, whole):
        dt = te / (n - 1)
        schedule, placements = assembler.layout(spans_of(g), g, te, dt, n)
        G = np.zeros((n_m, n, 3), dtype=np.float64)
        for m in range(n_m):
            truncated_end = None  # (cap, capped block) from a same-row "end" placement that a following
                                   # "start" at the SAME cap mirrors, rather than slicing independently
                                   # (dmipy-replay#645 -- see the note below)
            for kind, where, factor, *rest in placements[m]:
                blk = fill(m, g[m], dt, rest[0]) if kind == "at" else sample(m, g[m], dt)
                blk = np.asarray(blk, dtype=np.float64)
                if blk.ndim == 1:
                    blk = blk[:, None] * dirs[m]
                cap = rest[0] if kind != "at" and rest else None            # dmipy-replay#645: margin-safe length
                if cap is not None and cap < blk.shape[0]:
                    if whole or not assembler.truncates_to_margin:
                        raise _NoRoom(f"row {m}: an encoding block of {blk.shape[0]} samples has room for {cap} "
                                      f"between the budget's lead-in and readout tail")
                    if cap < 1:
                        raise ValueError(f"row {m}: the budget's lead-in / readout-tail margin leaves no room "
                                         f"on the grid (dt = {dt*1e3:.4f} ms) for this encoding block")
                    if kind == "end":
                        blk = blk[-cap:]                                    # drop the half farther from the 180
                        truncated_end = (cap, blk)
                    elif truncated_end is not None and truncated_end[0] == cap:
                        # The SAME shape, independently sampled, would be sliced from its OTHER end here
                        # (`blk[:cap]`) -- correct for which half stays nearest the 180, but the raw shape need
                        # not be palindromic at this (span, dt), so that independent slice's sum need not equal
                        # the paired "end" block's, breaking q(TE) = 0 by exactly this kind of margin
                        # truncation. The exact time-reverse of the paired block has the identical sum (and
                        # values) by construction, so mirroring it instead keeps the cancellation exact.
                        blk = truncated_end[1][::-1]
                    else:
                        blk = blk[:cap]
                elif kind == "end":
                    truncated_end = None
                if kind == "at":
                    i0 = int(where)
                elif kind == "start":
                    i0 = _first_step_at(where, dt)
                else:
                    i0 = _steps_before(where, dt) - blk.shape[0]
                _place(G, m, i0, factor * amp[m] * blk)
        sign = schedule.sign(np.arange(n) * dt)[None, :, None]
        return G, dt, schedule, sign

    def spans_of(g):
        return np.array([float(span(m, g[m])) for m in range(n_m)])

    def _build(g, amp):
        te_min = assembler.te_min(spans_of(g), g)
        te = te_min if TE is None else float(TE)
        if te < te_min - _EPS_T:
            raise ValueError(f"TE = {te*1e3:.3f} ms is below the {te_min*1e3:.3f} ms this encoding needs "
                             f"(its blocks, their gap and the pulse / readout windows)")
        n = int(n_t) if not hasattr(assembler, "n_t_of") else assembler.n_t_of(n_t)
        if TE is not None or square or hasattr(assembler, "n_t_of"):
            return (*_lay(g, amp, te, n, whole=False), te, te_min)
        for k in range(min(_ROOM_STEPS, n - 3) + 1):
            te_k = te_min if k == 0 else te_min * (n - 1) / (n - 1 - k)
            try:
                return (*_lay(g, amp, te_k, n, whole=True), te_k, te_min)
            except _NoRoom as e:
                last = e
        raise ValueError(f"the encoding blocks do not fit whole on the {n}-sample grid within {_ROOM_STEPS} steps "
                         f"of TE = {te_min*1e3:.4f} ms ({last}): give TE, or a finer grid (n_t)")

    if b_target is None:
        G, dt, schedule, sign, te, te_min = _build(g, g)
        g_ramp = g
    else:
        G, dt, schedule, sign, te, te_min = _build(np.zeros(n_m), np.ones(n_m))   # square unit blocks seed the amplitude
        b_unit = b_from_gradient(G * sign, dt)
        g = np.where((b_target > 0) & (b_unit > 0), np.sqrt(b_target / np.where(b_unit > 0, b_unit, 1.0)), 0.0)
        # The ramps follow the amplitude. The declared b IS the numeric b of the effective gradient the walk
        # integrates, so the sequence returned is the last build scaled to the target b; that build is the last
        # one whose scale does not raise any row above the amplitude its ramps were made for (a ramp of g / slew
        # played at a higher amplitude would step faster than the slew). Where the iteration settles this is the
        # last build; where a block's sample count or the automatic TE flips between two builds it is the one
        # of the two that scales down.
        kept = None
        for _ in range(_B_ITER):
            built = _build(g, g)
            b_num = b_from_gradient(built[0] * built[3], built[1])
            live = (b_target > 0) & (b_num > 0)
            scale = np.where(live, np.sqrt(b_target / np.where(b_num > 0, b_num, 1.0)), 1.0)
            if np.all(scale <= 1.0 + _B_RTOL):
                kept = (built, g, scale)
            g = g * scale
            if np.all(np.abs(scale - 1.0) < _B_RTOL):
                break
        if kept is None:
            raise ValueError(f"the amplitude for b = {b_target.tolist()} did not settle in {_B_ITER} builds at a "
                             f"scale that keeps the ramps within the slew limit")
        (G, dt, schedule, sign, te, te_min), g_ramp, scale = kept
        G = G * scale[:, None, None]
        g = g_ramp * scale
    b_out = b_from_gradient(G * sign, dt) if b_target is None else b_target
    extra = {} if encoding is None else dict(encoding(g_ramp, te, te_min))     # the ramps as played: g_ramp / slew
    q = _peak_q(G * sign, dirs, dt) if state_q else None
    if state_pair:
        stored = np.asarray(G, np.float32).astype(np.float64)           # the samples the sequence holds (float32)
        extra["delta_built"], extra["Delta_built"] = pair_timings(stored * sign, dirs, dt)
    enc = Encoding(bvalues=b_out, gradient_directions=dirs, TE=np.full(n_m, te), qvalues=q, gradient_strengths=g,
                   minimum_te=te_min, te_auto=TE is None, **extra)
    seq = ScannerSequence(G=G, dt=dt, rf=schedule, timing=timing, family=family, encoding=enc,
                          build_spec=build_spec)
    if not square:
        seq.check_slew(slew)
    return seq.validate()
