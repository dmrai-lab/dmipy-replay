"""The amplitude of a sequence's readout pathway, derived from its RF schedule by the extended phase graph of
:mod:`dmipy_analytic.epg`: what a replay of a :class:`~dmipy_replay.acquisition.scanner_sequence.ScannerSequence`
multiplies its echo by, at the sequence's own flips and at any transmit scale."""
from __future__ import annotations

import numpy as np

__all__ = ["pathway_weight", "transmit_amplitude"]


def pathway_weight(sequence):
    """The amplitude of the coherence pathway ``sequence``'s readout is, derived from its RF schedule.

    Every case here is ENUMERATED rather than written down, and the cases are exactly those a single number
    can express:

    * no refocusing and no store -- a gradient echo, the magnetisation the excitation ``alpha`` tips,
      ``sin(alpha)``, 1 at a 90;
    * one refocused echo at flip ``beta`` -- ``sin(alpha) sin^2(beta/2)``, one pathway, and 1 at a 90 and a
      perfect 180;
    * a store and a recall -- the stimulated echo, :func:`dmipy_analytic.epg.ste_amplitude`, which a real
      sequence isolates by crushing the rest;
    * a train of perfect 180s -- every echo is the same single pathway, ``sin(alpha)``.

    The excitation is the pulse labelled ``excite``, else the schedule's first pulse, at the flip it carries: a
    transmit scale that acts on every pulse of a sequence scales the excitation too, and its ``sin`` is part of
    the amplitude (dmipy-replay#391).

    A train whose refocusing pulses are NOT 180 is **refused**. Its echoes differ from one another (a six-echo
    train at 120 degrees runs 0.75, 0.94, 0.84, 0.86, 0.88, 0.86) and each is a sum over several pathways, so
    no single amplitude describes the readout: that needs the pathway sum (dmipy-replay#307). Returning 1 there
    would be wrong by up to 25 per cent on the first echo alone.

    Duck-typed on ``stimulated_echo``, ``rf`` and ``readout``, so it reads a
    :class:`~dmipy_replay.acquisition.scanner_sequence.ScannerSequence` without importing one.
    """
    from dmipy_analytic.epg import Pulse, Schedule, Winding, enumerate_pathways, ste_amplitude

    from .rf import role_of
    rf = getattr(sequence, "rf", None) or ()
    if getattr(sequence, "stimulated_echo", False):
        flips = {}
        for e in rf:
            role = role_of(e)
            if role in ("excite", "store", "recall") and role not in flips:
                flips[role] = float(e.flip_deg)
        if set(flips) != {"excite", "store", "recall"}:
            raise ValueError("a stimulated echo needs an excite, a store and a recall in its RF schedule; "
                             f"this one labels {sorted(flips)}")
        return ste_amplitude(flips["excite"], flips["store"], flips["recall"])

    excite = [float(e.flip_deg) for e in rf if role_of(e) == "excite"] or [float(e.flip_deg) for e in rf if float(e.flip_deg) != 0.0]
    alpha = excite[0] if excite else 90.0
    tipped = float(abs(np.sin(np.radians(alpha))))            # what the excitation puts in the plane
    refocus = [float(e.flip_deg) for e in rf if role_of(e) == "refocus"]
    imperfect = [b for b in refocus if abs(b - 180.0) > 1e-6]
    if not imperfect:
        return tipped                   # no refocusing at all, or every pulse a perfect 180: one pathway, whole
    n_readout = len(tuple(getattr(sequence, "readout", ()) or ()))
    if n_readout > 1 or len(refocus) > 1:
        raise ValueError(
            f"this schedule refocuses {len(refocus)} time(s) at {sorted(set(imperfect))} degrees and reads "
            f"{n_readout} echo(es): its readout is a SUM over several coherence pathways, of different "
            "amplitudes at each echo, so no single amplitude describes it. The pathway sum is dmipy-replay#307 "
            "(dmipy_analytic.epg.enumerate_pathways gives the terms); on the vector-Bloch route it also needs the crusher that "
            "route ignores, dmipy-replay#305. Use a perfect 180, or a single refocused echo.")
    sch = Schedule((Pulse(alpha), Winding(+1, 1.0), Pulse(imperfect[0], 90.0), Winding(+1, 1.0, readout=True)))
    return float(abs(sum(p.eta for p in enumerate_pathways(sch, threshold=1e-12))))


def transmit_amplitude(sequence, kappa):
    """The amplitude of ``sequence``'s readout pathway when every pulse plays at ``kappa`` times its flip
    (:func:`pathway_weight` of the schedule with each flip scaled), one per entry of ``kappa`` (``(n,)``).

    A transmit scale acts on the pulses and on nothing else, so for a crushed single echo -- the one pathway a
    diffusion preparation's own lobes isolate -- it is a factor on the voxel's signal and no new replay:
    ``sin(kappa alpha) sin^2(kappa beta / 2)`` for a spin echo, ``0.5 sin(kappa a1) sin(kappa a2) sin(kappa a3)``
    for a stimulated echo. A pulse given as an envelope with no flip is refused: its nutation at another scale is
    the Bloch equation's, not a product."""
    from dataclasses import replace
    from .rf import RFSchedule
    rf = getattr(sequence, "rf", None) or ()
    if any(e.flip_deg is None for e in rf):
        raise ValueError("a pulse given by its envelope alone has no flip to scale; its amplitude at another transmit "
                         "scale is a Bloch propagation, not a pathway product")
    k = np.atleast_1d(np.asarray(kappa, dtype=np.float64))
    uniq, inv = np.unique(k, return_inverse=True)
    amp = np.array([pathway_weight(replace(sequence, rf=RFSchedule(replace(e, flip_deg=float(e.flip_deg) * float(u)) for e in rf)))
                    for u in uniq])
    return amp[np.asarray(inv).reshape(-1)]


def __getattr__(name):
    from .._moved import refuse
    raise refuse(__name__, name)
