"""A b-value table as the toolkits write it -- b in s/mm^2, one direction per volume -- made into the acquisition a
replay plays.

A table names b and a direction per volume and nothing about the waveform that produced them, so the timing is stated
beside it: the pulse width ``delta`` and separation ``Delta`` of the PGSE every diffusion toolkit's table assumes,
and the echo time ``TE``. A table without them is refused, never completed from a default -- two acquisitions with
one table and different timings give different signals in any restricted or relaxing substrate, and the table cannot
tell them apart.
"""
from __future__ import annotations

import numpy as np

from .builders import pgse

__all__ = ["from_btable"]

#: the toolkits' b-value unit, s/mm^2, in SI s/m^2
S_PER_MM2 = 1e6


def from_btable(bvals, bvecs, *, delta, Delta, TE=None, b0_threshold=50.0):
    """The PGSE that plays this table: ``bvals`` in s/mm^2 (the DIPY / FSL / MRtrix unit), ``bvecs`` ``(n, 3)``
    directions in the frame the replay's ``orientation`` is stated in, ``delta`` the lobe width (at half amplitude;
    the ramps sit on top of it) and ``Delta`` the lobe separation, centre to centre, in seconds -- one number, or one
    per volume. ``TE`` is the echo time in seconds, ``None`` for the shortest that fits.

    A volume with ``b <= b0_threshold`` is a b = 0 volume (the toolkits' convention: a scanner's b0 is a few s/mm^2,
    never exactly 0) and plays with no diffusion gradient; every other volume's direction must be a unit vector to
    1e-2 (DIPY's own tolerance) and is normalised. ``delta`` or ``Delta`` ``None`` is refused: the table does not carry
    the timing and nothing here guesses it.
    """
    if delta is None or Delta is None:
        raise ValueError("a b-value table carries no timing: state the PGSE's delta and Delta (seconds) -- from the "
                         "protocol, a gradient table's big_delta / small_delta, or the scan's sidecar -- or replay "
                         "the acquisition itself (from_pulseq)")
    b = np.asarray(bvals, dtype=float).reshape(-1)
    g = np.asarray(bvecs, dtype=float)
    if g.shape == (3, b.size) and b.size != 3:                 # FSL's bvecs file is 3 x n
        g = g.T
    if g.shape != (b.size, 3):
        raise ValueError(f"bvecs {g.shape} does not match {b.size} b-values: (n, 3) or FSL's (3, n)")
    if np.any(b < 0):
        raise ValueError("a negative b-value")
    dw = b > b0_threshold
    norm = np.linalg.norm(g[dw], axis=1)
    bad = np.flatnonzero(np.abs(norm - 1.0) > 1e-2)
    if bad.size:
        rows = np.flatnonzero(dw)[bad]
        raise ValueError(f"volumes {rows[:8].tolist()} have b > {b0_threshold:g} s/mm^2 and a direction of norm "
                         f"{norm[bad][:8].round(4).tolist()}, not a unit vector")
    dirs = np.zeros_like(g)
    dirs[:, 0] = 1.0                                           # b = 0: any axis, zero amplitude
    dirs[dw] = g[dw] / norm[:, None]
    return pgse(dirs, delta, Delta, bvalues=np.where(dw, b, 0.0) * S_PER_MM2, TE=TE)
