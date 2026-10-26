"""DIPY's voxel simulator, played from replay packs: :func:`multi_pack` is ``dipy.sims.voxel.multi_tensor`` with a
pack where the tensor was.

The call keeps DIPY's conventions -- a ``GradientTable`` (b in s/mm^2, its ``big_delta`` / ``small_delta`` in
seconds), ``angles`` as polar/azimuth pairs in degrees or unit vectors, ``fractions`` in percent, ``S0``, an
``snr`` with Rician noise of sigma ``S0 / snr`` -- and returns what ``multi_tensor`` returns, ``(S, sticks)``. What a
pack adds are the replay knobs: ``tissue`` (pool T2 / T1, surface relaxivity, susceptibility) and ``scanner`` (the
static field, along z of the frame the gradient table and ``angles`` share).

DIPY is not imported: the table is read by its attributes (``bvals``, ``bvecs``, ``big_delta``, ``small_delta``,
``b0_threshold``), so any object carrying them plays.
"""
from __future__ import annotations

import numpy as np

from .replay.replay import ReplayPack
from .sequences.btable import from_btable
from .spec.tissue import Tissue

__all__ = ["multi_pack", "sticks_of", "add_noise", "pack_licenses", "Tissue"]


def sticks_of(angles):
    """``(K, 3)`` unit vectors from DIPY's ``angles``: ``(K, 2)`` (polar from z, azimuth from x) in degrees, or
    ``(K, 3)`` vectors already."""
    a = np.atleast_2d(np.asarray(angles, dtype=float))
    if a.shape[1] == 3:
        return a / np.linalg.norm(a, axis=1, keepdims=True)
    if a.shape[1] != 2:
        raise ValueError(f"angles {a.shape}: (K, 2) polar/azimuth in degrees or (K, 3) unit vectors")
    th, ph = np.deg2rad(a[:, 0]), np.deg2rad(a[:, 1])
    return np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)], axis=1)


def add_noise(signal, snr, S0, *, rng=None):
    """Rician noise as DIPY's ``add_noise(..., noise_type='rician')``: two Gaussian channels of sigma ``S0 / snr``,
    the magnitude of the sum. ``snr=None`` returns the signal unchanged."""
    if snr is None:
        return signal
    rng = np.random.default_rng(rng)
    sigma = S0 / snr
    n1, n2 = rng.normal(0.0, sigma, np.shape(signal)), rng.normal(0.0, sigma, np.shape(signal))
    return np.sqrt((signal + n1) ** 2 + n2 ** 2)


def _pack(p):
    return p if isinstance(p, ReplayPack) else ReplayPack.load(p)


def pack_licenses(packs):
    """The terms each pack is published under (its header's ``pack_license``), one per pack, in order."""
    return [_pack(p).meta.get("pack_license") for p in packs]


def multi_pack(gtab, packs, *, angles, fractions, tissue=None, TE=None, scanner=None, S0=1.0, snr=None, rng=None,
               sequence=None):
    """The signal of a voxel mixing ``packs`` (a :class:`ReplayPack`, a path or an ``hf://`` URI each), pack ``k``'s
    axis along stick ``k`` with ``fractions[k]`` percent of the signal, as ``multi_tensor(gtab, mevals, S0, angles,
    fractions, snr)``. Returns ``(S, sticks)``: ``S`` of shape ``(n_volumes,)``, ``sticks`` ``(K, 3)``.

    The acquisition is the PGSE of ``gtab``'s b-values and directions with its ``small_delta`` and ``big_delta``
    (seconds) and the echo time ``TE``, or ``sequence`` (a :class:`~dmipy_replay.ScannerSequence`, e.g. from a
    Pulseq file) in place of the table. A table without its timing is refused, and so is a ``tissue`` without a
    ``TE``: the relaxation and the field a tissue switches on act over the echo time, which the table does not carry.

    ``tissue`` is one :class:`Tissue` for every pack or one per pack (``None`` entries for bare diffusion);
    ``scanner`` the static field (tesla, or a :class:`~dmipy_replay.ScannerLimits`). ``rng`` seeds the noise.
    """
    sticks = sticks_of(angles)
    f = np.asarray(fractions, dtype=float).reshape(-1)
    packs = list(packs) if isinstance(packs, (list, tuple)) else [packs]
    if not (len(packs) == len(f) == len(sticks)):
        raise ValueError(f"{len(packs)} packs, {len(f)} fractions and {len(sticks)} angles: one of each per pack")
    if not np.isclose(f.sum(), 100.0, atol=1e-6):
        raise ValueError(f"fractions are percent and sum to 100 (DIPY's convention); these sum to {f.sum():g}")
    tissues = list(tissue) if isinstance(tissue, (list, tuple)) else [tissue] * len(packs)
    if len(tissues) != len(packs):
        raise ValueError(f"{len(tissues)} tissues for {len(packs)} packs")
    if sequence is None:
        if any(t is not None for t in tissues) and TE is None:
            raise ValueError("a tissue sets relaxation or a field, which act over the echo time; the gradient table "
                             "does not carry it: state TE (seconds)")
        sequence = from_btable(gtab.bvals, gtab.bvecs, delta=getattr(gtab, "small_delta", None),
                               Delta=getattr(gtab, "big_delta", None), TE=TE,
                               b0_threshold=getattr(gtab, "b0_threshold", 50.0))
    elif TE is not None:
        raise ValueError("TE is the sequence's own: state it in the sequence, not beside it")
    S = sum(frac / 100.0 * _pack(p).replay(sequence, tissue=t, scanner=scanner, orientation=axis)
            for p, frac, axis, t in zip(packs, f, sticks, tissues))
    return add_noise(S0 * S, snr, S0, rng=rng), sticks
