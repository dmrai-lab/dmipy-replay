"""The storage rule's band: how many sine bands a window of a pack holds for a scanner class (#643). One derivation,
read by the producer that stores a pack and by the reader that plans what part of one to fetch."""
from __future__ import annotations

import numpy as np

from ..acquisition.scanners import ScannerLimits

__all__ = ["STORAGE_BAND_MARGIN", "storage_band_hz", "mode_count"]

STORAGE_BAND_MARGIN = 1.4  #: the storage rule's margin over a class's certified band (the paper's K = 256 per 100 ms for the human classes)


def storage_band_hz(scanner):
    """The band a pack built for ``scanner`` is stored at (Hz): the class's certified band -- the highest frequency any
    waveform the class can deliver needs of a pack, the catalogue's ``band.certified_band_hz`` -- times
    :data:`STORAGE_BAND_MARGIN`. A class the catalogue holds no certified band for is refused by name: the band is a
    measured property of the class (the closed-form certificate of the replayable-MC paper, ``figures/fig_band_data.py``),
    never a guess from its amplitude or slew."""
    lim = ScannerLimits.of(scanner)
    if lim.certified_band_hz is None:
        raise ValueError(f"the catalogue holds no certified band for scanner class {lim.name!r}: a pack cannot be stored for it "
                         f"until band.certified_band_hz is measured and cited (the replayable-MC paper's fig_band_data on the class's "
                         f"envelope); pass K= or temporal_bandwidth_hz= to store at a stated band instead")
    return float(STORAGE_BAND_MARGIN) * float(lim.certified_band_hz)


def mode_count(T, scanner=None, *, temporal_bandwidth_hz=None):
    """The band count ``K = ceil(2 f T)`` of a window of ``T`` seconds stored at ``f`` = :func:`storage_band_hz` of
    ``scanner`` (or a stated ``temporal_bandwidth_hz``): the one derivation every producer reads (#643)."""
    f = float(temporal_bandwidth_hz) if temporal_bandwidth_hz is not None else storage_band_hz(scanner)
    return max(2, int(np.ceil(2.0 * f * float(T))))
