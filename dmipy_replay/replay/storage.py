"""The storage rule: how many sine bands a window of a pack holds for a band in hertz (#643). One derivation, read by
the producer that stores a pack and by the reader that plans what part of one to fetch; a scanner class's band is
its :class:`~dmipy_replay.replay.envelope.Envelope`'s (:func:`~dmipy_replay.replay.envelope.envelope_of`)."""
from __future__ import annotations

import numpy as np

__all__ = ["STORAGE_BAND_MARGIN", "mode_count"]

#: the storage rule's margin over a class's certified band (the paper's K = 256 per 100 ms for the human classes);
#: applied once, by :func:`~dmipy_replay.replay.envelope.envelope_of`
STORAGE_BAND_MARGIN = 1.4


def mode_count(T, band_hz):
    """The band count ``K = ceil(2 f T)`` of a window of ``T`` seconds holding the band ``f = band_hz`` (Hz, an
    :attr:`~dmipy_replay.replay.envelope.Envelope.band_hz`): the one derivation every producer reads (#643)."""
    f = float(band_hz)
    if not (np.isfinite(f) and f > 0.0):
        raise ValueError(f"a band is a finite positive frequency in Hz; got {band_hz!r}")
    return max(2, int(np.ceil(2.0 * f * float(T) - 1e-9)))
