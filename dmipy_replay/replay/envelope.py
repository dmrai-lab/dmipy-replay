"""A replay pack's guarantee, :class:`Envelope`: the region of acquisitions and tissues its certificate covers, in
physical terms.

The certificate of a pack is a set of bounded terms (the band truncation, the block quadrature, the codec's
quantisation, the float32 rounding, the field's quadrature and read, the contact's) plus the Monte-Carlo floor. Read
term by term, each depends on the acquisition through a waveform's temporal band and its per-axis amplitude alone,
and on the field and tissue through the field strengths, the susceptibility scale, the refocusing depth and the
surface relaxivity. Those are the fields of an :class:`Envelope`; nothing else enters. A scanner name is a preset
that produces one (:func:`envelope_of`), recorded as its :attr:`~Envelope.name`.

A pack's header states its envelope in ``replay_envelope`` (RPK.md 11): ``acquisition.bound = "envelope"`` with
``name``, ``band_hz``, ``G_max``, ``B0_list`` and, for a field channel, ``max_refocusing_pulses``; ``tissue`` with
``rho2_over_D_max`` (the contact bound over the walk's diffusivity) and ``chi_max``. :meth:`Envelope.from_header` reads
it, and also reads a header that states a scanner class (``bound = "scanner_class"`` with ``scanner``, ``G_max``,
``slew_max``) by the rule :data:`CLASS_HEADER_RULE`.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Optional

import numpy as np

__all__ = ["Envelope", "envelope_of", "ENVELOPE_BOUND", "CLASS_HEADER_BOUND", "CLASS_HEADER_RULE"]

#: ``replay_envelope.acquisition.bound`` of a header that states its :class:`Envelope`.
ENVELOPE_BOUND = "envelope"

#: ``replay_envelope.acquisition.bound`` of a header that states a scanner class: ``scanner``, ``G_max`` (T/m, per
#: axis) and ``slew_max`` (T/m/s), the stored band in ``compression.temporal_bandwidth_hz``.
CLASS_HEADER_BOUND = "scanner_class"

#: The rule an :class:`Envelope` is read by from a scanner-class header, version 1:
#:
#: * ``band_hz`` is the pack's stored band, ``compression.temporal_bandwidth_hz`` (``K / (2 T)`` per window). The
#:   truncation term of such a pack is bounded per waveform at its stored bands, so the stored band is the band the
#:   certificate covers; the catalogue is not read, so the envelope does not move when the catalogue does.
#: * ``G_max`` is ``acquisition.G_max``, the bound the quadrature, quantisation and rounding terms were read at.
#: * ``name`` is ``acquisition.scanner``; ``B0_list``, ``max_refocusing_pulses``, ``tissue.chi_max`` and
#:   ``tissue.rho2_over_D_max`` are read as stated.
#: * ``slew_max`` is not read: no term of the certificate depends on it. A waveform's slew enters only through the band
#:   its ramps need, which the band check holds directly.
CLASS_HEADER_RULE = "scanner_class/1"


def _finite(x, what, *, positive=False):
    v = float(x)
    if not np.isfinite(v) or v < 0.0 or (positive and v == 0.0):
        raise ValueError(f"an envelope's {what} is a finite {'positive' if positive else 'non-negative'} number; got {x!r}")
    return v


@dataclass(frozen=True)
class Envelope:
    """The acquisitions and tissues a replay pack is certified for, in physical terms. Every field is stated; ``None``
    states that the pack certifies no such term (no field channel, no contact channel), never a value to be filled in.

    * ``name``: provenance only -- the preset the envelope came from (:func:`envelope_of`) or a label the producer
      gives. No check reads it.
    * ``band_hz`` (Hz): the temporal band. A waveform is inside when the band it needs on the pack
      (:meth:`~dmipy_replay.replay.replay.ReplayPack.waveform_band`, at the pack's truncation budget) is at most this.
      A pack stores ``K = ceil(2 band_hz T)`` bands per window of ``T`` seconds (:meth:`mode_count`) or more.
    * ``G_max`` (T/m): the largest gradient per axis, played by the coils.
    * ``B0_list`` (T): the field strengths the field tier is read at; the certificate covers ``[0, max(B0_list)]``
      (the field term is linear in ``B0 chi``). Empty when no field term is certified.
    * ``chi_max``: the largest ``|chi_iso| + |chi_aniso|`` the field tier is certified at.
    * ``rho2_max`` (m/s): the largest surface relaxivity the contact tier is certified at. A header states it over the
      walk's diffusivity, ``rho2_over_D_max``.
    * ``max_refocusing_pulses``: the refocusing depth the field channel is certified for (a request with a field term
      and more refocusing pulses is outside).
    """
    name: str
    band_hz: float
    G_max: float
    B0_list: tuple
    chi_max: Optional[float]
    rho2_max: Optional[float]
    max_refocusing_pulses: Optional[int]

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name:
            raise ValueError(f"an envelope's name is a non-empty string (its provenance); got {self.name!r}")
        object.__setattr__(self, "band_hz", _finite(self.band_hz, "band_hz (Hz)", positive=True))
        object.__setattr__(self, "G_max", _finite(self.G_max, "G_max (T/m per axis)"))
        if self.B0_list is None or isinstance(self.B0_list, (str, bytes)) or np.ndim(self.B0_list) != 1:
            raise ValueError(f"an envelope's B0_list is a sequence of field strengths in tesla (empty for none); got "
                             f"{self.B0_list!r}")
        object.__setattr__(self, "B0_list", tuple(_finite(b, "B0_list entry (T)") for b in self.B0_list))
        if self.chi_max is not None:
            object.__setattr__(self, "chi_max", _finite(self.chi_max, "chi_max", positive=True))
        if self.rho2_max is not None:
            object.__setattr__(self, "rho2_max", _finite(self.rho2_max, "rho2_max (m/s)"))
        if self.max_refocusing_pulses is not None:
            n = self.max_refocusing_pulses
            if isinstance(n, bool) or int(n) != n or int(n) < 0:
                raise ValueError(f"an envelope's max_refocusing_pulses is a non-negative integer; got {n!r}")
            object.__setattr__(self, "max_refocusing_pulses", int(n))

    @property
    def B0_max(self):
        """The largest field strength (T) the field tier is certified up to, or None when ``B0_list`` is empty."""
        return max(self.B0_list) if self.B0_list else None

    def mode_count(self, T):
        """The bands ``K = ceil(2 band_hz T)`` a window of ``T`` seconds needs to hold this envelope's band
        (:func:`~dmipy_replay.replay.storage.mode_count`)."""
        from .storage import mode_count
        return mode_count(T, self.band_hz)

    def replace(self, **changes):
        """This envelope with ``changes`` (field names) stated in place of its own."""
        return type(self)(**{**self.to_dict(), **changes})

    def to_dict(self):
        """Every field, JSON-ready (``B0_list`` a list)."""
        d = {f.name: getattr(self, f.name) for f in fields(self)}
        d["B0_list"] = list(d["B0_list"])
        return d

    @classmethod
    def from_dict(cls, d):
        """The envelope :meth:`to_dict` wrote. Every field must be present: a missing one is refused by name."""
        if isinstance(d, cls):
            return d
        names = [f.name for f in fields(cls)]
        missing = [n for n in names if n not in d]
        extra = sorted(set(d) - set(names))
        if missing or extra:
            raise ValueError(f"an envelope states exactly {names}; "
                             + (f"missing {missing}" if missing else "") + (f" unknown {extra}" if extra else ""))
        return cls(**{n: d[n] for n in names})

    def header_acquisition(self):
        """The envelope's part of ``replay_envelope.acquisition``: ``bound`` :data:`ENVELOPE_BOUND`, ``name``, ``band_hz``,
        ``G_max`` and ``B0_list``. A producer adds what the pack realised beside it (``max_refocusing_pulses`` of the
        field channel it stored, ``b_max``)."""
        return dict(bound=ENVELOPE_BOUND, name=self.name, band_hz=float(self.band_hz), G_max=float(self.G_max),
                    B0_list=[float(b) for b in self.B0_list])

    @classmethod
    def from_header(cls, meta):
        """The envelope a pack or layout header ``meta`` states (``replay_envelope``). A header with ``bound``
        :data:`ENVELOPE_BOUND` is read as stated; one with ``bound`` :data:`CLASS_HEADER_BOUND` by
        :data:`CLASS_HEADER_RULE`. Any other header -- no bound, or a battery of b values, which is no guarantee for a
        waveform -- is refused by name, as is one missing a field."""
        env = meta.get("replay_envelope") or {}
        acq = env.get("acquisition") or {}
        tissue = env.get("tissue") or {}
        ident = meta.get("id")
        bound = acq.get("bound")
        if bound == ENVELOPE_BOUND:
            missing = [k for k in ("name", "band_hz", "G_max", "B0_list") if acq.get(k) is None]
            if missing:
                raise ValueError(f"pack {ident!r} states an envelope without replay_envelope.acquisition."
                                 f"{', '.join(missing)}")
            name, band = acq["name"], acq["band_hz"]
        elif bound == CLASS_HEADER_BOUND:
            missing = [k for k in ("scanner", "G_max") if acq.get(k) is None]
            if missing:
                raise ValueError(f"pack {ident!r} declares the scanner class {acq.get('scanner')!r} without "
                                 f"replay_envelope.acquisition.{', '.join(missing)}")
            band = (meta.get("compression") or {}).get("temporal_bandwidth_hz")
            if band is None:
                raise ValueError(f"pack {ident!r} states the scanner class {acq['scanner']!r} and no stored band "
                                 f"(compression.temporal_bandwidth_hz), which the {CLASS_HEADER_RULE} rule reads as its "
                                 "envelope's band")
            name = str(acq["scanner"])
        else:
            raise ValueError(f"pack {ident!r} states no envelope (replay_envelope.acquisition.bound = {bound!r}): its "
                             "certificate is a battery of b values, which is no guarantee for a waveform. Walk the "
                             "substrate again with an envelope (walk(spec, ..., envelope=envelope_of(scanner, ...)))")
        rho = tissue.get("rho2_over_D_max")
        rho2_max = None
        if rho is not None:
            D = (meta.get("walk_params") or {}).get("diffusivity")
            if D is None:
                raise ValueError(f"pack {ident!r} states a contact bound rho2_over_D_max and no diffusivity "
                                 "(walk_params.diffusivity) to read it in m/s by")
            rho2_max = float(rho) * float(D)
        return cls(name=name, band_hz=band, G_max=acq["G_max"], B0_list=tuple(acq.get("B0_list") or ()),
                   chi_max=tissue.get("chi_max"), rho2_max=rho2_max,
                   max_refocusing_pulses=acq.get("max_refocusing_pulses"))


def envelope_of(scanner, *, B0_list=(), chi_max=None, rho2_max=None, max_refocusing_pulses=None):
    """The :class:`Envelope` of a scanner class from the catalogue (``acquisition/scanner_constants.json``):
    ``band_hz`` the class's certified band (``band.certified_band_hz``: the band every waveform the class can deliver
    needs, the closed-form certificate at the class's amplitude and slew) times
    :data:`~dmipy_replay.replay.storage.STORAGE_BAND_MARGIN`, ``G_max`` its gradient amplitude, ``name`` the name it was
    asked by (any of the catalogue's names for it). This is the one place a scanner's band rule is applied.

    The field and tissue bounds are not a scanner's: ``B0_list``, ``chi_max``, ``rho2_max`` and
    ``max_refocusing_pulses`` are stated here, and their defaults state that no such term is certified (an empty
    ``B0_list``, ``None``). A producer refuses a walk whose channels need a bound the envelope does not state.

    A class the catalogue holds no certified band for is refused by name: the band is a measured property of the class,
    never a guess from its amplitude or slew; state an :class:`Envelope` with its band instead."""
    from ..acquisition.scanners import ScannerLimits
    from .storage import STORAGE_BAND_MARGIN
    lim = ScannerLimits.of(scanner)
    if lim.certified_band_hz is None:
        raise ValueError(f"the catalogue holds no certified band for scanner class {lim.name!r}: its envelope's band is "
                         "measured and cited before a pack is stored for it (band.certified_band_hz); state an "
                         "Envelope(band_hz=...) for a band of your own")
    return Envelope(name=(scanner if isinstance(scanner, str) else str(lim.name)), band_hz=float(STORAGE_BAND_MARGIN) * float(lim.certified_band_hz),
                    G_max=float(lim.G_max), B0_list=tuple(B0_list), chi_max=chi_max, rho2_max=rho2_max,
                    max_refocusing_pulses=max_refocusing_pulses)
