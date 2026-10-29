"""What one acquisition leaves of every walker of a pack before any tissue or scanner is applied, and the one reduction
that turns it into a signal for a (tissue, scanner) pair: :class:`Primitives`. A pack's per-walker route, the columnar
image and the time-segment layout's image all read their tissue and scanner terms from
:meth:`Primitives.reduction_terms`, so a term belongs to the reduction once.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .planning import planned


@dataclass
class Primitives:
    """What one acquisition leaves of every walker before any tissue or scanner is applied: ``w`` the weights,
    ``phi`` the gradient phase ``(n, m)``, ``field_iso`` / ``field_aniso`` the gated path integrals under the field
    direction ``(n,)`` (None without the path channel), ``exposure_t2`` / ``exposure_t1`` the transverse and
    longitudinal time per pool ``(n, n_pools)`` (None without the compartment channel), ``contact`` / ``contact_t1``
    the boundary local time gated by the coherence and by its complement (on while stored along B0, e.g. a
    stimulated echo's mixing time) ``(n,)`` (None without the contact channel), ``D_walk`` the walk's diffusivity, and
    ``pathway`` the amplitude of the coherence pathway the acquisition's readout is
    (:func:`~dmipy_replay.acquisition.epg.pathway_weight`; 1 for a refocused echo, a stimulated echo's
    ``0.5 sin a1 sin a2 sin a3`` for a store-and-recall schedule), and ``voxel`` the voxel's factor per
    measurement (:meth:`~dmipy_replay.acquisition.scanner_sequence.ScannerSequence.voxel_factor`; required, since a
    default of 1 is wrong for an unbalanced encoding rather than merely absent), and ``susceptibility_field`` the
    pack's susceptibility field (``"present"`` or ``"absent"``, from its spec's pools; required: it decides what a tissue's chi may do, through
    :func:`~dmipy_replay.replay.replay.field_term`). A tissue and a
    scanner turn these into the per-walker weights and phases of :meth:`signals`, whose ensemble mean is
    :meth:`signal`. ``plan`` is the :class:`~dmipy_replay.replay.planning.ReadPlan` the bands and modes were read under;
    the signal of every pair carries it."""
    w: np.ndarray
    phi: np.ndarray
    field_iso: Optional[np.ndarray]
    field_aniso: Optional[np.ndarray]
    exposure_t2: Optional[np.ndarray]
    exposure_t1: Optional[np.ndarray]
    contact: Optional[np.ndarray]
    D_walk: Optional[float]
    voxel: np.ndarray                                          # required: there is no sane default (see __post_init__)
    susceptibility_field: str                                  # the pack's field: "present" or "absent"
    pathway: float = 1.0
    by_pool: object = field(repr=False, default=None)          # the pack's resolver of a per-pool value
    rho2_over_D_max: Optional[float] = None                     # the contact tier's envelope (RPK.md 8.7), None unbounded
    field_envelope: Optional[dict] = None                       # the field tier's {B0_max, B0_chi_max} (ReplayPack.field_envelope), None unbounded
    contact_t1: Optional[np.ndarray] = None                    # the SAME channel, gated by chi_parallel = active - chi (dmipy-sim#574)
    plan: object = None                                        # the ReadPlan of the bands and modes contracted (None: unplanned)

    def __post_init__(self):
        if self.voxel is None:
            raise ValueError(
                "Primitives.voxel is required: it is the acquisition's voxel factor per measurement "
                "(ScannerSequence.voxel_factor, 1 for a refocused encoding and the spoiler for an unbalanced one). "
                "A default of 1 is not safe, because for an unbalanced encoding it silently serves the un-crushed "
                "substrate signal instead of the voxel's")

    @property
    def n_pools(self):
        return None if self.exposure_t2 is None else int(self.exposure_t2.shape[1])

    def rates(self, tissue):
        """``(invT2, invT1, rho2_over_D, rho1_over_D)`` of a tissue on this pack's pools: per-pool rates (0 where the
        tissue declares no time, or ``inf``) and the two contact rates (0 without a relaxivity). A per-pool value
        is judged complete on the pack's spec, not on the pools this chunk of walkers labels."""
        t = tissue
        n = self.n_pools or 1
        invT2 = np.zeros(n); invT1 = np.zeros(n); rho2_D = 0.0; rho1_D = 0.0
        if t is not None and (t.T2 is not None or t.T1 is not None):
            if self.exposure_t2 is None:
                raise ValueError("T2 / T1 were given but the pack carries no compartment channel (C1)")
            for name, vals in (("T2", self.by_pool(t.T2, "T2")), ("T1", self.by_pool(t.T1, "T1"))):
                if vals is None:
                    continue
                if len(vals) < n:
                    raise ValueError(f"the compartment channel addresses pool id {n - 1} and the embedded spec "
                                     f"declares {len(vals)} pools: the pack is inconsistent")
                rate = 1.0 / np.asarray(vals, float)[:n]           # the spec may name more pools than the channel labels (a dry sheath)
                if name == "T2":
                    invT2 = rate
                else:
                    invT1 = rate
        if t is not None and t.D is not None and self.D_walk is not None and float(t.D) != float(self.D_walk):
            raise ValueError(
                f"the primitives were contracted on the walked grid, at D = {float(self.D_walk):.3g} m^2/s, and a "
                f"tissue at D = {float(t.D):.3g} reads the pack on another grid (dmipy-sim#289): every term "
                f"changes, not the surface divisor alone. Contract the view instead -- "
                f"pack.at_diffusivity(D).study(...) or .walker_primitives(...)")
        if t is not None and t.rho2 is not None and float(t.rho2) != 0.0:
            if self.contact is None:
                raise ValueError("surface relaxivity was requested but this pack carries no C2 channel")
            D = self.D_walk if t.D is None else t.D
            if D is None:
                raise ValueError("rho2 needs the walk's diffusivity: the pack did not record it, pass D=")
            rho2_D = float(t.rho2) / float(D)
            if self.rho2_over_D_max is not None and rho2_D > float(self.rho2_over_D_max) * (1.0 + 1e-12):
                raise ValueError(f"rho2 / D = {rho2_D:.4g} 1/m is beyond this pack's contact envelope, rho2 / D <= "
                                 f"{float(self.rho2_over_D_max):.4g} 1/m (replay_envelope.tissue.rho2_over_D_max)")
        if t is not None and t.rho1 is not None and float(t.rho1) != 0.0:
            # the SAME contact channel, gated by the complement (the longitudinal term; dmipy-sim#574)
            if self.contact_t1 is None:
                raise ValueError("a longitudinal surface relaxivity (rho1) was requested but this pack/layout "
                                 "carries no C2 channel gated by the stored period")
            D = self.D_walk if t.D is None else t.D
            if D is None:
                raise ValueError("rho1 needs the walk's diffusivity: the pack did not record it, pass D=")
            rho1_D = float(t.rho1) / float(D)
            if self.rho2_over_D_max is not None and rho1_D > float(self.rho2_over_D_max) * (1.0 + 1e-12):
                raise ValueError(f"rho1 / D = {rho1_D:.4g} 1/m is beyond this pack's contact envelope, rho2 / D <= "
                                 f"{float(self.rho2_over_D_max):.4g} 1/m (replay_envelope.tissue.rho2_over_D_max)")
        return invT2, invT1, rho2_D, rho1_D

    def field_scalars(self, tissue, scanner):
        """``(B0 chi_iso, B0 chi_aniso)`` for a pair as :func:`~dmipy_replay.replay.replay.field_term` resolves it on the
        pack's declared field, ``(0, 0)`` when that term is zero."""
        from .replay import field_term
        f = field_term(self.susceptibility_field, tissue, scanner, self.field_envelope)
        if not f.active:
            return 0.0, 0.0
        if self.field_iso is None:
            raise ValueError("this pair has a field term but these primitives carry no path channel (C3): contract "
                             "them from a pack read with its field modes")
        return f.B0 * f.chi_iso, f.B0 * f.chi_aniso

    def reduction_terms(self, tissue=None, scanner=None):
        """Every term that turns these primitives into a signal, resolved once for a pair: ``invT2`` / ``invT1`` the
        per-pool rates, ``rho2_over_D`` / ``rho1_over_D`` the two contact rates, ``a_iso`` / ``a_aniso`` the field
        scalars, ``amplitude`` the readout's pathway weight and ``voxel`` the per-measurement voxel factor.

        :meth:`signals` evaluates these on the host and
        :meth:`~dmipy_replay.replay.columnar.ColumnarPack.image_study` the same ones on the device, so a term belongs
        to the reduction once and reaches both. That device kernel had grown its own copy and left ``amplitude``
        out, which made a stimulated-echo study exactly ``1 / eta`` times the per-voxel ``walker_signals`` -- 2x
        for a 90/90/90 schedule (dmipy-sim#484)."""
        invT2, invT1, rho2_D, rho1_D = self.rates(tissue)
        a_i, a_a = self.field_scalars(tissue, scanner)
        return dict(invT2=invT2, invT1=invT1, rho2_over_D=rho2_D, rho1_over_D=rho1_D, a_iso=a_i, a_aniso=a_a,
                    amplitude=float(self.pathway), voxel=self.voxel)

    def signals(self, tissue=None, scanner=None):
        """``(w, ew, E)`` as :meth:`ReplayPack.walker_signals` gives them for this acquisition under the pair --
        the same numbers, since ``phi`` came from :func:`~dmipy_replay.replay.replay._band_phase`, ``E`` from
        :func:`~dmipy_replay.replay.replay._signal_factor` and the rest from :meth:`reduction_terms`, which that route
        and the columnar image read too."""
        from .replay import _signal_factor
        t = self.reduction_terms(tissue, scanner)
        logw = np.zeros(len(self.w))
        if self.exposure_t2 is not None:
            logw = logw - self.exposure_t2 @ t["invT2"] - self.exposure_t1 @ t["invT1"]
        if t["rho2_over_D"]:
            logw = logw + t["rho2_over_D"] * self.contact
        if t["rho1_over_D"]:
            logw = logw + t["rho1_over_D"] * self.contact_t1
        phi = self.phi
        if t["a_iso"] or t["a_aniso"]:
            phi = phi + (t["a_iso"] * self.field_iso + t["a_aniso"] * self.field_aniso)[:, None]
        return self.w, t["amplitude"] * self.w * np.exp(logw), _signal_factor(phi, t["voxel"])

    def signal(self, tissue=None, scanner=None, *, complex_signal=False):
        """The ensemble signal of this acquisition under the pair, ``(n_meas,)``: what
        :meth:`ReplayPack.replay` returns, magnitude unless ``complex_signal``.

        This is the ONE reduction of :meth:`signals`. The real part alone -- ``cos(phi).mean(0)`` -- is not the
        signal: it drops the quadrature the ensemble carries, which for a substrate seeded or shaped
        asymmetrically is ``s^2 / 2c`` of the magnitude and grows with ``b`` (5.2e-4 at b = 13,190 s/mm^2 on the
        MC/DC parity fixture, against a codec error of 1.1e-4; dmipy-sim#484)."""
        w, ew, E = self.signals(tissue, scanner)
        S = (ew[:, None] * E).sum(0) / w.sum()
        return planned(S if complex_signal else np.abs(S), self.plan)
