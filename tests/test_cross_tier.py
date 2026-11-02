"""Cross-tier replays against the one-dimensional Bloch-Torrey equation (:class:`dmipy_analytic.bloch_torrey.BlochTorrey1D`):
the situations in which two or more tiers act on the same walkers, so that only a replay that combines them walker by
walker reproduces the exact signal.

Two committed packs, walked by ``tools/make_xtier_fixtures.py`` (each its own seed):

* ``plane_linear_field.rpk``: water between two reflecting plates 8 um apart, its positions and the field channel of a
  linear field next to the walls. **Restriction x background gradient** (the cross term of dmipy-analytic#7, item 1)
  replayed by both routes: (i) the background folded into the effective waveform
  (:meth:`~dmipy_replay.acquisition.scanner_sequence.ScannerSequence.with_background_gradient`: it is on through the
  whole echo, the 180 flips its effective sign as it flips the pulsed gradient's), and (ii) the stored field read along
  each walker's path (C3) at a susceptibility and a main field whose product is the same gradient.
* ``slab_exchange.rpk``: the permeable slab of the replayable Monte Carlo paper, 4 um with a membrane of 10 um/s at
  2 um, every walker starting in the first compartment: **compartments x exchange x field x RF** (T2 per compartment,
  the field of a magnetised layer at 3 and 7 T, spin and gradient echoes) and **relaxivity x exchange** (the surface
  relaxivity on the membrane's faces, from the boundary local time of the walk's reflections at it).

Each comparison holds the replayed magnitude to the reference's within ``z se + e_ref``: ``se`` the standard error of
the replayed signal (the walkers' spread projected on the mean's phase; zero where every walker's factor is the same),
``z`` the two-sided normal quantile at a family-wise false-alarm probability of ``1e-3`` shared by every comparison of
this file (Bonferroni), and ``e_ref`` twice the reference's error (Richardson: the change when its cells and its steps
are doubled). Every replay reads every stored band and mode (``tol=None``): the pack, not a read plan, is under test.
"""
import math

import numpy as np
import pytest

import dmipy_replay as dr
from dmipy_replay import Tissue
from dmipy_analytic.bloch_torrey import BlochTorrey1D
from dmipy_analytic.constants import GAMMA

from pathlib import Path

DATA = Path(__file__).parent / "data"
D0 = 2e-9
ALPHA = 1e-3

PLANE_L, PLANE_B0, PLANE_CHI, PLANE_G0 = 8e-6, 3.0, 1e-6, 0.1        # tools/make_xtier_fixtures.py
SLAB_L, SLAB_KAPPA, SLAB_CHI = 4e-6, 1e-5, -1e-7
T2_A, T2_B = 0.080, 0.020                                             # the first (seeded) compartment and the second


def _z_family(comparisons, alpha=ALPHA):
    """The two-sided normal quantile at the family-wise false-alarm probability ``alpha`` shared by ``comparisons``
    (Bonferroni): the ``z`` with ``erfc(z / sqrt 2) = alpha / comparisons``, by bisection."""
    p = alpha / comparisons
    lo, hi = 0.0, 40.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if math.erfc(mid / math.sqrt(2.0)) > p else (lo, mid)
    return 0.5 * (lo + hi)


def _replayed(pack, seq, **kw):
    """``(S, se)``: the replayed complex signal of every measurement and the standard error of its magnitude."""
    w, ew, E = pack.walker_signals(seq, tol=None, **kw)
    w, ew, E = np.asarray(w, np.float64), np.asarray(ew, np.float64), np.asarray(E)
    S = (ew[:, None] * E).sum(0) / w.sum()
    z = ew[:, None] * E / w.mean()
    u = S / np.abs(S)
    return S, np.sqrt(np.real(z * np.conj(u)).var(0) / z.shape[0])


def _reference(make, seq):
    """``(S, e_ref)``: the reference of the sequence ``seq`` (the physical gradient along x on its grid, the 180 at
    its refocusing time as a conjugation, its ``n_t`` samples spanning ``n_t - 1`` steps) from ``make(n_cells)``, and
    its error, twice the Richardson estimate of the cells and of the steps."""
    G = np.asarray(seq.G, np.float64)[:, :-1, 0]
    dt = float(seq.dt)
    t180 = seq.rf.refocus_time
    refocus = () if t180 is None else (int(round(float(t180) / dt)),)
    run = lambda n, sub: make(n).signal(G, dt, refocus=refocus, sub_steps=sub)
    S, S_cells, S_steps = run(200, 2), run(400, 2), run(200, 4)
    e = 2.0 * (np.abs(np.abs(S) - np.abs(S_cells)) + np.abs(np.abs(S) - np.abs(S_steps))) * 4.0 / 3.0
    return S, e


# ------------------------------------------------------------------------------------------------ the rows
def _pgse(directions, bvalues, TE=0.04, delta=0.008, Delta=0.02, n_t=401):
    return dr.pgse(directions, delta, Delta, bvalues=bvalues, TE=TE, n_t=n_t)


PLANE_PGSE = dict(directions=[[1.0, 0, 0]] * 3 + [[-1.0, 0, 0]] * 2, bvalues=[0.0, 0.5e9, 1.5e9, 0.5e9, 1.5e9])
PLANE_BACKGROUND = (0.05, -0.05, 0.1)                                 # T/m along x: route (i)
PLANE_CHIS = (0.5e-6, -1e-6)                                          # route (ii): the gradient G0 chi / PLANE_CHI
PLANE_GRE_TE = (0.01, 0.03)                                           # route (ii) only: a gradient echo has a net moment

SLAB_ROWS = (("gre", 0.010), ("gre", 0.020), ("gre", 0.050), ("se", 0.020), ("se", 0.050))
SLAB_CASES = (("C1", True, None, 0.0), ("C3 3T", False, 3.0, 0.0), ("C3 7T", False, 7.0, 0.0),
              ("C1+C3 7T", True, 7.0, 0.0), ("C2", False, None, 5e-5), ("C1+C2+C3 3T", True, 3.0, 5e-5))
SLAB_PGSE = dict(directions=[[1.0, 0, 0]] * 2, bvalues=[1e9, 3e9], TE=0.05, delta=0.01, Delta=0.03, n_t=501)

COMPARISONS = (len(PLANE_BACKGROUND) * 5 + len(PLANE_CHIS) * 5 + len(PLANE_GRE_TE)
               + len(SLAB_CASES) * len(SLAB_ROWS) + 2)
Z = _z_family(COMPARISONS)


def _slab_seq(kind, TE):
    n_t = int(round(TE / 1e-4)) + 1
    if kind == "gre":
        return dr.gre(TE, n_t=n_t)
    return dr.pgse([[1.0, 0, 0]], 0.1 * TE, 0.4 * TE, bvalues=[0.0], TE=TE, n_t=n_t)


@pytest.fixture(scope="module")
def plane():
    return dr.read_rpk(DATA / "plane_linear_field.rpk")


@pytest.fixture(scope="module")
def slab():
    return dr.read_rpk(DATA / "slab_exchange.rpk")


def _check(S, se, R, e_ref, what):
    resid = np.abs(S) - np.abs(R)
    tol = Z * se + e_ref
    assert np.all(np.abs(resid) <= tol), (f"{what}: replayed {np.abs(S)}, reference {np.abs(R)}, residual {resid}, "
                                          f"se {se}, z {Z:.2f}, e_ref {e_ref}")


def _plane_reference(g0):
    return lambda n: BlochTorrey1D([-PLANE_L / 2, PLANE_L / 2], D0, background_gradient=g0, n_cells=n)


@pytest.mark.parametrize("g0", PLANE_BACKGROUND)
def test_a_background_gradient_between_plates_is_the_exact_signal_by_the_effective_waveform(plane, g0):
    """Route (i): PGSE along +x and -x at three b-values in a static gradient along x that does not switch off. The
    cross term with the pulsed gradient changes sign with the direction, so the two directions differ, as the
    reference says they do."""
    seq = _pgse(**PLANE_PGSE)
    S, se = _replayed(plane, seq.with_background_gradient([g0, 0.0, 0.0]))
    R, e_ref = _reference(_plane_reference(g0), seq)
    _check(S, se, R, e_ref, f"route (i), g0 = {g0} T/m")


@pytest.mark.parametrize("chi", PLANE_CHIS)
def test_a_linear_field_between_plates_is_the_exact_signal_by_the_field_channel(plane, chi):
    """Route (ii): the same echoes with the field channel's linear field at ``chi``: the gradient ``G0 chi / chi_0``."""
    seq = _pgse(**PLANE_PGSE)
    S, se = _replayed(plane, seq, tissue=Tissue(chi_iso=chi), scanner=PLANE_B0)
    R, e_ref = _reference(_plane_reference(PLANE_G0 * chi / PLANE_CHI), seq)
    _check(S, se, R, e_ref, f"route (ii), chi = {chi}")


@pytest.mark.parametrize("TE", PLANE_GRE_TE)
def test_a_linear_field_between_plates_dephases_a_gradient_echo_as_the_exact_signal(plane, TE):
    """Route (ii) on a free induction decay: nothing refocuses the field, and the walls bound the dephasing."""
    seq = dr.gre(TE, n_t=int(round(TE / 1e-4)) + 1)
    S, se = _replayed(plane, seq, tissue=Tissue(chi_iso=PLANE_CHI), scanner=PLANE_B0)
    R, e_ref = _reference(_plane_reference(PLANE_G0), seq)
    _check(S, se, R, e_ref, f"route (ii), gradient echo at {TE} s")


def _slab_reference(T2, B0, rho):
    R2 = (1.0 / T2_A, 1.0 / T2_B) if T2 else (0.0, 0.0)
    w = 0.0 if B0 is None else GAMMA * SLAB_CHI * B0 / 6.0           # the layer's field chi B0 (1_B - 1/2) / 3
    return lambda n: BlochTorrey1D([0.0, SLAB_L / 2, SLAB_L], D0, permeability=SLAB_KAPPA, membrane_relaxivity=rho,
                                   R2=R2, omega=(-w, w), density=(1.0, 0.0), n_cells=n)


def _slab_tissue(T2, B0, rho):
    kw = {}
    if T2:
        kw["T2"] = {"intra": T2_A, "extra": T2_B}
    if B0 is not None:
        kw["chi_iso"] = SLAB_CHI
    if rho:
        kw["rho2"] = rho
    return Tissue(**kw) if kw else None


@pytest.mark.parametrize("kind, TE", SLAB_ROWS, ids=[f"{k} {TE * 1e3:g} ms" for k, TE in SLAB_ROWS])
@pytest.mark.parametrize("case, T2, B0, rho", SLAB_CASES, ids=[c[0] for c in SLAB_CASES])
def test_the_permeable_slab_is_the_exact_signal(slab, case, T2, B0, rho, kind, TE):
    """Gradient echoes at 10, 20 and 50 ms and spin echoes at 20 and 50 ms of the slab, every tier the case names
    replayed from the one pack: T2 per compartment (C1), the magnetised layer's field (C3), the membrane's surface
    relaxivity (C2), the exchange through the membrane always. The spin echo keeps the dephasing of the walkers that
    cross the membrane between the excitation and the echo, which only the field read along each path reproduces."""
    seq = _slab_seq(kind, TE)
    S, se = _replayed(slab, seq, tissue=_slab_tissue(T2, B0, rho), scanner=B0)
    R, e_ref = _reference(_slab_reference(T2, B0, rho), seq)
    _check(S, se, R, e_ref, f"slab {case}, {kind} at {TE} s")


def test_the_permeable_slab_under_pgse_is_the_exact_signal(slab):
    """Exchange and T2 per compartment under the pulsed gradient across the membrane: PGSE at 1 and 3 ms/um^2."""
    p = SLAB_PGSE
    seq = dr.pgse(p["directions"], p["delta"], p["Delta"], bvalues=p["bvalues"], TE=p["TE"], n_t=p["n_t"])
    S, se = _replayed(slab, seq, tissue=_slab_tissue(True, None, 0.0))
    R, e_ref = _reference(_slab_reference(True, None, 0.0), seq)
    _check(S, se, R, e_ref, "slab PGSE with T2 per compartment")
