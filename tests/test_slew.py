"""Every public builder plays its gradient within the slew it was given, and its lobes' areas and stated q are the
built ones.

The one slew rule, here and in :meth:`ScannerSequence.check_slew`: on every measurement and axis of the PLAYED
gradient (the gradient off before the first sample and after the last),
``max |diff G_a| <= slew dt + 2 spacing(float32(max |G_a|))``.

The area rule of a ramped lobe sampled at mid-step (every sample whose mid-step lies inside the span): the error of
the samples' sum against the ideal area is the midpoint rule's, which is exactly zero on every step where the shape
is straight. A kink of slope change ``s`` at ``u`` after the start of its step adds ``s e(u)``, with
``e(u) = min(u, dt - u)^2 / 2`` (between 0 and ``dt^2 / 8``, and ``dt / 2``-Lipschitz in ``u``). A unit lobe has two
kink pairs, ``+1/eps`` at a ramp's start and ``-1/eps`` at its end ``eps`` later, so each pair contributes
``(e(u) - e(u')) / eps`` with ``|u - u'| <= eps``, at most ``min(dt / 2, dt^2 / (8 eps))``. Hence
``|area - A_ideal| <= c g dt`` with ``c = min(1, dt / (4 eps))`` (``1/4`` for every ramp of at least one step; the
supremum over ``eps >= dt`` is ``3 - 2 sqrt(2) = 0.1716``, at ``eps = sqrt(2) dt``; a ramp shorter than a step is
covered too, at most ``g dt`` as for a square lobe). A lobe of a train is the
segment between two ideal zero crossings; a crossing inside a step contributes the same ``e(u) / eps`` as a ramp's
start, so every lobe of a train has the same bound. These are checked on the float64 shapes, before storage.
"""
import math

import numpy as np
import pytest

import dmipy_replay as dr
from dmipy_analytic.constants import GAMMA
from dmipy_replay.sequences.assemble import axis_pairs, bipolar, cosine, trapezoid, trapezoid_train

X = [1.0, 0.0, 0.0]


def _within(seq, slew):
    """``(ok, worst step / (slew dt))`` of the played gradient under the one slew rule."""
    G = np.asarray(seq.G, np.float32)
    G64 = G.astype(np.float64)
    pad = np.zeros_like(G64[:, :1])
    step = np.abs(np.diff(np.concatenate([pad, G64, pad], axis=1), axis=1)).max(axis=1)        # (n_meas, 3)
    allowed = slew * seq.dt + 2.0 * np.spacing(np.abs(G).max(axis=1)).astype(np.float64)
    return bool(np.all(step <= allowed)), float(step.max() / (slew * seq.dt))


# Each family over a scan of grid phases: the span of a lobe against the step moves with delta, the duration or
# n_t. The recorded points of the defect are in it: pgse delta 9 ms / Delta 29 ms / b 7.5e9 at 600 T/m/s (1.4885
# times the slew before), the short lobe delta 3.55 ms / Delta 6 ms / b 2e8 at 200 T/m/s (1.4872), and the
# abutting-lobe shapes ste (1.50), pte (1.75) and the trapezoidal ogse (1.91) at 200 T/m/s.
def _cases():
    c = []
    for d in np.linspace(8e-3, 9.9e-3, 40):
        c.append((f"pgse-{d*1e3:.4f}ms-200", 200.0, lambda d=d: dr.pgse([X], d, 0.03, bvalues=1e9, slew_rate=200.0)))
        c.append((f"pgse-{d*1e3:.4f}ms-nt500", 200.0,
                  lambda d=d: dr.pgse([X], d, 0.03, bvalues=1e9, slew_rate=200.0, n_t=500)))
        c.append((f"pgste-{d*1e3:.4f}ms", 200.0, lambda d=d: dr.pgste([X], d, 0.03, bvalues=1e9, slew_rate=200.0)))
        c.append((f"gre-{d*1e3:.4f}ms", 200.0,
                  lambda d=d: dr.gre(0.05, gradient_directions=[X], bvalues=1e8, delta=d, Delta=0.03, slew_rate=200.0)))
    for d in np.linspace(8.5e-3, 9.5e-3, 30):
        c.append((f"pgse-{d*1e3:.4f}ms-600", 600.0, lambda d=d: dr.pgse([X], d, 0.029, bvalues=7.5e9, slew_rate=600.0)))
    c.append(("pgse-9ms-29ms-7.5e9-600", 600.0, lambda: dr.pgse([X], 9e-3, 29e-3, bvalues=7.5e9, slew_rate=600.0)))
    c.append(("pgse-3.55ms-6ms-2e8-200", 200.0, lambda: dr.pgse([X], 3.55e-3, 6e-3, bvalues=2e8, slew_rate=200.0)))
    for n in range(700, 760, 2):
        c.append((f"ogse-trapezoid-nt{n}", 200.0,
                  lambda n=n: dr.ogse([X], 100.0, 0.04, gradient_strengths=0.03, slew_rate=200.0, shape="trapezoid", n_t=n)))
    for n in range(700, 760, 8):
        c.append((f"ogse-cosine-nt{n}", 200.0,
                  lambda n=n: dr.ogse([X], 100.0, 0.04, gradient_strengths=0.03, slew_rate=200.0, shape="cosine", n_t=n)))
    for s in np.linspace(0.02, 0.0299, 40):
        c.append((f"ste-{s*1e3:.4f}ms", 200.0, lambda s=s: dr.ste(s, gradient_strengths=0.03, slew_rate=200.0)))
        c.append((f"pte-{s*1e3:.4f}ms", 200.0, lambda s=s: dr.pte([0, 0, 1], s, gradient_strengths=0.03, slew_rate=200.0)))
    for d in np.linspace(3e-3, 4.9e-3, 30):
        c.append((f"fexi-{d*1e3:.4f}ms", 200.0,
                  lambda d=d: dr.fexi(X, d, 10e-3, 0.05, b_f=[0.0, 1e9], bvalues=[0.8e9, 0.8e9], dt=1e-5, slew_rate=200.0)))
    for d in np.linspace(12e-3, 15e-3, 30):
        c.append((f"splice-{d*1e3:.4f}ms", 200.0,
                  lambda d=d: dr.splice([X], d, 25e-3, 3, 10e-3, bvalues=[1e9], TE_prep=80e-3, n_t_per_echo=80)))
    for te in np.linspace(0.02, 0.0299, 8):
        c.append((f"cpmg-{te*1e3:.4f}ms", 200.0,
                  lambda te=te: dr.cpmg(2, te, gradient_directions=[X], bvalues=1e8, slew_rate=200.0, n_t_per_echo=400)))
        c.append((f"cpmg-alternate-{te*1e3:.4f}ms", 200.0,
                  lambda te=te: dr.cpmg(2, te, gradient_directions=[X], bvalues=1e8, slew_rate=200.0, n_t_per_echo=400,
                                        polarity="alternate")))
    return c


CASES = _cases()


@pytest.mark.parametrize("name,slew,build", CASES, ids=[c[0] for c in CASES])
def test_every_builder_plays_within_its_slew(name, slew, build):
    seq = build()
    ok, worst = _within(seq, slew)
    assert ok, f"{name}: the largest step is {worst:.6f} times the slew"
    seq.check_slew(slew)


def test_the_scan_covers_every_builder_and_at_least_thirty_phases_each():
    fams = {}
    for name, _, _ in CASES:
        fam = name.split("-")[0] + ("-" + name.split("-")[1] if name.startswith("ogse") else "")
        fams[fam] = fams.get(fam, 0) + 1
    assert set(fams) == {"pgse", "pgste", "gre", "ogse-trapezoid", "ogse-cosine", "ste", "pte", "fexi", "splice", "cpmg"}
    assert all(n >= 30 for f, n in fams.items() if f not in ("ogse-cosine", "cpmg"))
    assert fams["ogse-cosine"] >= 8 and fams["cpmg"] >= 8


TIMING = dict(t_excite=2e-3, t_refocus=4e-3, t_readout_pre_echo=8e-3)


@pytest.mark.parametrize("k", range(6))
@pytest.mark.parametrize("family", ["pgse", "pgste", "gre", "ste", "ogse"])
def test_a_timing_budget_layout_builds_whole_within_the_slew(family, k):
    tm = dr.SequenceTiming(**TIMING)
    d, s = 8e-3 + 0.37e-3 * k, 0.02 + 1.9e-3 * k
    seq = {"pgse": lambda: dr.pgse([X], d, 0.03, bvalues=1e9, timing=tm),
           "pgste": lambda: dr.pgste([X], d, 0.03, bvalues=1e9, timing=tm),
           "gre": lambda: dr.gre(0.06, gradient_directions=[X], bvalues=1e8, delta=d, Delta=0.03, timing=tm),
           "ste": lambda: dr.ste(s, gradient_strengths=0.03, timing=tm),
           "ogse": lambda: dr.ogse([X], 100.0, 0.04, gradient_strengths=0.03, timing=tm, n_t=700 + 7 * k)}[family]()
    assert _within(seq, 200.0)[0]


@pytest.mark.parametrize("d", np.linspace(8e-3, 9.9e-3, 12))
def test_an_automatic_te_is_the_floor_extended_by_whole_steps(d):
    seq = dr.pgse([X], d, 0.03, bvalues=1e9, slew_rate=200.0)
    te, floor, n = float(seq.T), float(seq.encoding.minimum_te), seq.n_t
    k = (n - 1) * (1.0 - floor / te)
    assert abs(k - round(k)) < 1e-6 and 0 <= round(k) <= 1
    square = dr.pgse([X], d, 0.03, bvalues=1e9, slew_rate=np.inf)
    assert float(square.T) == float(square.encoding.minimum_te)               # the square limit: the floor itself


# ── the guard ────────────────────────────────────────────────────────────────────────────────────────────────────

def test_the_guard_refuses_a_step_beyond_the_slew_naming_axis_value_and_limit():
    seq = dr.pgse([X, [0.0, 0.0, 1.0]], 0.01, 0.03, bvalues=[1e9, 1e9], slew_rate=200.0)
    assert seq.check_slew(200.0) is seq
    peak = seq.peak_slew()
    assert peak.shape == (2, 3) and np.all(peak <= 200.0 * (1 + 1e-5))
    G = np.asarray(seq.G, np.float64).copy()
    k = int(np.argmax(np.abs(G[1, :, 2])))
    G[1, k, 2] += 300.0 * seq.dt                                            # one sample stepped by 300 T/m/s more
    bad = seq.with_gradient(G)
    with pytest.raises(ValueError, match=r"measurement 1, axis z: the played gradient steps at [0-9.]+ T/m/s, "
                                         r"above the slew limit 200 T/m/s"):
        bad.check_slew(200.0)
    bad.check_slew([200.0, 200.0, 1e6])                                     # per axis: z allowed faster


def test_a_ramp_shorter_than_one_step_plays_within_the_slew():
    """A ramp of ``eps < dt`` has at most one sample inside it; the sampled lobe still steps by at most ``dt / eps``
    (unit amplitude) between samples, so it plays within the slew, and its area error is at most ``g dt``."""
    for n in (200, 300, 400):                                             # g = 0.03 T/m ramps over 150 us; dt 400-200 us
        seq = dr.ogse([X], 100.0, 0.04, gradient_strengths=0.03, slew_rate=200.0, shape="trapezoid", n_t=n)
        assert seq.encoding.gradient_rise_time[0] < seq.dt
        assert _within(seq, 200.0)[0]
    seq = dr.pgse([X], 35e-3, 42e-3, bvalues=1e8, n_t=400)                  # a low shell on a coarse grid
    assert seq.encoding.ramp_time[0] < seq.dt and _within(seq, 200.0)[0]
    seq = dr.cpmg(2, 0.02, gradient_directions=[X], bvalues=1e8, slew_rate=200.0)
    assert seq.encoding.ramp_time[0] < seq.dt and _within(seq, 200.0)[0]


# ── the area of a lobe ───────────────────────────────────────────────────────────────────────────────────────────

def _e(u, dt):
    u = u % dt
    return min(u, dt - u) ** 2 / 2.0


def _kink_error(kinks, dt):
    """The midpoint rule's exact error ``sum s e(u)`` of a piecewise-linear shape with kinks ``(time, slope change)``
    on the grid of steps ``[k dt, (k + 1) dt)``."""
    return sum(s * _e(t, dt) for t, s in kinks)


def _c(eps, dt):
    return min(1.0, dt / (4.0 * eps))


PHI = [0.0, 0.25, 0.5, 0.75, 0.999]


@pytest.mark.parametrize("r", [0.2, 0.5, 0.8, 1.0, 1.2, math.sqrt(2.0), 1.5, 2.7, 9.3, 31.0])
@pytest.mark.parametrize("phi", PHI)
def test_a_lobe_area_is_the_midpoint_rule_on_its_ideal_trapezoid(r, phi):
    dt = 1.0
    eps = r * dt
    for m in (3, 17, 120):
        span = (m + phi) * dt
        delta = span - eps
        if delta < eps:
            continue
        lobe = trapezoid(delta, eps, dt)
        area = float(lobe.sum()) * dt
        exact = _kink_error([(0.0, 1 / eps), (eps, -1 / eps), (delta, -1 / eps), (span, 1 / eps)], dt)
        assert area - delta == pytest.approx(-exact, abs=1e-12)                 # sum = integral - error
        assert abs(area - delta) <= _c(eps, dt) * dt + 1e-12
        assert np.all(np.abs(np.diff(np.concatenate([[0.0], lobe, [0.0]]))) <= dt / eps + 1e-12)


def test_the_area_bound_is_the_supremum_of_a_random_scan():
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(4000):
        dt = 1.0
        eps = rng.uniform(1.0, 40.0)
        delta = eps + rng.uniform(0.0, 300.0)
        err = abs(float(trapezoid(delta, eps, dt).sum()) - delta) / dt
        assert err <= _c(eps, dt) + 1e-12
        worst = max(worst, err)
    assert worst <= 3.0 - 2.0 * math.sqrt(2.0) + 1e-12                        # the supremum over eps >= dt


@pytest.mark.parametrize("phi", PHI)
def test_the_square_limit_keeps_whole_steps(phi):
    dt, delta = 1.0, 40.0 + phi
    lobe = trapezoid(delta, 0.0, dt)
    assert len(lobe) == 40 and np.all(lobe == 1.0)                            # floor(delta / dt) samples
    assert -dt < float(lobe.sum()) * dt - delta <= 0.0


def _segments(a, dt, crossings):
    """Each lobe of a sampled train: the samples whose mid-step lies between two ideal zero crossings."""
    t = (np.arange(len(a)) + 0.5) * dt
    edges = [0.0] + list(crossings) + [np.inf]
    return [a[(t >= lo) & (t < hi)] for lo, hi in zip(edges[:-1], edges[1:])]


@pytest.mark.parametrize("dt", [0.37, 0.5, 0.731, 1.0, 1.119])
def test_every_lobe_of_a_train_holds_the_lobe_bound_and_the_slew(dt):
    n_lobes, L, eps = 6, 23.4, 2.6
    a = trapezoid_train(n_lobes, L, eps, dt)
    assert len(a) == math.ceil(n_lobes * L / dt - 0.5)                        # every mid-step inside the train
    assert np.abs(np.diff(np.concatenate([[0.0], a, [0.0]]))).max() <= dt / eps + 1e-12
    lobes = _segments(a, dt, [k * L for k in range(1, n_lobes)])
    assert len(lobes) == n_lobes
    for k, lobe in enumerate(lobes):
        area = abs(float(lobe.sum())) * dt
        assert np.all(np.sign(lobe[lobe != 0]) == (1 if k % 2 == 0 else -1))
        assert abs(area - (L - eps)) <= _c(eps, dt) * dt + 1e-12


@pytest.mark.parametrize("dt", [0.37, 0.5, 0.731, 1.0, 1.119])
def test_an_axis_pair_is_a_lobe_and_its_exact_negative(dt):
    eps, dur = 2.6, 6 * 23.4
    a = axis_pairs(np.eye(3), dur, eps, dt)
    seg = dur / 6
    for ax in range(3):
        col = a[:, ax]
        on = np.flatnonzero(col)
        pos, neg = col[col > 0], col[col < 0]
        assert np.array_equal(pos, -neg)                                      # sample for sample
        assert abs(float(col.sum())) <= 1e-12 * float(np.abs(col).sum())   # q(end) = 0 per axis, float64
        assert abs(float(pos.sum()) * dt - (seg - eps)) <= _c(eps, dt) * dt + 1e-12
        assert np.abs(np.diff(np.concatenate([[0.0], col, [0.0]]))).max() <= dt / eps + 1e-12   # incl. the junction
        assert on[0] >= 0
    n1 = len(trapezoid(seg - eps, eps, dt))
    assert len(a) == 3 * (n1 + math.ceil(seg / dt)) and len(a) * dt < dur + 3 * dt   # each pair < one step longer


@pytest.mark.parametrize("Delta", [10.0, 10.2, 10.6, 11.0, 13.3])
def test_a_bipolar_pair_cancels_sample_for_sample_within_the_slew(Delta):
    dt, delta, eps = 1.0, 7.4, 2.6
    a = bipolar(delta, Delta, eps, dt)
    assert abs(float(a.sum())) <= 1e-12 * float(np.abs(a).sum())
    assert np.abs(np.diff(np.concatenate([[0.0], a, [0.0]]))).max() <= dt / eps + 1e-12


def _cos_area(lo, hi, w, eps, sigma):
    """The exact area of ``cos(w t) min(1, t / eps, (sigma - t) / eps)`` over ``[lo, hi]`` (fine-split Simpson on each
    smooth piece, whose error is below 1e-13 of the area here)."""
    cuts = sorted({lo, hi, *[x for x in (eps, sigma - eps) if lo < x < hi]})
    tot = 0.0
    for a, b in zip(cuts[:-1], cuts[1:]):
        t = np.linspace(a, b, 20001)
        f = np.cos(w * t) * np.minimum(1.0, np.minimum(t / eps, (sigma - t) / eps))
        h = (b - a) / 20000
        tot += h / 3 * (f[0] + f[-1] + 4 * f[1:-1:2].sum() + 2 * f[2:-1:2].sum())
    return tot


@pytest.mark.parametrize("dt", [0.37, 0.731, 1.0, 1.119])
def test_every_lobe_of_a_ramped_cosine_holds_the_midpoint_bound(dt):
    """A cosine lobe's bound: per step ``dt^3 M2 / 24`` where the shape is smooth (``M2 = w^2 + 2 w / eps`` bounds its
    second derivative), ``|s| dt^2 / 8`` at each ramp corner, and ``|f'| dt^2 / 8 + 2/3 M2 dt^3`` at each zero
    crossing inside a step (``f' <= 1 / eps``)."""
    f, n_cyc, eps = 1.0 / 80.0, 2, 6.0
    w, sigma = 2 * np.pi * f, n_cyc / f
    a = cosine(n_cyc, f, eps, dt)
    assert np.abs(np.diff(np.concatenate([[0.0], a, [0.0]]))).max() <= dt / eps + 1e-12
    cross = [(2 * j + 1) / (4 * f) for j in range(2 * n_cyc)]
    M2 = w * w + 2 * w / eps
    edges = [0.0] + cross + [sigma]
    for k, lobe in enumerate(_segments(a, dt, cross)):
        lo, hi = edges[k], edges[k + 1]
        ideal = _cos_area(lo, hi, w, eps, sigma)
        n_corner = sum(lo <= x <= hi for x in (0.0, eps, sigma - eps, sigma))
        n_cross = sum(1 for x in (lo, hi) if x in cross)
        bound = ((hi - lo) / dt + 2) * dt ** 3 * M2 / 24 + n_corner * dt ** 2 / (8 * eps) \
            + n_cross * (dt ** 2 / (8 * eps) + 2 / 3 * M2 * dt ** 3)
        assert abs(float(lobe.sum()) * dt - ideal) <= bound


# ── the stated q, the b and the refocusing of a built sequence ───────────────────────────────────────────────────

def _first_lobe(s):
    on = np.flatnonzero(s)
    i0 = on[0]
    same = np.sign(s[i0:]) == np.sign(s[i0])
    return s[i0:i0 + (int(np.argmin(same)) if not same.all() else same.size)]


@pytest.mark.parametrize("build", [
    lambda: dr.pgse([X], 9e-3, 29e-3, bvalues=7.5e9, slew_rate=600.0),
    lambda: dr.pgse([X], 3.55e-3, 6e-3, bvalues=2e8, slew_rate=200.0),
    lambda: dr.pgse([X], 9e-3, 29e-3, bvalues=7.5e9, slew_rate=np.inf),
    lambda: dr.pgste([X], 8.6e-3, 0.03, bvalues=1e9, slew_rate=200.0),
    lambda: dr.gre(0.05, gradient_directions=[X], bvalues=1e8, delta=8.6e-3, Delta=0.03, slew_rate=200.0),
    lambda: dr.splice([X], 13e-3, 25e-3, 3, 10e-3, bvalues=[1e9], TE_prep=80e-3, n_t_per_echo=80),
    lambda: dr.ogse([X], 100.0, 0.04, gradient_strengths=0.03, slew_rate=200.0, shape="trapezoid", n_t=731),
])
def test_the_stated_q_is_the_built_first_lobe_and_the_b_is_the_target(build):
    seq = build()
    G = np.asarray(seq.G, np.float64)
    for m in range(seq.n_meas):
        lobe = _first_lobe(G[m] @ seq.encoding.gradient_directions[m])
        tol = len(lobe) * float(np.spacing(np.float32(np.abs(G[m]).max()))) * seq.dt
        assert abs(seq.encoding.qvalues[m] - GAMMA / (2 * np.pi) * abs(lobe.sum()) * seq.dt) \
            <= GAMMA / (2 * np.pi) * tol
    if seq.family != "ogse":
        assert seq.b() == pytest.approx(seq.encoding.bvalues, rel=1e-6)       # float32 storage of G


@pytest.mark.parametrize("build", [
    lambda: dr.pgse([X], 9e-3, 29e-3, bvalues=7.5e9, slew_rate=600.0),
    lambda: dr.ogse([X], 100.0, 0.04, gradient_strengths=0.03, slew_rate=200.0, shape="trapezoid", n_t=731),
    lambda: dr.ste(0.0251, gradient_strengths=0.03, slew_rate=200.0),
    lambda: dr.pte([0, 0, 1], 0.0251, gradient_strengths=0.03, slew_rate=200.0),
    lambda: dr.gre(0.05, gradient_directions=[X], bvalues=1e8, delta=8.6e-3, Delta=0.03, slew_rate=200.0),
])
def test_every_axis_refocuses_at_the_echo(build):
    seq = build()
    G = np.asarray(seq.G, np.float32)
    Ge = np.asarray(seq.G_eff, np.float64)
    for m in range(seq.n_meas):
        for ax in range(3):
            tol = G.shape[1] * float(np.spacing(np.abs(G[m, :, ax]).max())) * seq.dt
            assert abs(float(Ge[m, :, ax].sum()) * seq.dt) <= tol + 1e-30
