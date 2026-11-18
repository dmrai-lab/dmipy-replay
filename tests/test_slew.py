"""Every public builder plays its gradient within the slew it was given, and its lobes' areas and stated q are the
built ones.

The one slew rule, here and in :meth:`ScannerSequence.check_slew`: on every measurement and axis of the PLAYED
gradient (the gradient off before the first sample and after the last),
``max |diff G_a| <= slew dt + 2 spacing(float32(max |G_a|))``.

The area rule of a ramped lobe sampled at mid-step (every sample whose mid-step lies inside the span): the error of
the samples' sum against the ideal area is the midpoint rule's, which is exactly zero on every step where the shape
is straight. A kink of slope change ``s`` at ``u`` after the start of its step adds ``s e(u)`` to ``integral - sum``,
with ``e(u) = min(u, dt - u)^2 / 2`` (between 0 and ``dt^2 / 8``, periodic in ``u``).

A unit lobe of a builder starts on the grid, so its first kink adds nothing and, per unit amplitude,

    sum dt - delta = E(eps, delta) = (e(eps) + e(delta) - e(delta + eps)) / eps.

With ``d = dist(eps, dt Z)``, ``e(eps) = d^2 / 2`` and ``max_b (e(b) - e(b + eps)) = d (dt - d) / 2`` (at ``b`` half a
step off the grid), while the other sign reaches only ``d (dt - 2 d) / 2``; so exactly

    max_delta |E| = d dt / (2 eps),   i.e.   c*(eps) = dist(eps / dt, Z) / (2 eps / dt)   (in units of g dt).

Its supremum over ``eps >= dt`` is ``1/6``, at ``eps = 1.5 dt`` (``dt = 1, eps = 1.5, span = 20``: ``1/6`` exactly); over
all ``eps > 0`` it is ``1/2``, at every ``eps <= dt / 2``. A lobe of a train starts at a zero crossing that is not on
the grid (the segment between two crossings; a crossing inside a step contributes the same ``e(u) / eps`` as a
ramp's start), so its two ramps are independent: ``c*_train(eps) = d (dt - d) / (eps dt)``, whose supremum over
``eps >= dt`` is ``3 - 2 sqrt(2) = 0.1716`` at ``eps = sqrt(2) dt`` and over all ``eps`` tends to 1 as ``eps -> 0``. Both
are covered by the bound the builders state, ``c = min(1, dt / (4 eps))`` (``(1 - r) <= 1 / (4 r)`` is
``(2 r - 1)^2 >= 0``). A ramp shorter than one step is in that domain: at most one sample lies inside it, and the step
from zero to it is at most ``g = S eps < S dt``. These are checked on the float64 shapes, before storage.
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


# Every builder whose TE may be left automatic, at its default layout (no budget, default n_t), over a scan of grid
# phases. gre takes TE always; cpmg, splice and fexi tie their grid to their echoes or to a given dt.
def _auto():
    c = []
    # a b target re-iterates the amplitude at any other TE, so "one step fewer" is checked on the amplitude twins
    for d in np.linspace(8e-3, 9.9e-3, 30):
        c.append((f"pgse-bvalues-{d*1e3:.4f}", lambda d=d, **k: dr.pgse([X], d, 0.03, bvalues=1e9, **k)))
        c.append((f"pgse-{d*1e3:.4f}", lambda d=d, **k: dr.pgse([X], d, 0.03, gradient_strengths=0.07, **k)))
        c.append((f"pgste-bvalues-{d*1e3:.4f}", lambda d=d, **k: dr.pgste([X], d, 0.03, bvalues=1e9, **k)))
        c.append((f"pgste-{d*1e3:.4f}", lambda d=d, **k: dr.pgste([X], d, 0.03, gradient_strengths=0.07, **k)))
    for d in np.linspace(8.5e-3, 9.5e-3, 30):
        c.append((f"pgse600-bvalues-{d*1e3:.4f}",
                  lambda d=d, **k: dr.pgse([X], d, 0.029, bvalues=7.5e9, slew_rate=600.0, **k)))
        c.append((f"pgse600-{d*1e3:.4f}",
                  lambda d=d, **k: dr.pgse([X], d, 0.029, gradient_strengths=0.22, slew_rate=600.0, **k)))
    for s_ in np.linspace(0.02, 0.0299, 30):
        c.append((f"ste-{s_*1e3:.4f}", lambda s_=s_, **k: dr.ste(s_, gradient_strengths=0.03, **k)))
        c.append((f"pte-{s_*1e3:.4f}", lambda s_=s_, **k: dr.pte([0, 0, 1], s_, gradient_strengths=0.03, **k)))
    for sig in np.linspace(0.04, 0.0436, 30):                      # whole lobes: 2 f sigma = 8 at f = 100 / (sigma/0.04)
        c.append((f"ogse-trapezoid-{sig*1e3:.4f}",
                  lambda sig=sig, **k: dr.ogse([X], 4.0 / sig, sig, gradient_strengths=0.03, shape="trapezoid", **k)))
        c.append((f"ogse-cosine-{sig*1e3:.4f}",
                  lambda sig=sig, **k: dr.ogse([X], 2.0 / sig, sig, gradient_strengths=0.03, shape="cosine", **k)))
    return c


AUTO = _auto()


@pytest.mark.parametrize("name,build", AUTO, ids=[c[0] for c in AUTO])
def test_an_automatic_te_is_the_floor_extended_by_the_fewest_whole_steps(name, build):
    """``TE = minimum_te (n_t - 1) / (n_t - 1 - k)`` with ``k`` whole steps, the fewest at which every block fits whole:
    one fewer runs a block off the grid."""
    seq = build()
    te, floor, n = float(seq.T), float(seq.encoding.minimum_te), seq.n_t
    k = (n - 1) * (1.0 - floor / te)
    assert seq.encoding.te_auto and abs(k - round(k)) < 1e-6 and 0 <= round(k) <= 8
    k = int(round(k))
    if k > 0 and "bvalues" not in name:
        with pytest.raises(ValueError, match="runs off the"):
            build(TE=floor * (n - 1) / (n - k))                                # one step fewer
    assert _within(seq, 200.0 if "600" not in name else 600.0)[0]


def test_the_square_limit_te_is_the_floor_itself():
    for d in np.linspace(8e-3, 9.9e-3, 8):
        sq = dr.pgse([X], d, 0.03, bvalues=1e9, slew_rate=np.inf)
        assert float(sq.T) == float(sq.encoding.minimum_te)


def test_the_amplitude_iteration_returns_the_build_that_scales_down_when_it_flips(monkeypatch):
    """At ``delta = 8.6333 ms`` the automatic TE needs one extra step at the higher of two amplitudes and none at the
    lower, so the builds alternate; the sequence returned is the one that scales down (its ramps were made for an
    amplitude at least the one it plays), its b is the target and it plays within the slew."""
    from dmipy_replay.sequences import assemble as A
    seen = []
    layout = A.SpinEcho.layout

    def spy(self, spans, g, TE, dt, n_t):
        seen.append(TE)
        return layout(self, spans, g, TE, dt, n_t)

    monkeypatch.setattr(A.SpinEcho, "layout", spy)
    seq = dr.pgse([X], 0.008633333333333333, 0.03, bvalues=1e9, slew_rate=200.0)
    floor = float(seq.encoding.minimum_te)
    accepted = sorted({round(t * 1e9) for t in seen})
    assert len(accepted) >= 2                                                   # the TE flips between builds
    assert float(seq.T) > floor                                                 # the returned build is the k = 1 one
    assert seq.encoding.ramp_time[0] * 200.0 >= seq.encoding.gradient_strengths[0] * (1 - 1e-12)
    assert seq.b() == pytest.approx([1e9], rel=1e-6) and _within(seq, 200.0)[0]


# ── an explicit TE that leaves no room for the built blocks ───────────────────────────────────────────────────

def test_an_axis_pair_encoding_needs_its_built_length_and_says_so():
    """An axis pair plays less than one step more than ``2 span`` per axis, so a gradient-echo TE equal to
    ``gradient_duration`` (no budget) or to the budget's floor leaves the block too little room: it is refused by
    name, not cut. The automatic TE extends by the steps it needs."""
    for n_t, build in ((1002, lambda **k: dr.ste(0.0251, gradient_strengths=0.03, n_t=1002, **k)),
                       (1000, lambda **k: dr.pte([0, 0, 1], 0.0251, gradient_strengths=0.03, **k))):
        auto = build()
        assert float(auto.T) > 0.0251 and _within(auto, 200.0)[0]
        with pytest.raises(ValueError, match=rf"an encoding block of \d+ samples at sample 0 runs off the {n_t}-sample grid"):
            build(TE=0.0251)
        tm = dr.SequenceTiming(**TIMING)
        floor = float(build(timing=tm).encoding.minimum_te)
        with pytest.raises(ValueError, match=r"row 0: an encoding block of \d+ samples has room for \d+ between the "
                                             r"budget's lead-in and readout tail"):
            build(timing=tm, TE=floor)


def test_an_explicit_te_with_a_budget_cuts_a_lobe_only_within_the_slew():
    """With an explicit TE at a budget's floor the margin cuts the far end of each lobe of the pair. A ramped lobe
    cut inside its ramp starts on a sample above ``S dt``: the guard refuses it. A ramp shorter than a step is cut
    within the slew: it builds, and the area it loses is the dropped samples' (stated here)."""
    from dmipy_replay.sequences.assemble import trapezoid as tz
    tm = dr.SequenceTiming(**TIMING)
    refused = cut = 0
    for d in np.linspace(8e-3, 9.9e-3, 20):
        for g in (0.05, 0.006):
            floor = float(dr.pgse([X], d, 0.03, gradient_strengths=g, timing=tm).encoding.minimum_te)
            try:
                seq = dr.pgse([X], d, 0.03, gradient_strengths=g, timing=tm, TE=floor)
            except ValueError as e:
                assert g == 0.05 and "above the slew limit 200 T/m/s" in str(e)
                refused += 1
                continue
            full = tz(d, g / 200.0, seq.dt)
            G = np.asarray(seq.G, np.float64)[0, :, 0]
            on = np.flatnonzero(G)
            first = G[on[0]:on[0] + len(full)]
            first = first[:np.argmax(first == 0)] if np.any(first == 0) else first
            lost = (len(full) - len(first))
            if lost:
                cut += 1
                assert np.allclose(first, g * full[lost:], rtol=1e-6)          # the far end dropped, nothing else
                area_lost = g * float(full[:lost].sum()) * seq.dt
                assert 0 < area_lost <= lost * g * seq.dt                       # at most one full sample per step cut
            assert _within(seq, 200.0)[0]
    assert refused > 0 and cut > 0


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


def _c_lobe(r):
    """The exact supremum over ``delta`` of a builder lobe's area error, in ``g dt``, at ``eps = r dt``."""
    return abs(r - round(r)) / (2.0 * r)


def _c_train(r):
    """The same for a lobe of a train, whose start is off the grid."""
    d = abs(r - round(r))
    return d * (1.0 - d) / r


@pytest.mark.parametrize("r", [0.1, 0.3, 0.5, 0.7, 1.0, 1.25, math.sqrt(2.0), 1.5, 1.8, 2.5, 7.3])
def test_the_area_error_attains_its_closed_form_supremum(r):
    dt, eps = 1.0, r
    phases = np.linspace(0.0, 1.0, 4001)
    lobe = max(abs(_kink_error([(eps, -1 / eps), (b, -1 / eps), (b + eps, 1 / eps)], dt)) for b in 50.0 + phases)
    assert lobe == pytest.approx(_c_lobe(r), abs=1e-7) and lobe <= _c_lobe(r) + 1e-12
    grid = 50.0 + phases[::8]
    train = max(abs(_kink_error([(u, 1 / eps), (u + eps, -1 / eps), (b, -1 / eps), (b + eps, 1 / eps)], dt))
                for u in phases[::8] for b in grid)
    assert train <= _c_train(r) + 1e-12 and train == pytest.approx(_c_train(r), abs=5e-4)    # a 0.002-step scan
    d = abs(r - round(r))
    if d > 0:                                                                   # attained where the analysis says
        u = 0.5 if r - math.floor(r) <= 0.5 else 0.5 - (r - math.floor(r))     # e(u) - e(u + eps) at its maximum
        b = 50.0 + ((0.5 - r) % 1.0 if r - math.floor(r) <= 0.5 else 0.5)       # and the other pair at its minimum
        at = abs(_kink_error([(u, 1 / eps), (u + eps, -1 / eps), (b, -1 / eps), (b + eps, 1 / eps)], dt))
        assert at == pytest.approx(_c_train(r), abs=1e-12)
    assert _c_lobe(r) <= _c(eps, dt) + 1e-15 and _c_train(r) <= _c(eps, dt) + 1e-15


def test_the_suprema_and_where_they_are_attained():
    r = np.linspace(1.0, 40.0, 390001)
    lobe = np.abs(r - np.round(r)) / (2 * r)
    assert lobe.max() == pytest.approx(1 / 6, abs=1e-12) and r[np.argmax(lobe)] == pytest.approx(1.5, abs=1e-9)
    d = np.abs(r - np.round(r))
    train = d * (1 - d) / r
    assert train.max() == pytest.approx(3 - 2 * math.sqrt(2), abs=1e-9)
    assert r[np.argmax(train)] == pytest.approx(math.sqrt(2), abs=1e-4)
    small = np.linspace(1e-3, 0.5, 500)
    assert np.allclose(np.abs(small - np.round(small)) / (2 * small), 0.5)         # all eps <= dt/2: 1/2
    # attained by a built lobe: dt = 1, eps = 1.5, span = 20 -> exactly 1/6; eps = 0.3, delta = 20.5 -> 1/2
    for eps, delta, want in ((1.5, 18.5, 1 / 6), (0.3, 20.5, 0.5)):
        assert abs(float(trapezoid(delta, eps, 1.0).sum()) - delta) == pytest.approx(want, abs=1e-12)


def test_the_area_bound_holds_on_a_random_scan_of_built_lobes():
    rng = np.random.default_rng(0)
    for _ in range(4000):
        dt = 1.0
        eps = rng.uniform(0.05, 40.0)
        delta = eps + rng.uniform(0.0, 300.0)
        err = abs(float(trapezoid(delta, eps, dt).sum()) - delta) / dt
        assert err <= _c_lobe(eps) + 1e-12 and err <= _c(eps, dt) + 1e-12


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

def _peak_q(seq, m):
    """The largest ``|q(t)|`` of row ``m``'s played effective gradient along its direction, 1/m."""
    d = np.asarray(seq.encoding.gradient_directions[m], np.float64)
    run = np.cumsum(np.asarray(seq.G_eff, np.float64)[m] @ (d / np.linalg.norm(d))) * seq.dt
    return GAMMA / (2 * np.pi) * float(np.abs(run).max()), len(run)


def _first_lobe(s):
    on = np.flatnonzero(s)
    i0 = on[0]
    same = np.sign(s[i0:]) == np.sign(s[i0])
    return s[i0:i0 + (int(np.argmin(same)) if not same.all() else same.size)]


Q_BUILDS = {
    "pgse-ramped": lambda: dr.pgse([X], 9e-3, 29e-3, bvalues=7.5e9, slew_rate=600.0),
    "pgse-short": lambda: dr.pgse([X], 3.55e-3, 6e-3, bvalues=2e8, slew_rate=200.0),
    "pgse-square": lambda: dr.pgse([X], 9e-3, 29e-3, bvalues=7.5e9, slew_rate=np.inf),
    "pgste": lambda: dr.pgste([X], 8.6e-3, 0.03, bvalues=1e9, slew_rate=200.0),
    "gre": lambda: dr.gre(0.05, gradient_directions=[X], bvalues=1e8, delta=8.6e-3, Delta=0.03, slew_rate=200.0),
    "splice": lambda: dr.splice([X], 13e-3, 25e-3, 3, 10e-3, bvalues=[1e9], TE_prep=80e-3, n_t_per_echo=80),
    "ogse-trapezoid": lambda: dr.ogse([X], 100.0, 0.04, gradient_strengths=0.03, slew_rate=200.0, shape="trapezoid", n_t=731),
    "ogse-cosine": lambda: dr.ogse([X], 50.0, 0.02, gradient_strengths=0.05, shape="cosine", slew_rate=200.0),
    "cpmg": lambda: dr.cpmg(2, 0.02, gradient_directions=[X], bvalues=1e8, n_t_per_echo=400),
}


@pytest.mark.parametrize("name", list(Q_BUILDS))
def test_the_stated_q_is_the_peak_of_the_built_q_and_the_b_is_the_target(name):
    seq = Q_BUILDS[name]()
    G = np.asarray(seq.G, np.float64)
    for m in range(seq.n_meas):
        want, n = _peak_q(seq, m)
        tol = GAMMA / (2 * np.pi) * n * float(np.spacing(np.float32(np.abs(G[m]).max()))) * seq.dt
        assert abs(seq.encoding.qvalues[m] - want) <= tol
        if name.startswith(("pgse", "pgste", "gre", "splice")):                # a pulsed pair: the first lobe's area
            lobe = _first_lobe(G[m] @ seq.encoding.gradient_directions[m])
            assert abs(want - GAMMA / (2 * np.pi) * abs(lobe.sum()) * seq.dt) <= tol
    if seq.family not in ("ogse",):
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
