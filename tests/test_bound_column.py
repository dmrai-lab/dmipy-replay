"""The C1 ``bound`` column is read, and the vector-Bloch route reads the exact occupancy runs (RPK.md 5.3, 6.5, 8.8, 9.5.7).

The specification: the occupancy over all declared pools sums to 1 at every instant, with the bound occupancy ``b`` and
the geometric label ``g`` the pool occupancies being ``f_bound = b`` and ``f_g = (1 - b) [comp = g]``, and replay weights
the per-pool rates by them, ``R(t) = sum_c f_c(t) R_c`` (5.3); the integral of a rate over an interval is the sum of the
rate of each label times its overlap with the interval, exact (9.5.7); a spin bound for a fraction of an interval has that
fraction of it at the bound pool's relaxation (6.5).

The expected values come from the oracle below (a piece-by-piece propagation with its own run arithmetic, written here), from
closed forms and from the committed golden values of the client as it was before the column was read
(``data/bound_regression_golden.npz``, written by ``data/make_bound_regression_golden.py`` and never regenerated).

Exact cases (one walker, no gradient, no off resonance, ``h = 1e-4`` s, three pools with ``R2 = (10, 200, 1000) / s`` and
``R1 = (1, 5, 20) / s``, the third the bound pool): hard 90 degree pulses about x from ``M = (0, 0, 1)``, ``M_z`` relaxing
by the propagator's model (it decays by ``exp(-R1 ell)``), the readout ``|M_xy|`` at the end.
"""
import json
import re
from pathlib import Path

import numpy as np
import pytest

import dmipy_replay as dr
from dmipy_replay import Tissue
from dmipy_replay.acquisition.rf import RFEvent
from dmipy_replay.replay.bloch import replay_bloch
from dmipy_replay.replay.compression import CumulativeGate, occupancy_events, occupancy_pieces, relaxation_logweight_runs
from dmipy_replay.replay.container import read_safetensors
from dmipy_replay.replay.replay import ReplayPack

DATA = Path(__file__).parent / "data"
H = 1e-4
R2 = (10.0, 200.0, 1000.0)
R1 = (1.0, 5.0, 20.0)
T2_B, T1_B = 1.0 / R2[2], 1.0 / R1[2]
COMP, BOUND = {"name": "comp", "kind": "events"}, {"name": "bound", "kind": "events"}
TE = 0.05

TEXT_NEEDS_T2_T1 = "a pack carrying the bound column needs T2_b and T1_b for a relaxation read (RPK.md 5.3, 8.8)"
TEXT_NEEDS_DELTA = "a pack carrying the bound column needs delta_omega_b for replay_bloch (RPK.md 6.5)"
TEXT_SCALAR_DELTA = "delta_omega_b is read by replay_bloch only"
TEXT_NO_COLUMN = "this pack carries no bound column: T2_b, T1_b and delta_omega_b are not read"


# ---- the oracle -----------------------------------------------------------------------------------------------------

def _label_at(events, t):
    """The label held at ``t``: that of the last event at or before it."""
    lab = events[0][1]
    for te, b in events:
        if te <= t:
            lab = b
    return lab


def _edges(pulses, T):
    """The block grid of ``[0, T]`` cut at every pulse instant, an instant that falls on a block end cutting nothing."""
    times = sorted([i * H for i in range(int(round(T / H)) + 1)] + list(pulses))
    out = [times[0]]
    for t in times[1:]:
        if t - out[-1] > 1e-9 * H:
            out.append(t)
    return out


def oracle(pulses, T, comp_ev, bound_ev):
    """``(|M_xy|, M_z)`` at ``T`` of one walker: the piece-by-piece propagation, a 90 degree rotation about x at the start of
    the piece that begins at its instant, then over the piece ``M_xy`` times ``exp(-R2 ell)`` and ``M_z`` times
    ``exp(-R1 ell)``, the rates ``R = sum over the runs inside the piece of (overlap) [(1 - b) R_g + b R_b]``."""
    edges = _edges(pulses, T)
    M = [0.0, 0.0, 1.0]
    for a, b in zip(edges[:-1], edges[1:]):
        if any(abs(a - p) < 1e-9 * H for p in pulses):
            M = [M[0], -M[2], M[1]]
        cuts = sorted({a, b} | {t for t, _ in comp_ev + bound_ev if a < t < b})
        int2 = int1 = 0.0
        for u, v in zip(cuts[:-1], cuts[1:]):
            g, bb = _label_at(comp_ev, u), _label_at(bound_ev, u)
            int2 += (v - u) * ((1 - bb) * R2[g] + bb * R2[2])
            int1 += (v - u) * ((1 - bb) * R1[g] + bb * R1[2])
        M = [M[0] * np.exp(-int2), M[1] * np.exp(-int2), M[2] * np.exp(-int1)]
    return float(np.hypot(M[0], M[1])), float(M[2])


# a walker in the bound pool on [1.25 h, 2.75 h] and in geometric pool 0 otherwise
COMP0 = [(0.0, 0)]
BOUND_RUN = [(0.0, 0), (1.25 * H, 1), (2.75 * H, 0)]
# ... or, with no bound column, in geometric pool 1 (whose rates are those of the bound pool) on it
COMP_RUN = [(0.0, 0), (1.25 * H, 1), (2.75 * H, 0)]
# case J: geometric pool 1 with b = 0 on [0, h/2], geometric pool 0 with b = 1 on [h/2, h]
J_COMP = [(0.0, 1), (0.5 * H, 0)]
J_BOUND = [(0.0, 0), (0.5 * H, 1)]

CASES = {                                        # pulses, readout time, |M_xy|
    "C": ([0.0], 4 * H, 0.8585588940),           # FID: exp(-(10 * 2.5e-4 + 1000 * 1.5e-4))
    "A": ([0.0, 4 * H, 6 * H], 8 * H, 0.8566721406),
    "B": ([0.0, 2.5 * H, 6 * H], 8 * H, 0.8789080450),     # the pulse at 2.5 h splits block 2
}
J = ([0.0], H, 0.9417645336)                     # exp(-600 h): f_1 = f_bound = 1/2


def _arrays(walkers):
    """Plain pack arrays of ``walkers`` ``[(comp_events, bound_events), ...]`` (each list opens with the label at the
    window's start)."""
    out = {}
    for name, k in (("comp", 0), ("bound", 1)):
        lists = [w[k] for w in walkers]
        out[f"{name}_events_offsets"] = np.cumsum([0] + [len(x) for x in lists])
        out[f"{name}_events_t"] = np.array([t for x in lists for t, _ in x])
        out[f"{name}_events_label"] = np.array([b for x in lists for _, b in x], np.int8)
    return out


def _client(walkers, pulses, T, *, bound, backend="numpy"):
    """``(|M_xy| per walker, |mean M_xy|)`` of the propagator on ``walkers``' occupancies, formed by
    :func:`occupancy_pieces` on the pieces of the walk. ``bound``: the third pool is the bound column's; else it is the
    geometric pool 1 (``walkers`` then carry no ``bound`` events and the pool 1 of the rates is the bound pool's)."""
    edges = np.array(_edges(pulses, T))
    n_t = int(round(T / H)) + 1
    arrays = _arrays(walkers)
    n_w = len(walkers)
    traj, G = np.zeros((n_w, n_t, 3)), np.zeros((1, n_t, 3))
    rf = [RFEvent(t, 90.0, "x", axis_deg=0.0) for t in pulses]
    if bound:
        f_comp, f_bound = occupancy_pieces(arrays, COMP, edges, 2, bound_column=BOUND)
        kw = dict(comp_traj=f_comp, bound_frac=f_bound, T2_bound=T2_B, T1_bound=T1_B,
                  T2_per_comp=[1 / R2[0], 1 / R2[1]], T1_per_comp=[1 / R1[0], 1 / R1[1]])
    else:
        f_comp, f_bound = occupancy_pieces(arrays, COMP, edges, 2)
        assert f_bound is None
        kw = dict(comp_traj=f_comp, T2_per_comp=[1 / R2[0], 1 / R2[2]], T1_per_comp=[1 / R1[0], 1 / R1[2]])
    if backend == "jax":
        pytest.importorskip("jax")
        from dmipy_replay.replay.bloch_jax import replay_bloch_jax as fn
    else:
        fn = replay_bloch
    M, sig = fn(traj, H, G, H, rf, return_walker_signals=True, **kw)
    return np.hypot(M[0], M[1]), float(np.abs(sig)[0])


def test_the_oracle_reproduces_the_stated_values():
    for name, (pulses, T, value) in CASES.items():
        assert oracle(pulses, T, COMP0, BOUND_RUN)[0] == pytest.approx(value, abs=1e-9), name
    pulses, T, value = J
    assert oracle(pulses, T, J_COMP, J_BOUND)[0] == pytest.approx(value, abs=1e-9)
    assert value == pytest.approx(np.exp(-600.0 * H), abs=1e-9)
    # the block-end label, the whole-block average and no bound pool, each of which the specification rules out
    assert np.exp(-(10 * 2.5e-4 + 1000 * 1.5e-4)) == pytest.approx(CASES["C"][2], abs=1e-9)


@pytest.mark.parametrize("backend", ["numpy", "jax"])
@pytest.mark.parametrize("bound", [True, False])
@pytest.mark.parametrize("case", ["C", "A", "B"])
def test_the_bloch_route_reads_the_exact_runs_of_each_piece(case, bound, backend):
    """Cases C (FID; the events inside blocks 1 and 2), A (three pulses, none inside a block) and B (the block that a
    pulse splits: each side of the split reads the runs on its side). Two walkers: the one with the run and one that stays
    in geometric pool 0."""
    pulses, T, value = CASES[case]
    run = (COMP0, BOUND_RUN) if bound else (COMP_RUN, [(0.0, 0)])
    walkers = [run, (COMP0, [(0.0, 0)])]
    per, mean = _client(walkers, pulses, T, bound=bound, backend=backend)
    exact = [oracle(pulses, T, *w)[0] for w in ([COMP0, BOUND_RUN], [COMP0, [(0.0, 0)]])]
    assert exact[0] == pytest.approx(value, abs=1e-9)
    rel = 1e-12 if backend == "numpy" else 1e-5
    assert np.allclose(per, exact, rtol=rel, atol=0.0), (per, exact)
    assert mean == pytest.approx(np.mean(exact), rel=rel)


@pytest.mark.parametrize("backend", ["numpy", "jax"])
def test_the_joint_occupancy_is_averaged_not_the_product_of_averages(backend):
    """Case J: ``comp`` and ``bound`` change together inside one block, so ``f_0 = 0``, ``f_1 = f_bound = 1/2`` and
    ``R2 = 600 / s`` (``exp(-600 h)`` = 0.9417645336); the product of the two block averages would give ``f_0 = f_1 =
    1/4`` and ``R2 = 552.5 / s`` (0.9462485562), the block-end label 0.9048374180."""
    pulses, T, value = J
    per, _ = _client([(J_COMP, J_BOUND)], pulses, T, bound=True, backend=backend)
    rel = 1e-12 if backend == "numpy" else 1e-5
    assert value == pytest.approx(np.exp(-600.0 * H), abs=1e-9)
    assert per[0] == pytest.approx(np.exp(-600.0 * H), rel=rel)          # f_1 = f_bound = 1/2 over the whole block
    product_of_averages = float(np.exp(-(0.25 * R2[0] + 0.25 * R2[1] + 0.5 * R2[2]) * H))
    assert product_of_averages == pytest.approx(0.9462485562, abs=1e-9)
    assert abs(per[0] - product_of_averages) > 1e-3
    assert abs(per[0] - np.exp(-R2[2] * H)) > 1e-3                      # not the block-end label (comp 0, b = 1)


def test_occupancy_pieces_are_the_averages_of_the_pool_occupancies():
    """Case B's pieces by hand: the bound run ``[1.25 h, 2.75 h]`` cut by the pulse at 2.5 h; a piece with no event strictly
    inside is the label itself, one-hot, and no quotient of overlap and length is formed for it."""
    pulses, T, _ = CASES["B"]
    edges = np.array(_edges(pulses, T))
    f_comp, f_bound = occupancy_pieces(_arrays([(COMP0, BOUND_RUN), (J_COMP, J_BOUND)]), COMP, edges, 2, bound_column=BOUND)
    # piece [h, 2h]: bound on [1.25 h, 2 h] -> 3/4; [2h, 2.5h]: 1; [2.5h, 3h]: 1/2 (the run ends at 2.75 h)
    for i, want in ((1, 0.75), (2, 1.0), (3, 0.5)):
        assert f_bound[0, i] == pytest.approx(want, rel=1e-14), i
        assert f_comp[0, i, 0] == pytest.approx(1.0 - want, abs=1e-14), i
        assert f_comp[0, i, 1] == 0.0
    assert f_bound[0, 0] == 0.0 and np.array_equal(f_comp[0, 0], [1.0, 0.0])        # no event inside: the label, exactly
    assert f_bound[0, 6] == 0.0 and np.array_equal(f_comp[0, 6], [1.0, 0.0])
    # the joint average of the second walker over [0, h]: f_0 = 0, f_1 = 1/2, f_bound = 1/2 (a piece from 0 to h/2 is not cut)
    e2 = np.array([0.0, H])
    f_c, f_b = occupancy_pieces(_arrays([(J_COMP, J_BOUND)]), COMP, e2, 2, bound_column=BOUND)
    assert np.allclose(f_c[0, 0], [0.0, 0.5], atol=1e-15) and f_b[0, 0] == pytest.approx(0.5, abs=1e-15)
    assert f_c.sum(axis=2)[0, 0] + f_b[0, 0] == pytest.approx(1.0, abs=1e-15)       # the simplex


def test_the_propagator_takes_per_piece_occupancies_only():
    """The occupancy of a block end, one per block end, is not a spelling the propagator reads."""
    n_t = 5
    traj, G = np.zeros((1, n_t, 3)), np.zeros((1, n_t, 3))
    rf = [RFEvent(0.0, 90.0, "x", axis_deg=0.0)]
    with pytest.raises(ValueError, match="PIECE"):
        replay_bloch(traj, H, G, H, rf, T2_per_comp=[0.1, 0.2], T1_per_comp=[1.0, 1.0], comp_traj=np.zeros((1, n_t), int))
    with pytest.raises(ValueError, match="PIECE"):
        replay_bloch(traj, H, G, H, rf, T2=0.1, T1=1.0, bound_frac=np.zeros((1, n_t)), T2_bound=1e-3, T1_bound=0.05)


# ---- the scalar routes ----------------------------------------------------------------------------------------------

def _hand_logw(comp_ev, bound_ev, R2c, R2b):
    """``-int_0^TE R2(t) dt`` of one walker transverse throughout, summed by hand over the merged runs."""
    cuts = sorted({0.0, TE} | {t for t, _ in comp_ev + bound_ev if 0.0 < t < TE})
    tot = 0.0
    for u, v in zip(cuts[:-1], cuts[1:]):
        g, b = _label_at(comp_ev, u), _label_at(bound_ev, u)
        tot += (v - u) * ((1 - b) * R2c[g] + b * R2b)
    return -tot


def test_the_scalar_runs_weigh_the_joint_occupancy():
    """A spin echo of 50 ms (transverse throughout), ``R2`` 10 / 200 / 1000 per pool. Walker 0: bound from 1 ms to 11 ms
    in geometric pool 0 -> ``-(10 * 0.040 + 1000 * 0.010)``; walker 1: geometric pool 1 on [5, 8] ms inside the bound run
    (``f_1 = (1 - b) [comp = 1] = 0`` there) -> the same; walker 2: geometric pool 1 on [0, 4] ms, bound on [2, 6] ms ->
    ``-(200 * 0.002 + 1000 * 0.004 + 10 * 0.044)``."""
    ms = 1e-3
    walkers = [(COMP0, [(0.0, 0), (1 * ms, 1), (11 * ms, 0)]),
               ([(0.0, 0), (5 * ms, 1), (8 * ms, 0)], [(0.0, 0), (1 * ms, 1), (11 * ms, 0)]),
               ([(0.0, 1), (4 * ms, 0)], [(0.0, 0), (2 * ms, 1), (6 * ms, 0)])]
    arrays = _arrays(walkers)
    gate = CumulativeGate(np.ones(int(round(TE / H)) + 1), H)
    got = relaxation_logweight_runs(arrays, COMP, [1 / R2[0], 1 / R2[1]], [np.inf, np.inf], gate, gate, t0=0.0, t1=TE,
                                    bound_column=BOUND, T2_b=T2_B, T1_b=np.inf)
    want = [-(10 * 0.040 + 1000 * 0.010), -(10 * 0.040 + 1000 * 0.010), -(200 * 0.002 + 1000 * 0.004 + 10 * 0.044)]
    assert np.allclose(got, want, rtol=1e-12, atol=0.0), (got, want)
    hand = [_hand_logw(c, b, R2, R2[2]) for c, b in walkers]
    assert np.allclose(got, hand, rtol=1e-12, atol=0.0)
    assert want[0] == pytest.approx(-10.4, rel=1e-12)


# ---- a pack with a bound column --------------------------------------------------------------------------------------

def _read(path):
    st, hdr = read_safetensors(path)
    return dict(st), json.loads(hdr["rpk"])


def _bound_pack(n_bound_walkers=4000):
    """The committed ``slab_exchange.rpk`` with a ``bound`` column: the first ``n_bound_walkers`` walkers bound from 1 ms
    to 11 ms, the others never."""
    st, meta = _read(DATA / "slab_exchange.rpk")
    n_w = meta["walk_params"]["n_walkers"]
    bound = [[(0.0, 0), (1e-3, 1), (11e-3, 0)] if w < n_bound_walkers else [(0.0, 0)] for w in range(n_w)]
    st.update({k: v for k, v in _arrays([(COMP0, b) for b in bound]).items() if k.startswith("bound_")})   # comp stays the pack's
    meta["compression"]["channels"]["compartment"]["columns"].append({"name": "bound", "kind": "events"})
    return ReplayPack(st, meta), bound


@pytest.fixture(scope="module")
def spin_echo():
    n_t = int(round(TE / 1e-4)) + 1
    return dr.pgse([[1.0, 0, 0]], 0.1 * TE, 0.4 * TE, bvalues=[0.0], TE=TE, n_t=n_t)


TISSUE = Tissue(T2={"intra": 0.080, "extra": 0.020})
R2_POOLS = (1 / 0.020, 1 / 0.080)             # by pool id on the committed pack: id 0 is "extra", id 1 "intra" (walkers start in id 1)


@pytest.fixture(scope="module")
def bound_pack():
    return _bound_pack()


def _comp_events(pack):
    col = next(d for d in pack.meta["compression"]["channels"]["compartment"]["columns"] if d["name"] == "comp")
    start, ev = occupancy_events(pack.arrays, col)
    return [[(0.0, int(start[i]))] + [(float(t), int(b)) for t, b in zip(ev.t[ev.offsets[i]:ev.offsets[i + 1]],
                                                                         ev.label[ev.offsets[i]:ev.offsets[i + 1]])]
            for i in range(len(start))]


def test_the_scalar_route_and_the_study_route_read_the_bound_column(bound_pack, spin_echo):
    pack, bound = bound_pack
    comp = _comp_events(pack)
    rng = np.random.default_rng(0)
    pick = np.concatenate([rng.choice(4000, 40, replace=False), 4000 + rng.choice(4000, 20, replace=False),
                           np.flatnonzero([len(c) > 1 for c in comp])[:20]])
    t2b, t1b = 1e-3, 0.05
    w, ew, E = pack.walker_signals(spin_echo, tissue=TISSUE, tol=None, T2_b=t2b, T1_b=t1b)
    w, ew = np.asarray(w), np.asarray(ew)
    want = np.array([_hand_logw(comp[i], bound[i], R2_POOLS, 1 / t2b) for i in pick])
    assert np.allclose(np.log(ew[pick] / w[pick]), want, rtol=1e-12, atol=0.0)
    # the bound walkers are not in the free pool for those 10 ms: the same walkers without the column read lose less
    free = dr.read_rpk(DATA / "slab_exchange.rpk")
    w0, ew0, _ = free.walker_signals(spin_echo, tissue=TISSUE, tol=None)
    assert np.all(np.asarray(ew)[:4000] < np.asarray(ew0)[:4000]) and np.array_equal(np.asarray(ew)[4000:], np.asarray(ew0)[4000:])
    # the replay is the walker sum
    S = pack.replay(spin_echo, tissue=TISSUE, tol=None, complex_signal=True, T2_b=t2b, T1_b=t1b)
    assert np.allclose(S, (ew[:, None] * np.asarray(E)).sum(0) / w.sum(), rtol=1e-12, atol=0.0)
    # the study route: one exposure column more, the bound pool's, from the joint occupancy; its signals are the same numbers
    prim = pack.walker_primitives(spin_echo, tol=None)
    assert prim.has_bound and prim.exposure_t2.shape[1] == 3 and prim.n_pools == 3
    assert np.allclose(prim.exposure_t2[:4000, 2], 0.010, rtol=1e-12, atol=0.0) and np.all(prim.exposure_t2[4000:, 2] == 0.0)
    assert np.allclose(prim.exposure_t2.sum(axis=1), TE, rtol=1e-12, atol=0.0)          # the simplex: every instant in one pool
    w1, ew1, E1 = prim.signals(TISSUE, T2_b=t2b, T1_b=t1b)
    assert np.allclose(ew1, ew, rtol=1e-12, atol=0.0) and np.allclose(E1, np.asarray(E), rtol=1e-12, atol=1e-12)
    want_study = np.array([_hand_logw(comp[i], bound[i], R2_POOLS, 1 / t2b) for i in pick])
    assert np.allclose(np.log(ew1[pick] / w1[pick]), want_study, rtol=1e-12, atol=0.0)


def test_the_bloch_route_reads_the_bound_column(bound_pack, spin_echo):
    """A spin echo is transverse throughout, so the scalar route (exact over the runs of both columns) is the reference of
    the Bloch route per walker; both propagators, with ``delta_omega_b`` 0."""
    pack, _ = bound_pack
    t2b, t1b = 1e-3, 0.05
    w, ew, E = pack.walker_signals(spin_echo, tissue=TISSUE, tol=None, T2_b=t2b, T1_b=t1b)
    sc = np.asarray(ew) * np.asarray(E)[:, 0] / np.asarray(w)
    kw = dict(tissue=TISSUE, per_walker=True, complex_signal=True, T2_b=t2b, T1_b=t1b, delta_omega_b=0.0)
    b_np = np.asarray(pack.replay_bloch(spin_echo, **kw))[:, 0]
    assert np.abs(np.abs(b_np) - np.abs(sc)).max() < 1e-12
    free = np.abs(np.load(DATA / "bound_regression_golden.npz")["bloch_walker"][:, 0])        # the same pack without the column
    assert np.all(np.abs(b_np[:4000]) < free[:4000] - 1e-6)                       # the column is read
    pytest.importorskip("jax")
    b_jx = np.asarray(pack.replay_bloch(spin_echo, backend="jax", **kw))[:, 0]
    assert np.abs(np.abs(b_jx) - np.abs(b_np)).max() < 1e-5


def test_the_bound_pool_knobs_are_required_and_refused_by_name(bound_pack, spin_echo):
    pack, _ = bound_pack
    free = dr.read_rpk(DATA / "slab_exchange.rpk")
    def refused(text, fn, *a, **kw):
        with pytest.raises(ValueError, match=re.escape(text)):
            fn(*a, **kw)
    t = dict(tissue=TISSUE)
    refused(TEXT_NEEDS_T2_T1, pack.replay, spin_echo, T1_b=0.05, **t)                                  # no T2_b
    refused(TEXT_NEEDS_T2_T1, pack.replay, spin_echo, T2_b=1e-3, **t)                                  # no T1_b
    refused(TEXT_NEEDS_T2_T1, pack.walker_signals, spin_echo, **t)                                     # no knob at all
    refused(TEXT_NEEDS_T2_T1, pack.replay_bloch, spin_echo, T1_b=0.05, delta_omega_b=0.0, **t)
    refused(TEXT_NEEDS_DELTA, pack.replay_bloch, spin_echo, T2_b=1e-3, T1_b=0.05, **t)                 # a Bloch read without delta_omega_b
    refused(TEXT_SCALAR_DELTA, pack.replay, spin_echo, T2_b=1e-3, T1_b=0.05, delta_omega_b=0.0, **t)    # delta_omega_b to a scalar route
    refused(TEXT_SCALAR_DELTA, pack.walker_signals, spin_echo, T2_b=1e-3, T1_b=0.05, delta_omega_b=0.0, **t)
    refused(TEXT_NO_COLUMN, free.replay, spin_echo, T2_b=1e-3, **t)                                    # a pack without the column
    refused(TEXT_NO_COLUMN, free.replay, spin_echo, delta_omega_b=0.0, **t)
    refused(TEXT_NO_COLUMN, free.replay_bloch, spin_echo, T1_b=0.05, **t)
    prim = free.walker_primitives(spin_echo, tol=None)
    refused(TEXT_NO_COLUMN, prim.signals, TISSUE, T2_b=1e-3, T1_b=0.05)
    refused(TEXT_NEEDS_T2_T1, pack.walker_primitives(spin_echo, tol=None).signals, TISSUE)


# ---- compatibility: a pack without the column, against the client as it was ------------------------------------------

def _with_event_in(pack, t_hi):
    return np.array([any(0.0 < t <= t_hi for t, _ in ev[1:]) for ev in _comp_events(pack)])


def test_a_pack_without_the_bound_column_reads_as_it_did(spin_echo):
    """(i) The scalar route is bit for bit what it was. (ii) The Bloch route is bit for bit what it was for every walker whose
    ``comp`` holds no event in ``(0, TE]`` and agrees with the exact scalar route for every walker that has one (the golden
    value of 1758 of them was more than 1e-9 off)."""
    g = np.load(DATA / "bound_regression_golden.npz")
    pack = dr.read_rpk(DATA / "slab_exchange.rpk")
    w, ew, E = pack.walker_signals(spin_echo, tissue=TISSUE, tol=None)
    assert np.array_equal(np.asarray(w), g["scalar_w"]) and np.array_equal(np.asarray(ew), g["scalar_ew"])
    assert np.array_equal(np.asarray(E), g["scalar_E"])
    assert np.array_equal(np.asarray(pack.replay(spin_echo, tissue=TISSUE, complex_signal=True, tol=None)), g["scalar_mean"])
    prim = pack.walker_primitives(spin_echo, tol=None)
    assert not prim.has_bound and prim.exposure_t2.shape[1] == 2
    w2, ew2, E2 = prim.signals(TISSUE)
    assert np.array_equal(ew2, g["study_ew"]) and np.array_equal(E2, g["study_E"])
    assert np.array_equal(np.asarray(prim.signal(TISSUE, complex_signal=True)), g["study_mean"])

    has = _with_event_in(pack, TE)
    assert np.array_equal(has, g["has_event"])                                   # the recomputed set on (0, TE]
    bl = np.asarray(pack.replay_bloch(spin_echo, tissue=TISSUE, per_walker=True, complex_signal=True))
    exact = np.abs(np.asarray(ew) * np.asarray(E)[:, 0] / np.asarray(w))
    old = g["bloch_walker"]
    scalar_old = np.abs(g["scalar_ew"] * g["scalar_E"][:, 0] / g["scalar_w"])
    changed_old = np.abs(np.abs(old[:, 0]) - scalar_old) > 1e-9                 # the unmodified client's departure from the exact runs
    counts = dict(no_event=int((~has).sum()), with_event=int(has.sum()), old_off_by_more_than_1e9=int(changed_old.sum()),
                  new_differs_from_golden=int((np.abs(np.abs(bl[:, 0]) - np.abs(old[:, 0])) > 1e-9).sum()))
    print(counts)
    assert counts == dict(no_event=6234, with_event=1766, old_off_by_more_than_1e9=1758, new_differs_from_golden=1758)
    assert np.array_equal(bl[~has], old[~has])                                   # the protected set: bit for bit
    assert np.abs(np.abs(bl[has, 0]) - exact[has]).max() < 1e-12                 # the others: the exact runs
