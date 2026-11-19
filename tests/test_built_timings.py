"""``Encoding.delta_built`` and ``Encoding.Delta_built``: the pulsed pair of a built ``pgse`` / ``pgste`` as it is played,
read off the stored samples of its effective gradient (half-amplitude width, centre-to-centre separation), beside the
nominal ``delta`` and ``Delta``, which keep their values; None on every other family."""
from dataclasses import replace
import itertools

import numpy as np
import pytest

import dmipy_replay as dr
from dmipy_replay.sequences import fexi, from_waveform, instantaneous
from dmipy_replay.sequences.assemble import pair_timings

DIRS = [[0.0, 0.0, 1.0], [0.6, 0.0, 0.8]]


def _crossings(g, dt):
    """This test's own measurement: ``[(t_up, t_down), ...]`` of the first two lobes of ``g`` (runs of non-zero samples,
    sample ``k`` at ``k dt``, zero outside the grid), each where ``|g|`` crosses half that lobe's own peak, interpolated
    linearly between the samples about the crossing."""
    a = np.r_[0.0, np.abs(g), 0.0]
    t = (np.arange(a.size) - 1) * dt
    nz = np.flatnonzero(a)
    runs = np.split(nz, np.flatnonzero(np.diff(nz) > 1) + 1)
    out = []
    for r in runs[:2]:
        seg, ts = a[r[0] - 1:r[-1] + 2], t[r[0] - 1:r[-1] + 2]
        h = seg.max() / 2
        i = np.flatnonzero((seg[:-1] < h) & (seg[1:] >= h))[0]
        j = np.flatnonzero((seg[:-1] >= h) & (seg[1:] < h))[-1]
        out.append((ts[i] + (h - seg[i]) / (seg[i + 1] - seg[i]) * dt, ts[j] + (seg[j] - h) / (seg[j] - seg[j + 1]) * dt))
    return out


def _measured(seq, m):
    """``(width, separation, last crossing)`` of row ``m`` by :func:`_crossings` on the sequence's stored samples."""
    d = np.asarray(seq.encoding.gradient_directions[m], np.float64)
    g = np.asarray(seq.G_eff, np.float64)[m] @ (d / np.linalg.norm(d))
    (u1, d1), (u2, d2) = _crossings(g, seq.dt)
    return 0.5 * ((d1 - u1) + (d2 - u2)), 0.5 * (u2 + d2) - 0.5 * (u1 + d1), d2


def _build(family, delta, Delta, te, n_t, slew):
    if family == "pgse":
        TE = None if te is None else Delta + delta + 0.02
        return dr.pgse(DIRS, delta, Delta, bvalues=[2e8, 1e9], TE=TE, n_t=n_t, slew_rate=slew)
    TM = Delta - delta
    TE = None if te is None else 2 * delta + TM + 0.01
    return dr.pgste(DIRS, delta, TM, bvalues=[2e8, 1e9], TE=TE, n_t=n_t, slew_rate=slew)


SCAN = list(itertools.product(["pgse", "pgste"], [(5e-3, 12e-3), (9e-3, 29e-3), (7.3e-3, 20e-3)], [None, "given"],
                              [100, 200, 500, 1000], [200.0, np.inf]))


@pytest.mark.parametrize("family,timings,te,n_t,slew", SCAN)
def test_the_built_pair_is_the_tests_own_measurement_of_the_stored_samples(family, timings, te, n_t, slew):
    seq = _build(family, *timings, te, n_t, slew)
    e = seq.encoding
    for m in range(2):
        width, sep, last = _measured(seq, m)
        tol = 8 * np.spacing(last)                                       # a few float64 ulps of the crossing time
        assert abs(e.delta_built[m] - width) <= tol
        assert abs(e.Delta_built[m] - sep) <= tol


#: (family, delta, Delta or TM, TE, the built half-amplitude width (s), the built centre-to-centre separation (s)),
#: one direction, b 2e8 s/m^2, n_t 1000, slew 200 T/m/s
FIGURES = [("pgse", 5e-3, 12e-3, None, 5.000000e-3, 12.025009e-3),
           ("pgse", 9e-3, 29e-3, None, 9.000000e-3, 29.009219e-3),
           ("pgse", 7.3e-3, 20e-3, None, 7.300000e-3, 20.010523e-3),
           ("pgse", 9e-3, 29e-3, 0.0701, 9.000000e-3, 29.050450e-3),
           ("pgste", 4e-3, 10e-3, None, 4.000000e-3, 14.600808e-3)]


def _figure_seq(family, a, b, te):
    if family == "pgse":
        return dr.pgse([[0, 0, 1]], a, b, bvalues=[2e8], TE=te, n_t=1000, slew_rate=200.0)
    return dr.pgste([[0, 0, 1]], a, b, bvalues=[2e8], n_t=1000, slew_rate=200.0)


@pytest.mark.parametrize("family,a,b,te,width,sep", FIGURES)
def test_the_built_pair_has_its_figures(family, a, b, te, width, sep):
    seq = _figure_seq(family, a, b, te)
    # the figures' rounding, or the float32 storage of G over the ramp's slope, whichever is larger
    tol = max(1e-9, 2 * float(np.spacing(np.float32(np.abs(seq.G).max()))) / 200.0)
    assert abs(seq.encoding.delta_built[0] - width) <= tol
    assert abs(seq.encoding.Delta_built[0] - sep) <= tol


def test_the_pair_is_the_first_two_lobes_not_a_later_readout_lobe():
    seq = _figure_seq("pgse", 5e-3, 12e-3, None)
    G = np.asarray(seq.G_eff, np.float64)
    on = np.flatnonzero(np.abs(G[0, :, 2]))
    readout = np.zeros((1, 400, 3)); readout[0, 100:300] = G[0, on[:200]]   # lobe 1's first 200 samples, played again
    G_ro = np.concatenate([G, readout], axis=1)                          # a third lobe after the echo
    width, sep = pair_timings(G_ro, [[0, 0, 1]], seq.dt)
    assert abs(sep[0] - 12.025009e-3) <= 1e-9 and abs(width[0] - 5e-3) <= 1e-9
    # lobe 1 and the readout lobe (lobe 2 taken out) are another pair: picking them misses the figure by milliseconds
    lobe2 = on[on > on[0] + int(round(7e-3 / seq.dt))]
    G_wrong = G_ro.copy(); G_wrong[0, lobe2] = 0.0
    assert abs(pair_timings(G_wrong, [[0, 0, 1]], seq.dt)[1][0] - 12.025009e-3) > 1e-3


def test_a_lobe_abutting_one_of_the_opposite_sign_is_read_through_zero():
    # samples at k dt: a negative lobe on 2-3 directly followed by a positive one on 4-6; g linear between samples
    # crosses -1/2 at 1.5 and 3.25, +1/2 at 3.75 and 6.5
    g = np.array([0, 0, -1, -1, 1, 1, 1, 0, 0], float)
    G = np.zeros((1, g.size, 3)); G[0, :, 2] = g
    width, sep = pair_timings(G, [[0, 0, 1]], 1.0)
    assert width[0] == pytest.approx(0.5 * ((3.25 - 1.5) + (6.5 - 3.75)), abs=1e-15)
    assert sep[0] == pytest.approx(0.5 * (3.75 + 6.5) - 0.5 * (1.5 + 3.25), abs=1e-15)


#: the nominal values on the base commit, recorded as float.hex: delta and Delta are unchanged bit for bit
NOMINAL = [("0x1.47ae147ae147bp-8", "0x1.89374bc6a7efap-7"), ("0x1.26e978d4fdf3bp-7", "0x1.db22d0e560419p-6"),
           ("0x1.de69ad42c3c9fp-8", "0x1.47ae147ae147bp-6"), ("0x1.26e978d4fdf3bp-7", "0x1.db22d0e560419p-6"),
           ("0x1.0624dd2f1a9fcp-8", "0x1.cac083126e979p-7")]


@pytest.mark.parametrize("fig,nominal", list(zip(FIGURES, NOMINAL)))
def test_the_nominal_delta_and_Delta_are_unchanged(fig, nominal):
    e = _figure_seq(*fig[:4]).encoding
    assert float(e.delta[0]).hex() == nominal[0] and float(e.Delta[0]).hex() == nominal[1]


def test_a_row_with_no_pair_states_nan():
    e = dr.pgse(DIRS, 5e-3, 12e-3, bvalues=[0.0, 1e9], n_t=500, slew_rate=200.0).encoding
    assert np.isnan(e.delta_built[0]) and np.isnan(e.Delta_built[0])
    assert np.isfinite(e.delta_built[1]) and np.isfinite(e.Delta_built[1])


def test_the_built_timings_survive_the_encodings_round_trips():
    seq = _figure_seq("pgse", 5e-3, 12e-3, None)
    e = seq.encoding
    copied = replace(e)
    assert copied.delta_built[0] == e.delta_built[0] and copied.Delta_built[0] == e.Delta_built[0]
    for view in (dr.set_b(seq, [1e9]), dr.rotate_waveform(seq, theta=0.7)):
        assert view.encoding.delta_built[0] == e.delta_built[0] and view.encoding.Delta_built[0] == e.Delta_built[0]
        width, sep, last = _measured(view, 0)                            # still what the view's samples play
        tol = 2 * float(np.spacing(np.float32(np.abs(view.G).max()))) / 200.0 + 8 * np.spacing(last)
        assert abs(width - e.delta_built[0]) <= tol and abs(sep - e.Delta_built[0]) <= tol
    square = instantaneous(seq)                                          # the build_spec rebuilt at infinite slew
    width, sep, _ = _measured(square, 0)
    assert square.encoding.delta_built[0] == pytest.approx(width, abs=1e-15)
    assert square.encoding.Delta_built[0] == pytest.approx(sep, abs=1e-15)


def test_the_other_families_state_no_built_pair():
    sq = dr.pgse([[0, 0, 1]], 5e-3, 12e-3, bvalues=[1e9], n_t=500, slew_rate=np.inf)
    others = [dr.ogse([[0, 0, 1]], 50.0, 0.02, bvalues=[1e9], n_t=500),
              dr.gre(0.04, gradient_directions=[[0, 0, 1]], bvalues=[1e8], delta=5e-3, Delta=10e-3, n_t=500),
              dr.cpmg(2, 0.004),
              dr.splice([[1.0, 0.0, 0.0]], 15e-3, 25e-3, 3, 10e-3, bvalues=[1e9], TE_prep=80e-3, beta_deg=150.0,
                        n_t_per_echo=40),
              dr.ste(0.02, bvalues=[1e9], n_t=500, slew_rate=np.inf),
              fexi([1.0, 0.0, 0.0], 4e-3, 10e-3, 0.1, b_f=[1e9], bvalues=[0.8e9], dt=1e-5, slew_rate=np.inf),
              from_waveform(np.asarray(sq.G_eff), sq.dt, [[0, 0, 1]], delta=5e-3, Delta=12e-3)]
    for seq in others:
        assert seq.encoding.delta_built is None and seq.encoding.Delta_built is None, seq.family
