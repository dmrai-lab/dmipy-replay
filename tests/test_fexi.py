"""The filter-exchange builder: one ScannerSequence of three 90s and two self-refocused bipolar blocks, its crushers
bracketing the storage, the b-values of its two blocks read back off the object, its standard protocol, and its replay
through the one effective-gradient contraction at the stimulated echo's amplitude."""
from pathlib import Path

import numpy as np
import pytest

import dmipy_replay as dr
from dmipy_analytic.constants import GAMMA
from dmipy_analytic.epg import ste_amplitude
from dmipy_replay.acquisition.rf import role_of
from dmipy_replay.sequences import fexi, fexi_bvalues, fexi_protocol, from_waveform, instantaneous

U32 = 2.0 ** -24
DT = 1e-5
DELTA, BIG_DELTA, T_M = 4e-3, 10e-3, 0.1


def _seq(**kw):
    args = dict(b_f=[0.0, 0.0, 1.0e9, 1.0e9], bvalues=[0.0, 0.8e9, 0.0, 0.8e9], dt=DT, slew_rate=np.inf)
    args.update(kw)
    return fexi([1.0, 0.0, 0.0], DELTA, BIG_DELTA, T_M, **args)


def test_three_ninety_degree_pulses_store_and_recall_and_there_is_no_refocusing_pulse():
    seq = _seq()
    assert [role_of(e) for e in seq.rf] == ["excite", "store", "recall"]
    assert [e.flip_deg for e in seq.rf] == [90.0, 90.0, 90.0] and seq.rf.refocus_time is None
    assert seq.stimulated_echo and seq.TM == pytest.approx(T_M, rel=1e-12) and seq.family == "fexi"
    assert seq.rf.gate_integral(seq.T) == pytest.approx(0.0, abs=1e-15)     # equal transverse times: statics refocus
    assert seq.T == pytest.approx(T_M + 2 * (seq.rf[1].t_s - seq.rf[0].t_s), rel=1e-12)


def test_each_block_is_a_bipolar_pair_that_refocuses_on_its_own():
    seq = _seq()
    i_store, i_recall = (int(round(e.t_s / seq.dt)) for e in seq.rf[1:])
    G = np.asarray(seq.G, np.float64)
    q = np.cumsum(G * seq.dt, axis=1)
    qmax = np.abs(q).max()
    assert np.abs(q[:, i_store - 1]).max() <= 1e-6 * qmax and np.abs(q[:, -2]).max() <= 1e-6 * qmax
    assert np.all(G[:, i_store:i_recall] == 0.0)                            # nothing encodes while stored
    f = G[2, :i_store, 0]
    on = np.flatnonzero(f)
    assert f[on[0]] > 0 and f[on[-1]] < 0 and np.all(f[on[0]:on[0] + int(round(DELTA / DT))] == f[on[0]])
    assert seq.refocusing_residual < 1e-6


def test_the_block_b_values_are_what_was_asked():
    """``fexi_bvalues`` reads the requests back to the float32 rounding of the stored amplitude (``b`` is quadratic in
    it, ``4 U32``); the encoding's b is the whole effective gradient's, the two added."""
    seq = _seq()
    b_f, b_d = fexi_bvalues(seq)
    assert b_f == pytest.approx([0.0, 0.0, 1.0e9, 1.0e9], rel=4 * U32, abs=0.0)
    assert b_d == pytest.approx([0.0, 0.8e9, 0.0, 0.8e9], rel=4 * U32, abs=0.0)
    assert seq.encoding.bvalues == pytest.approx(b_f + b_d, rel=4 * U32)
    assert seq.b() == pytest.approx(seq.encoding.bvalues, rel=1e-12)


def test_a_square_block_plays_the_bipolar_stejskal_tanner_amplitude():
    """``b = gamma^2 g^2 delta^2 (Delta - delta / 3)`` for rectangular lobes; the package's b integral (the trapezoid
    of ``|k|^2``) over-reads a lobe's quadratic ``|q|^2`` by ``h^2 / 12`` of its second derivative, ``(h^2 / (3 delta
    (Delta - delta / 3)))`` of b in all, so the amplitude sits below the closed form by half that."""
    seq = _seq()
    g = np.abs(np.asarray(seq.G, np.float64)[2, :, 0]).max()
    g_st = np.sqrt(1.0e9 / (GAMMA ** 2 * DELTA ** 2 * (BIG_DELTA - DELTA / 3.0)))
    rel = DT ** 2 / (3.0 * DELTA * (BIG_DELTA - DELTA / 3.0))
    assert abs(g / g_st - 1.0) <= rel


def test_the_crushers_bracket_the_storage_in_the_transverse_periods():
    """One crusher ends at the store and one starts at the recall, each ``crusher_duration`` long and of the same
    winding: the pair the stimulated echo is selected by. A crusher during the mixing time alone would also keep the
    pathway that adds the two blocks' phases and the magnetisation recovered along B0."""
    seq = _seq(crusher_duration=1.5e-3, crusher_cycles=12.0)
    (a0, a1), (b0, b1) = seq.crusher["windows_s"]
    t_store, t_recall = seq.rf[1].t_s, seq.rf[2].t_s
    assert a1 == pytest.approx(t_store) and b0 == pytest.approx(t_recall)
    assert a1 - a0 == pytest.approx(1.5e-3) and b1 - b0 == pytest.approx(1.5e-3)
    assert seq.crusher["n_cycles"] == 12.0 and seq.voxel_declared
    assert np.all(np.asarray(seq.G)[:, int(round(a0 / DT)):int(round(a1 / DT))] == 0.0)


def test_the_finite_pulse_budget_keeps_the_blocks_off_the_pulses_and_the_statics_refocused():
    tm = dr.SequenceTiming(t_excite=2e-3, t_refocus=4e-3, t_readout_pre_echo=8e-3)
    seq = fexi([0.0, 1.0, 0.0], DELTA, BIG_DELTA, 0.05, b_f=[0.0, 1e9], bvalues=[1e9, 0.0], dt=DT, timing=tm)
    assert all(e.duration_s == 2e-3 for e in seq.rf) and seq.timing is tm
    assert seq.rf.gate_integral(seq.T) == pytest.approx(0.0, abs=1e-15)
    seq.validate()                                                          # off through every pulse and dead window
    assert fexi_bvalues(seq)[0] == pytest.approx([0.0, 1e9], rel=4 * U32)


def test_the_square_limit_rebuilds_with_vertical_ramps():
    seq = fexi([1.0, 0.0, 0.0], DELTA, BIG_DELTA, T_M, b_f=[0.0, 1.5e9], bvalues=[0.5e9, 0.0], dt=DT, slew_rate=200.0)
    assert np.all(seq.encoding.ramp_time > 0)
    sq = instantaneous(seq)
    assert np.all(sq.encoding.ramp_time == 0.0) and fexi_bvalues(sq)[0] == pytest.approx([0.0, 1.5e9], rel=4 * U32)


def test_off_grid_times_and_a_missing_crusher_are_refused():
    with pytest.raises(ValueError, match="whole number"):
        fexi([1.0, 0.0, 0.0], DELTA, BIG_DELTA, 0.1000049, b_f=1e9, bvalues=0.0, dt=DT, slew_rate=np.inf)
    with pytest.raises(ValueError, match="crusher"):
        fexi([1.0, 0.0, 0.0], DELTA, BIG_DELTA, T_M, b_f=1e9, bvalues=0.0, dt=DT, slew_rate=np.inf, crusher_duration=0.0)


def test_a_sequence_without_separate_blocks_has_no_filter_b_value():
    with pytest.raises(ValueError, match="net moment at the store"):
        fexi_bvalues(dr.pgste([[1.0, 0.0, 0.0]], DELTA, 0.05, bvalues=[1e9], slew_rate=np.inf))
    with pytest.raises(ValueError, match="store / recall"):
        fexi_bvalues(dr.pgse([[1.0, 0.0, 0.0]], DELTA, BIG_DELTA, bvalues=[1e9], slew_rate=np.inf))


def test_the_protocol_is_filter_by_mixing_time_by_detection():
    t_m = [0.02, 0.1, 0.3]
    P = fexi_protocol([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], DELTA, BIG_DELTA, t_m, b_f=1.2e9, bvalues=[0.0, 0.5e9, 1e9],
                      dt=DT, slew_rate=np.inf)
    assert isinstance(P, dr.Protocol) and len(P) == 3 and P.n_meas == 3 * 2 * 3 * 2
    for seq, tm in zip(P, t_m):
        b_f, b_d = fexi_bvalues(seq)
        assert seq.TM == pytest.approx(tm, rel=1e-12)
        assert b_f == pytest.approx(np.repeat([0.0, 1.2e9], 6), rel=4 * U32)
        assert b_d == pytest.approx(np.tile(np.repeat([0.0, 0.5e9, 1e9], 2), 2), rel=4 * U32)
        assert seq.encoding.gradient_directions[:2] == pytest.approx(np.array([[1, 0, 0], [0, 0, 1]]))
    with pytest.raises(ValueError, match="positive"):
        fexi_protocol([1.0, 0.0, 0.0], DELTA, BIG_DELTA, t_m, b_f=0.0, bvalues=[0.0], dt=DT)


def test_a_replay_is_the_effective_gradient_contraction_at_the_stimulated_echo_amplitude():
    """On an impermeable cylinder pack, the FEXI rows replay to ``ste_amplitude(90, 90, 90)`` times the replay of a
    plain waveform carrying the same effective gradient: nothing in the replay is particular to the family."""
    pack = dr.read_rpk(Path(__file__).parent / "data" / "cylinder.rpk")
    seq = fexi([1.0, 0.0, 0.0], 2e-3, 5e-3, 60 * pack.dt, b_f=[0.0, 0.0, 1e9, 1e9], bvalues=[0.0, 1e9, 0.0, 1e9],
               dt=pack.dt, crusher_duration=3 * pack.dt, slew_rate=np.inf)
    S = np.asarray(pack.replay(seq, tol=None))
    plain = from_waveform(np.asarray(seq.G_eff), seq.dt, seq.encoding.gradient_directions)
    S_plain = np.asarray(pack.replay(plain, tol=None))
    assert S[0] == pytest.approx(ste_amplitude(90.0, 90.0, 90.0), rel=1e-6)
    assert S == pytest.approx(0.5 * S_plain, rel=1e-6)
