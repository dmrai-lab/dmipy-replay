"""A replay is held to the :class:`~dmipy_replay.Envelope` its pack states (RPK.md 11): the waveform's band and its
per-axis amplitude, the field channel's refocusing depth and the field strength, each refused by name, on every route
that takes a waveform; the views are held to their moved envelopes by the same check. A header that states a scanner
class is read as an envelope by the stated rule."""
import re
import sys
from pathlib import Path

import numpy as np
import pytest

import dmipy_replay as dr
from dmipy_replay.replay.columnar import open_columnar
from dmipy_replay.replay.envelope import CLASS_HEADER_RULE
from dmipy_replay.replay.replay import check_acquisition_envelope

DATA = Path(__file__).parent / "data"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
TD = dict(delta=0.006, Delta=0.02)


def pack_of(fixture, **acq):
    """The fixture pack ``fixture`` with its envelope's acquisition fields updated by ``acq`` (a ``None`` drops a key);
    the fixtures state ``G_max = 0.3`` T/m here."""
    p = dr.read_rpk(DATA / f"{fixture}.rpk")
    meta = {k: (dict(v) if isinstance(v, dict) else v) for k, v in p.meta.items()}
    meta["replay_envelope"] = {k: (dict(v) if isinstance(v, dict) else v) for k, v in p.meta["replay_envelope"].items()}
    a = {**meta["replay_envelope"]["acquisition"], "G_max": 0.3, **acq}
    meta["replay_envelope"]["acquisition"] = {k: v for k, v in a.items() if v is not None}
    return type(p)(p.arrays, meta, source=p.source)


def class_pack_of(fixture, **acq):
    """The fixture pack ``fixture`` as a scanner-class header states it (``bound = "scanner_class"``, ``scanner``,
    ``G_max``, ``slew_max``), the form the published packs carry."""
    return pack_of(fixture, **{**dict(bound="scanner_class", scanner="test_class", slew_max=200.0, name=None, band_hz=None),
                               **acq})


def pgse(g=0.1, slew_rate=200.0):
    return dr.pgse([[1.0, 0.0, 0.0]], TD["delta"], TD["Delta"], gradient_strengths=g, slew_rate=slew_rate)


# ------------------------------------------------------------------------------------------------ the object
def test_an_envelope_states_every_field_and_refuses_a_bad_one():
    e = dr.Envelope(name="x", band_hz=500.0, G_max=0.3, B0_list=[3.0, 7.0], chi_max=None, rho2_max=1e-4,
                    max_refocusing_pulses=None)
    assert e.B0_list == (3.0, 7.0) and e.B0_max == 7.0 and e.mode_count(0.1) == 100
    assert dr.Envelope.from_dict(e.to_dict()) == e
    with pytest.raises(TypeError):
        dr.Envelope(name="x", band_hz=500.0, G_max=0.3)                   # no field is defaulted
    with pytest.raises(ValueError, match="missing"):
        dr.Envelope.from_dict({k: v for k, v in e.to_dict().items() if k != "chi_max"})
    for bad in (dict(band_hz=0.0), dict(band_hz=np.inf), dict(G_max=-1.0), dict(chi_max=0.0), dict(name=""),
                dict(max_refocusing_pulses=1.5), dict(B0_list=None)):
        with pytest.raises(ValueError):
            e.replace(**bad)


def test_a_scanner_preset_is_the_catalogues_band_times_the_storage_margin():
    from dmipy_replay.acquisition.scanners import ScannerLimits
    from dmipy_replay.replay.storage import STORAGE_BAND_MARGIN
    for scanner in ("prisma", "connectome_2"):
        e, lim = dr.envelope_of(scanner), ScannerLimits.of(scanner)
        assert e.name == scanner
        assert e.band_hz == pytest.approx(STORAGE_BAND_MARGIN * lim.certified_band_hz)
        assert e.G_max == lim.G_max
        assert (e.B0_list, e.chi_max, e.rho2_max, e.max_refocusing_pulses) == ((), None, None, None)
    assert dr.envelope_of("prisma", B0_list=[3.0], chi_max=1e-7).B0_list == (3.0,)
    with pytest.raises(ValueError, match="no certified band"):
        dr.envelope_of("low_field")


# ------------------------------------------------------------------------------------------------ the check
def test_a_waveform_inside_the_envelope_passes():
    p = pack_of("cylinder")
    assert np.all(np.isfinite(p.replay(pgse(0.1))))
    assert np.all(np.isfinite(p.replay(pgse(0.3))))                      # at G_max


def test_a_gradient_beyond_g_max_is_refused_by_name():
    with pytest.raises(ValueError, match=r"plays 0\.35\d* T/m on one axis, beyond the G_max = 0\.3 T/m"):
        pack_of("cylinder").replay(pgse(0.35))


def test_no_slew_is_checked_the_band_holds_the_ramps():
    p = pack_of("cylinder")
    square = pgse(0.1, slew_rate=np.inf)                                 # switches within one sample
    check_acquisition_envelope(p.meta, square)                           # amplitude inside: no slew to refuse
    hz, bands, _ = p.waveform_band(square)
    assert bands <= p.K
    p.replay(square)
    with pytest.raises(ValueError, match=r"beyond the band of the envelope 'fixture'.*band_hz = 20 Hz"):
        pack_of("cylinder", band_hz=20.0).replay(square)                  # a narrower envelope than the pack stores


def test_a_waveform_beyond_the_envelopes_band_is_refused_by_name():
    p = pack_of("cylinder", band_hz=20.0)
    hz, bands, _ = p.waveform_band(pgse(0.1))
    assert hz > 20.0
    with pytest.raises(ValueError, match=r"needs \S+ Hz .* admits band_hz = 20 Hz"):
        p.replay(pgse(0.1))
    pack_of("cylinder", band_hz=hz).replay(pgse(0.1))                     # at the band it needs


def test_refocusing_pulses_beyond_the_field_channels_depth_are_refused():
    chi = dr.Tissue(chi_iso=-1e-7)
    p = pack_of("field", max_refocusing_pulses=2)
    assert p.envelope.max_refocusing_pulses == 2
    ok, over = dr.cpmg(2, 0.004), dr.cpmg(3, 0.004)
    with pytest.raises(ValueError, match=r"plays 3 refocusing pulses with a field term, beyond the 2"):
        p.replay(over, tissue=chi, scanner=3.0)
    with pytest.raises(ValueError, match="refocusing pulses"):
        p.plan(over, tissue=chi, scanner=3.0)
    p.plan(ok, tissue=chi, scanner=3.0)                                   # at the depth
    p.plan(over, tissue=dr.Tissue(), scanner=3.0)                         # no field term: the field channel is not read


def test_the_refocusing_depth_of_a_windowed_pack_is_a_windows():
    """A pack stored in windows certifies each window's field channel to its depth: a train is held to the pulses it
    plays within one window, so a train across windows within the depth passes and one deeper in a window is refused."""
    p = pack_of("field", max_refocusing_pulses=2)
    T = (p.n_t - 1) * p.dt
    meta = {**p.meta, "walk_params": {**p.meta["walk_params"], "windows": {**p.meta["walk_params"]["windows"], "n": 4, "T": T / 4}}}
    across = dr.cpmg(8, T / 8)                                            # two pulses in each quarter
    deep = dr.cpmg(4, T / 16)                                             # four pulses in the first quarter
    check_acquisition_envelope(meta, across, field_active=True)
    with pytest.raises(ValueError, match=r"plays 4 refocusing pulses within one .* ms storage window"):
        check_acquisition_envelope(meta, deep, field_active=True)
    with pytest.raises(ValueError, match="plays 8 refocusing pulses with a field term"):
        check_acquisition_envelope(p.meta, across, field_active=True)    # one window: the whole train


def test_a_field_strength_is_certified_over_the_range_up_to_the_largest_b0_list_entry():
    p = pack_of("field")
    assert p.envelope.B0_list == (3.0, 7.0)
    chi = dr.Tissue(chi_iso=-1e-7)
    p.replay(dr.gre(0.006), tissue=chi, scanner=5.0)                      # between the listed points
    p.replay(dr.gre(0.006), tissue=chi, scanner=1.5)
    with pytest.raises(ValueError, match=r"B0 = 11\.7 T is beyond the 7 T"):
        p.replay(dr.gre(0.006), tissue=chi, scanner=11.7)


def test_a_susceptibility_beyond_the_certified_field_scale_is_refused():
    with pytest.raises(ValueError, match="certified field scale"):
        pack_of("field").replay(dr.gre(0.006), tissue=dr.Tissue(chi_iso=-1e-5), scanner=7.0)


def test_a_header_without_an_envelope_is_refused_by_name():
    with pytest.raises(ValueError, match="states no envelope"):
        pack_of("cylinder", bound=None).replay(pgse())
    with pytest.raises(ValueError, match="battery of b values"):
        pack_of("cylinder", bound="b_max").replay(pgse())
    with pytest.raises(ValueError, match=r"acquisition\.band_hz"):
        pack_of("cylinder", band_hz=None).replay(pgse())
    with pytest.raises(ValueError, match=r"acquisition\.G_max"):
        class_pack_of("cylinder", G_max=None).replay(pgse())


@pytest.mark.parametrize("route", ["replay", "walker_signals", "walker_phases", "replay_bloch", "plan"])
def test_every_route_applies_the_check(route):
    p = pack_of("cylinder")
    with pytest.raises(ValueError, match="G_max"):
        getattr(p, route)(pgse(0.35))
    getattr(p, route)(pgse(0.1))


def test_the_columnar_plan_applies_the_check():
    from columnar_scenarios import seqs
    seq = seqs(dr)["pgse_short"]                                          # square lobes, 6.97 T/m
    lay = open_columnar(str(DATA / "columnar_quant"))
    lay.plan(seq)                                                         # the fixture envelope is wide
    acq = lay.meta["replay_envelope"]["acquisition"]
    acq.update(G_max=1.0)
    with pytest.raises(ValueError, match=r"plays 6\.9\d* T/m on one axis, beyond the G_max = 1 T/m"):
        lay.plan(seq)
    acq.update(G_max=10.0, band_hz=1.0)                                  # the bands read stop at the envelope's
    assert lay.plan(seq)["K"] <= dr.Envelope.from_header(lay.meta).mode_count(lay.T_window) < lay.K


# ------------------------------------------------------------------------------------------------ the views
def test_the_length_scaled_view_is_held_to_its_moved_amplitude():
    p = pack_of("cylinder")
    smaller, larger = p.at_length_scale(0.5), p.at_length_scale(2.0)
    assert smaller.envelope.G_max == pytest.approx(0.6)
    assert smaller.envelope.band_hz == p.envelope.band_hz                # a frequency: the view keeps the clock
    smaller.replay(pgse(0.5))                                             # beyond the pack's, inside the view's
    with pytest.raises(ValueError, match="G_max"):
        p.replay(pgse(0.5))
    with pytest.raises(ValueError, match=r"G_max = 0\.15 T/m.*length-scaled view, s = 2"):
        larger.replay(pgse(0.2))                                          # inside the pack's, beyond the view's


def test_the_diffusivity_view_moves_band_and_amplitude_with_its_clock():
    p = pack_of("cylinder")
    v = p.at_diffusivity(2.0 * p.diffusivity)
    assert v.envelope.band_hz == pytest.approx(2.0 * p.envelope.band_hz)
    assert v.envelope.G_max == pytest.approx(2.0 * p.envelope.G_max)
    assert v.envelope.band_hz == pytest.approx(v.temporal_bandwidth_hz)


def _pgse_on_view(a, g=0.1, slew=200.0):
    """The walk's ``pgse(g)`` as the view at ``a`` plays it: ``G(t)`` on the view is ``G(t' / a) / a`` on the walk's
    clock ``t' = a t``, so the view's waveform has every time over ``a``, its amplitude times ``a`` and its slew times
    ``a^2`` (``a = 1`` is the walk's own)."""
    return dr.pgse([[1.0, 0.0, 0.0], [0.0, 0.6, 0.8]], TD["delta"] / a, TD["Delta"] / a, gradient_strengths=g * a,
                   slew_rate=slew * a * a)


@pytest.mark.parametrize("a", [0.25, 0.5, 2.0])
def test_the_time_view_is_the_walk_at_the_mapped_waveform_in_both_directions(a):
    """A view at ``a`` replays the walk at the mapped waveform, slower (``a < 1``) as faster, to rounding: the signal,
    the surface term (``rho2`` on the view is ``rho2 / a`` on the walk) and the band the waveform needs (the same bands,
    at ``a`` times the frequency)."""
    p = pack_of("cylinder")
    v = p.at_diffusivity(a * p.diffusivity)
    on_view, on_walk = _pgse_on_view(a), _pgse_on_view(1.0)
    np.testing.assert_allclose(v.replay(on_view, complex_signal=True), p.replay(on_walk, complex_signal=True), rtol=1e-12)
    rho2 = 2e-5
    surface = p.replay(on_walk, tissue=dr.Tissue(rho2=rho2))
    assert not np.allclose(surface, p.replay(on_walk), rtol=1e-3)                       # the contact tier is live
    np.testing.assert_allclose(v.replay(on_view, tissue=dr.Tissue(rho2=a * rho2)), surface, rtol=1e-12)
    (hz_v, bands_v, err_v), (hz, bands, err) = v.waveform_band(on_view), p.waveform_band(on_walk)
    assert bands_v == bands and hz_v == pytest.approx(a * hz) and err_v == pytest.approx(err, rel=1e-6)
    assert v.envelope.band_hz == pytest.approx(a * p.envelope.band_hz) and v.envelope.G_max == pytest.approx(a * p.envelope.G_max)


def test_a_slower_view_holds_the_band_times_a_and_refuses_beyond_it_by_name():
    """A slower view spreads the walk's ``K`` bands over ``T / a``: its band is ``a`` times the walk's. A 300 Hz OGSE
    the walk (800 Hz) and the view at ``a = 1/2`` (400 Hz) serve is refused by the view at ``a = 1/4`` (200 Hz), naming
    the view; the walk refuses the same request mapped onto its clock (1200 Hz), so the guard is the walk's own."""
    p = pack_of("cylinder", G_max=10.0)                                   # the fixture's own amplitude: the band binds first
    seq = dr.ogse([[1.0, 0.0, 0.0]], 300.0, 0.01, shape="cosine", gradient_strengths=1.0, slew_rate=np.inf)
    half, quarter = (p.at_diffusivity(a * p.diffusivity) for a in (0.5, 0.25))
    assert quarter.temporal_bandwidth_hz == pytest.approx(200.0) and quarter.envelope.band_hz == pytest.approx(200.0)
    p.replay(seq)
    half.replay(seq)
    with pytest.raises(ValueError, match=r"beyond this pack's temporal band: it needs \S+ Hz .* resolves 200 Hz "
                                         r"\(K = 64 bands over 160 ms; time-scaled view, a = 0\.25"):
        quarter.replay(seq)
    with pytest.raises(ValueError, match="beyond this pack's temporal band"):
        p.replay(dr.ogse([[1.0, 0.0, 0.0]], 1200.0, 0.0025, shape="cosine", gradient_strengths=4.0, slew_rate=np.inf))


def test_a_slower_view_holds_g_max_times_a_by_name():
    p = pack_of("cylinder")                                               # G_max = 0.3 T/m
    v = p.at_diffusivity(0.25 * p.diffusivity)
    assert v.envelope.G_max == pytest.approx(0.075)
    p.replay(pgse(0.1))
    v.replay(pgse(0.05))
    with pytest.raises(ValueError, match=r"G_max = 0\.075 T/m per axis .*time-scaled view, a = 0\.25"):
        v.replay(pgse(0.1))                                               # inside the walk's, beyond the view's


def test_a_slower_view_holds_the_field_scale_times_a_and_a_faster_one_keeps_the_walks():
    """``B0 chi`` on the view is ``B0 chi / a`` on the walk: a slower view's field scale is ``a`` times the walk's, and
    the view at ``chi`` replays as the walk at ``chi / a``. A faster view keeps the walk's ``chi_max``, inside the
    ``a chi_max`` it could state, so its field modes are planned as the walk's are."""
    p = pack_of("field")                                                  # chi_max 2e-7 up to 7 T
    slow, fast = p.at_diffusivity(0.5 * p.diffusivity), p.at_diffusivity(2.0 * p.diffusivity)
    assert slow.envelope.chi_max == pytest.approx(1e-7) and fast.envelope.chi_max == p.envelope.chi_max
    TE = 0.006
    p.replay(dr.gre(TE), tissue=dr.Tissue(chi_iso=-1.5e-7), scanner=7.0)
    with pytest.raises(ValueError, match="certified field scale"):
        slow.replay(dr.gre(TE / 0.5), tissue=dr.Tissue(chi_iso=-1.5e-7), scanner=7.0)
    np.testing.assert_allclose(slow.replay(dr.gre(TE / 0.5), tissue=dr.Tissue(chi_iso=-0.5e-7), scanner=7.0, complex_signal=True),
                               p.replay(dr.gre(TE), tissue=dr.Tissue(chi_iso=-1e-7), scanner=7.0, complex_signal=True), rtol=1e-12)


def test_a_view_of_a_view_records_a_cumulative_and_the_walks_values():
    p = pack_of("permeable")
    D, kappa = p.diffusivity, p.permeability
    twice = p.at_diffusivity(0.5 * D).at_diffusivity(0.25 * D)
    rec = twice.meta["provenance"]["diffusivity_scaled"]
    assert rec["a"] == pytest.approx(0.25) and rec["walked"] == D and rec["replayed"] == pytest.approx(0.25 * D)
    assert rec["grid_dt"]["walked"] == p.dt and rec["permeability"]["wall_0.in_to_out"]["walked"] == kappa
    once = p.at_diffusivity(0.25 * D)
    seq = _pgse_on_view(0.25)
    np.testing.assert_allclose(twice.replay(seq, complex_signal=True), once.replay(seq, complex_signal=True), rtol=1e-12)


# ------------------------------------------------------------------------------------------------ class headers
def test_a_scanner_class_header_reads_as_its_stored_band_and_amplitude():
    c = class_pack_of("cylinder", slew_max=1.0)                           # a slew the replay no longer reads
    e = c.envelope
    assert (e.name, e.G_max, e.band_hz) == ("test_class", 0.3, c.temporal_bandwidth_hz)
    assert e.rho2_max == pytest.approx(c.meta["replay_envelope"]["tissue"]["rho2_over_D_max"] * c.diffusivity)
    c.replay(pgse(0.1))
    with pytest.raises(ValueError, match=r"G_max = 0\.3 T/m per axis of the envelope 'test_class'"):
        c.replay(pgse(0.35))
    with pytest.raises(ValueError, match="no stored band"):
        dr.Envelope.from_header({**c.meta, "compression": {}})
    assert CLASS_HEADER_RULE == "scanner_class/1"


def test_the_stamp_states_what_the_rule_reads_and_changes_only_the_header(tmp_path):
    from stamp_envelope import stamp_file, stamp_meta
    from dmipy_replay.replay.container import read_safetensors, write_safetensors
    c = class_pack_of("field")
    stamped, change = stamp_meta(c.meta)
    assert change["before"]["bound"] == "scanner_class" and change["after"]["bound"] == "envelope"
    assert stamped["provenance"]["envelope_stamp"]["rule"] == CLASS_HEADER_RULE
    assert dr.Envelope.from_header(stamped) == c.envelope
    assert stamp_meta(stamped)[1] is None                                 # twice is a no-op
    src = tmp_path / "class" / "field.rpk"
    src.parent.mkdir()
    arrays, md = read_safetensors(DATA / "field.rpk")
    import json
    write_safetensors(src, {k: np.array(v) for k, v in arrays.items()}, dict(md, rpk=json.dumps(c.meta)))
    row = stamp_file(src, tmp_path / "out")
    assert row["stamped"]
    a2, m2 = read_safetensors(tmp_path / "out" / "field.rpk")
    assert sorted(a2) == sorted(arrays) and all(np.array_equal(a2[k], arrays[k]) for k in arrays)
    assert dr.read_rpk(tmp_path / "out" / "field.rpk").envelope == c.envelope
    with pytest.raises(ValueError, match="no guarantee to stamp"):
        stamp_meta(pack_of("cylinder", bound="b_max").meta)


# ------------------------------------------------------------------------------------------------ a spent cap
def _capped(p, cap, other=3e-3):
    """``p`` with a stated numerical cap beside one certified non-truncation term of ``other``."""
    fid = dict(p.meta["fidelity"], target_floor=cap, err_quadrature=other)
    fid.pop("err_max", None)
    fid.pop("err_truncation", None)
    return type(p)(p.arrays, dict(p.meta, fidelity=fid), source=p.source)


def test_a_cap_spent_by_the_other_terms_refuses_by_that_cause_not_by_a_needed_frequency():
    """Where the other certified terms reach the cap the truncation budget is zero: a waveform with a gradient is refused
    for that, naming the cap, its source and the terms that spend it, and no frequency the waveform needs; the same pack
    certified to a larger cap admits it."""
    p = pack_of("cylinder")
    spent = _capped(p, cap=3e-3)
    assert spent.truncation_budget["budget"] == 0.0
    with pytest.raises(ValueError) as e:
        spent.replay(pgse(0.1))
    msg = str(e.value)
    b = spent.truncation_budget
    assert f"cap {b['cap']:.2e} (from fidelity.target_floor)" in msg and f"{b['other_terms']:.2e} of other certified terms" in msg
    assert b["other_terms"] > b["cap"]
    assert "exhaust the cap" in msg and "zero allowance for truncation error" in msg
    assert "larger cap or smaller numerical terms" in msg
    assert "Hz" not in msg and "needs" not in msg and "temporal band" not in msg
    _capped(p, cap=3e-2).replay(pgse(0.1))


def test_a_zero_gradient_waveform_is_admitted_on_a_pack_with_a_spent_cap():
    spent = _capped(pack_of("cylinder"), cap=3e-3)
    assert spent.truncation_budget["budget"] == 0.0
    spent.replay(dr.pgse([[1.0, 0.0, 0.0]], TD["delta"], TD["Delta"], gradient_strengths=0.0, slew_rate=200.0))


def test_a_waveform_beyond_the_band_of_a_pack_with_a_budget_keeps_its_message():
    p = _capped(pack_of("cylinder", G_max=10.0), cap=3e-2)
    b = p.truncation_budget
    assert b["budget"] > 0.0
    seq = dr.ogse([[1.0, 0.0, 0.0]], 1200.0, 0.0025, shape="cosine", gradient_strengths=4.0, slew_rate=np.inf)
    with pytest.raises(ValueError, match=r"^the waveform's gradient reaches beyond this pack's temporal band: it needs "
                                         r"\S+ Hz \(\d+ bands over the .* ms it spans\) and the pack resolves 800 Hz .* which would drop up to .* against "
                                         + re.escape(f"the {b['budget']:.2e} it may cost (its cap {b['cap']:.2e} from fidelity.target_floor less "
                                                      f"{b['other_terms']:.2e} of other certified terms). Walk a pack whose envelope "
                                                      "holds this band")):
        p.replay(seq)


def test_the_columnar_plan_names_a_spent_cap_the_same_way():
    from columnar_scenarios import seqs
    seq = seqs(dr)["pgse_short"]
    lay = open_columnar(str(DATA / "columnar_quant"))
    fid = lay.meta["fidelity"]
    fid.update(target_floor=3e-3, err_quadrature=3e-3)
    fid.pop("err_max", None)
    fid.pop("err_truncation", None)
    assert lay.truncation_budget["budget"] == 0.0
    with pytest.raises(ValueError, match=r"they exhaust the cap and leave zero allowance for truncation error"):
        lay.plan(seq)
