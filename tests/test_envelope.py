"""A replay is held to the :class:`~dmipy_replay.Envelope` its pack states (RPK.md 11): the waveform's band and its
per-axis amplitude, the field channel's refocusing depth and the field strength, each refused by name, on every route
that takes a waveform; the views are held to their moved envelopes by the same check. A header that states a scanner
class is read as an envelope by the stated rule."""
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
