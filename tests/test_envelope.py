"""A replay is held to the scanner class its pack states (RPK.md 11): the per-axis gradient, the sampled slew, the field
channel's refocusing depth and the field strength, each refused by name, on every route that takes a waveform, and the
length-scaled view is held to its moved limits by the same check."""
from pathlib import Path

import numpy as np
import pytest

import dmipy_replay as dr
from dmipy_replay.replay.columnar import open_columnar

DATA = Path(__file__).parent / "data"
CLASS = dict(bound="scanner_class", scanner="test_class", G_max=0.3, slew_max=200.0)
TD = dict(delta=0.006, Delta=0.02)


def pack_of(name, **acq):
    """The fixture pack ``name`` with its scanner class replaced by ``CLASS`` updated with ``acq`` (a ``None`` drops a key)."""
    p = dr.read_rpk(DATA / f"{name}.rpk")
    meta = {k: (dict(v) if isinstance(v, dict) else v) for k, v in p.meta.items()}
    meta["replay_envelope"] = {k: (dict(v) if isinstance(v, dict) else v) for k, v in p.meta["replay_envelope"].items()}
    a = {k: v for k, v in {**CLASS, **acq}.items() if v is not None}
    meta["replay_envelope"]["acquisition"] = {**{k: v for k, v in p.meta["replay_envelope"]["acquisition"].items()
                                                  if k not in CLASS}, **a}
    return type(p)(p.arrays, meta, source=p.source)


def pgse(g=0.1, slew_rate=200.0):
    return dr.pgse([[1.0, 0.0, 0.0]], TD["delta"], TD["Delta"], gradient_strengths=g, slew_rate=slew_rate)


def test_a_waveform_inside_the_class_passes():
    p = pack_of("cylinder")
    assert np.all(np.isfinite(p.replay(pgse(0.1))))
    assert np.all(np.isfinite(p.replay(pgse(0.3))))                      # at G_max
    assert np.all(np.isfinite(pack_of("cylinder", slew_max=200.0).replay(pgse(0.1, slew_rate=200.0))))   # at slew_max


def test_a_gradient_beyond_g_max_is_refused_by_name():
    with pytest.raises(ValueError, match=r"plays 0\.35\d* T/m on one axis, beyond the G_max = 0\.3 T/m"):
        pack_of("cylinder").replay(pgse(0.35))


def test_a_slew_beyond_slew_max_is_refused_by_name():
    with pytest.raises(ValueError, match=r"slews \S+ T/m/s on one axis \(sampled\), beyond the slew_max = 200 T/m/s"):
        pack_of("cylinder").replay(pgse(0.1, slew_rate=400.0))
    with pytest.raises(ValueError, match="slew_max"):                    # a square lobe switches within one sample
        pack_of("cylinder").replay(pgse(0.1, slew_rate=np.inf))


def test_refocusing_pulses_beyond_the_field_channels_depth_are_refused():
    chi = dr.Tissue(chi_iso=-1e-7)
    p = pack_of("field", max_refocusing_pulses=2)
    assert p.replay_envelope["acquisition"]["max_refocusing_pulses"] == 2
    ok, over = dr.cpmg(2, 0.004), dr.cpmg(3, 0.004)
    with pytest.raises(ValueError, match=r"plays 3 refocusing pulses with a field term, beyond the 2"):
        p.replay(over, tissue=chi, scanner=3.0)
    with pytest.raises(ValueError, match="refocusing pulses"):
        p.plan(over, tissue=chi, scanner=3.0)
    p.plan(ok, tissue=chi, scanner=3.0)                                   # at the depth
    p.plan(over, tissue=dr.Tissue(), scanner=3.0)                         # no field term: the field channel is not read


def test_a_field_strength_is_certified_over_the_range_up_to_the_largest_b0_list_entry():
    p = pack_of("field")
    assert p.replay_envelope["acquisition"]["B0_list"] == [3.0, 7.0]
    chi = dr.Tissue(chi_iso=-1e-7)
    p.replay(dr.gre(0.006), tissue=chi, scanner=5.0)                      # between the listed points
    p.replay(dr.gre(0.006), tissue=chi, scanner=1.5)
    with pytest.raises(ValueError, match=r"B0 = 11\.7 T is beyond the 7 T"):
        p.replay(dr.gre(0.006), tissue=chi, scanner=11.7)


def test_a_susceptibility_beyond_the_certified_field_scale_is_refused():
    with pytest.raises(ValueError, match="certified field scale"):
        pack_of("field").replay(dr.gre(0.006), tissue=dr.Tissue(chi_iso=-1e-5), scanner=7.0)


def test_a_header_without_a_scanner_class_is_refused_by_name():
    p = pack_of("cylinder")
    meta = {**p.meta, "replay_envelope": {**p.meta["replay_envelope"],
                                          "acquisition": {k: v for k, v in p.meta["replay_envelope"]["acquisition"].items() if k != "bound"}}}
    with pytest.raises(ValueError, match="states no scanner-class envelope"):
        type(p)(p.arrays, meta).replay(pgse())
    with pytest.raises(ValueError, match=r"acquisition\.slew_max"):
        pack_of("cylinder", slew_max=None).replay(pgse())
    with pytest.raises(ValueError, match="battery of b values"):
        pack_of("cylinder", bound="b_max").replay(pgse())


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
    lay.plan(seq)                                                         # the fixture class is wide
    acq = lay.meta["replay_envelope"]["acquisition"]
    acq.update(G_max=1.0)
    with pytest.raises(ValueError, match=r"plays 6\.9\d* T/m on one axis, beyond the G_max = 1 T/m"):
        lay.plan(seq)
    acq.update(G_max=10.0, slew_max=1e5)
    with pytest.raises(ValueError, match=r"slew_max = 1e\+05 T/m/s"):
        lay.plan(seq)


def test_the_length_scaled_view_is_held_to_its_moved_limits():
    p = pack_of("cylinder")
    smaller, larger = p.at_length_scale(0.5), p.at_length_scale(2.0)
    assert smaller.replay_envelope["acquisition"]["G_max"] == pytest.approx(0.6)
    assert smaller.replay_envelope["acquisition"]["slew_max"] == pytest.approx(400.0)
    smaller.replay(pgse(0.5, slew_rate=400.0))                            # beyond the pack's, inside the view's
    with pytest.raises(ValueError, match="G_max"):
        p.replay(pgse(0.5))
    with pytest.raises(ValueError, match=r"G_max = 0\.15 T/m.*length-scaled view, s = 2"):
        larger.replay(pgse(0.2))                                          # inside the pack's, beyond the view's
    with pytest.raises(ValueError, match=r"slew_max = 100 T/m/s.*length-scaled view"):
        larger.replay(pgse(0.1, slew_rate=200.0))
    larger.replay(pgse(0.1, slew_rate=100.0))
