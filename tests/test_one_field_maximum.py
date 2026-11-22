"""An envelope states one maximum field strength (``B0_max``); a header's ``B0_list`` is read at its largest entry."""
import json
from pathlib import Path

import numpy as np
import pytest

import dmipy_replay as dr
from dmipy_replay.replay.envelope import Envelope
from dmipy_replay.replay.replay import field_envelope_of

DATA = Path(__file__).parent / "data"
FIXTURES = ("cylinder", "field", "permeable", "plane_linear_field", "slab_exchange", "windows")


def _env(**kw):
    return Envelope(**{**dict(name="x", band_hz=500.0, G_max=0.3, B0_max=7.0, chi_max=1e-7, rho2_max=None,
                              max_refocusing_pulses=1), **kw})


def _meta(fixture, **acq):
    meta = json.loads(json.dumps(dr.read_rpk(DATA / f"{fixture}.rpk").meta))
    a = {**meta["replay_envelope"]["acquisition"], **acq}
    meta["replay_envelope"]["acquisition"] = {k: v for k, v in a.items() if v is not None}
    return meta


def test_a1_an_envelope_holds_one_maximum_and_no_list():
    e = _env()
    assert e.B0_max == 7.0 and not hasattr(e, "B0_list")
    d = e.to_dict()
    assert d["B0_max"] == 7.0 and "B0_list" not in d
    assert Envelope.from_dict(d) == e


def test_a2_a_constructor_refuses_anything_but_one_finite_non_negative_number():
    for bad in ([3.0, 7.0], (7.0,), True):
        with pytest.raises(ValueError, match="one maximum field strength"):
            _env(B0_max=bad)
    for bad in ("7", -1.0, np.inf):
        with pytest.raises(ValueError):
            _env(B0_max=bad)
    with pytest.raises(TypeError):
        Envelope(name="x", band_hz=500.0, G_max=0.3, B0_list=(7.0,), chi_max=None, rho2_max=None,
                 max_refocusing_pulses=None)
    with pytest.raises(TypeError):
        dr.envelope_of("prisma", B0_list=[3.0])
    assert _env(B0_max=None).B0_max is None


def test_a3_a_header_is_read_at_its_largest_entry():
    assert Envelope.from_header(dr.read_rpk(DATA / "field.rpk").meta).B0_max == 7.0
    assert Envelope.from_header(dr.read_rpk(DATA / "plane_linear_field.rpk").meta).B0_max == 3.0
    assert Envelope.from_header(_meta("field", B0_list=[7.0, 3.0, 5.0])).B0_max == 7.0
    assert Envelope.from_header(_meta("field", B0_list=[])).B0_max is None
    scanner = _meta("field", bound="scanner_class", scanner="test_class", slew_max=200.0, name=None, band_hz=None,
                    B0_list=None)
    scanner.setdefault("compression", {})["temporal_bandwidth_hz"] = 500.0
    assert Envelope.from_header(scanner).B0_max is None
    with pytest.raises(ValueError, match=r"acquisition\.B0_list"):
        Envelope.from_header(_meta("field", B0_list=None))


def test_a4_a_header_states_one_entry_under_the_same_five_keys():
    assert _env().header_acquisition()["B0_list"] == [7.0]
    assert _env(B0_max=None).header_acquisition()["B0_list"] == []
    assert set(_env().header_acquisition()) == {"bound", "name", "band_hz", "G_max", "B0_list"}


def test_a5_the_field_envelope_of_every_fixture_is_its_largest_strength_times_chi():
    for name in FIXTURES:
        meta = dr.read_rpk(DATA / f"{name}.rpk").meta
        got = field_envelope_of(meta)
        chi = (meta.get("replay_envelope") or {}).get("tissue", {}).get("chi_max")
        if chi is None:
            assert got is None, name
        else:
            B0 = max(meta["replay_envelope"]["acquisition"]["B0_list"])
            assert got == dict(B0_max=B0, chi_max=chi, B0_chi_max=B0 * chi), name
    f = field_envelope_of(dr.read_rpk(DATA / "field.rpk").meta)
    assert f["B0_max"] == 7.0 and f["B0_chi_max"] == pytest.approx(1.4e-6)
    assert field_envelope_of(_meta("field", B0_list=[])) is None


def test_a6_a_stored_dictionary_with_a_list_is_read_at_its_largest_entry():
    d = _env().to_dict()
    listed = {k: v for k, v in d.items() if k != "B0_max"}
    assert Envelope.from_dict({**listed, "B0_list": [3.0, 7.0]}).B0_max == 7.0
    assert Envelope.from_dict({**listed, "B0_list": []}).B0_max is None
    with pytest.raises(ValueError, match="B0_list"):
        Envelope.from_dict({**d, "B0_list": [3.0, 7.0]})
    with pytest.raises(ValueError, match="unknown"):
        Envelope.from_dict({**d, "B0_extra": 1})
