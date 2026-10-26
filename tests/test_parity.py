"""Every scenario of ``scenarios.py`` replayed here against dmipy-sim's own replay of the same pack
(``tools/make_fixtures.py``): the code is the same code, so the numbers are the same to rounding."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import dmipy_replay as dr
from scenarios import SCENARIOS

DATA = Path(__file__).parent / "data"
REF = np.load(DATA / "reference.npz")
INDEX = json.loads((DATA / "reference.json").read_text())
NS = SimpleNamespace(pgse=dr.pgse, pgste=dr.pgste, ogse=dr.ogse, gre=dr.gre, cpmg=dr.cpmg, Tissue=dr.Tissue)
_PACKS = {}


def _pack(key):
    if key not in _PACKS:
        _PACKS[key] = dr.read_rpk(DATA / f"{key}.rpk")
    return _PACKS[key]


@pytest.mark.parametrize("name,key,fn", SCENARIOS, ids=[s[0] for s in SCENARIOS])
def test_matches_sim(name, key, fn):
    if name in INDEX["failed"]:
        with pytest.raises(Exception) as e:
            fn(NS, _pack(key))
        assert type(e.value).__name__ == INDEX["failed"][name].split(":")[0]
        return
    got = np.asarray(fn(NS, _pack(key)))
    want = REF[name]
    assert got.shape == want.shape
    np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-12)
