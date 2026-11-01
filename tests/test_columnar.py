"""The columnar reader against dmipy-sim's on the same layouts (``tools/make_columnar_fixtures.py``): the plan names
the same bands and bytes, the image is sim's to float64 rounding, a voxel subset reads only its rows, a scanner
class plans its band, and the remote route reads the layout by byte range and refuses a server that ignores it."""
import json
from pathlib import Path

import numpy as np
import pytest

import dmipy_replay as dr
from dmipy_replay.replay import hub
from dmipy_replay.replay.columnar import open_columnar
from columnar_scenarios import seqs, tissue_of

DATA = Path(__file__).parent / "data"
REF = np.load(DATA / "columnar_reference.npz")
INFO = json.loads((DATA / "columnar_reference.json").read_text())
LAYOUTS = sorted(INFO["layouts"])
SEQS = seqs(dr)


@pytest.mark.parametrize("name", LAYOUTS)
def test_the_plan_is_sims(name):
    lay = open_columnar(str(DATA / name))
    assert lay.band_groups == INFO["layouts"][name]["band_groups"]
    for key, seq in SEQS.items():
        for tol in (0.25, 1e-12):
            p = lay.plan(seq, tol=tol)
            assert {k: p[k] for k in ("K", "modes", "rows", "bytes")} == INFO["layouts"][name][f"{key}/plan/{tol:g}"]


def test_a_low_band_acquisition_reads_fewer_bytes():
    lay = open_columnar(str(DATA / "columnar_quant"))
    lean, whole = lay.plan(SEQS["pgse_short"]), lay.plan(SEQS["pgse_short"], tol=1e-12)
    assert lean["K"] < whole["K"] == lay.K and lean["bytes"] < whole["bytes"]


@pytest.mark.parametrize("name", LAYOUTS)
@pytest.mark.parametrize("key", sorted(SEQS))
def test_the_image_is_sims(name, key):
    lay = open_columnar(str(DATA / name))
    S, floor, plan = lay.image(SEQS[key], tol=1e-12)
    np.testing.assert_allclose(S, REF[f"{name}/{key}/S"], rtol=1e-12, atol=1e-15)
    np.testing.assert_array_equal(floor, REF[f"{name}/{key}/floor"])
    S2, _, _ = lay.image(SEQS[key], settings=[(None, None), (tissue_of(dr, lay.meta), 3.0)], tol=1e-12)
    np.testing.assert_allclose(S2, REF[f"{name}/{key}/settings"], rtol=1e-12, atol=1e-15)


def test_a_voxel_subset_reads_only_its_rows():
    lay = open_columnar(str(DATA / "columnar_quant"))
    seq = SEQS["pgse_short"]
    whole, _, _ = lay.image(seq, tol=1e-12)
    part, _, plan = lay.image(seq, voxels=[(0, 0, 0)], tol=1e-12)
    assert plan["rows"] == sum(e - s for s, e in lay.rows_of([(0, 0, 0)])) < lay.n_rows
    np.testing.assert_allclose(part[0, 0, 0], whole[0, 0, 0], rtol=1e-12)
    assert np.isnan(part[1, 0, 0]).all()


def test_an_envelope_plans_its_band():
    lay = open_columnar(str(DATA / "columnar_quant"))
    for scanner in ("prisma", "connectome_2"):
        env = dr.envelope_of(scanner)
        need = env.mode_count(lay.T_window)
        if need > lay.K:
            with pytest.raises(ValueError, match="narrower band"):
                lay.plan_for_envelope(env)
        else:
            p = lay.plan_for_envelope(env)
            assert p["K"] == min(g for g in lay.band_groups if g >= need)
    with pytest.raises(TypeError, match="takes an Envelope"):
        lay.plan_for_envelope("prisma")


@pytest.fixture
def remote(server, monkeypatch):
    """``hf://fixture/local/<layout>`` served by byte range from ``tests/data``."""
    monkeypatch.setattr(hub, "revision_of", lambda repo, revision="main": "pinned")
    monkeypatch.setattr(hub, "_resolve_url", lambda repo, path, revision: f"{server}/{path}")
    return "hf://fixture/local/columnar_quant"


def test_the_remote_layout_reads_by_range(remote):
    lay = open_columnar(remote)
    seq = SEQS["pgse_short"]
    S, _, plan = lay.image(seq, tol=1e-12)
    np.testing.assert_allclose(S, REF["columnar_quant/pgse_short/S"], rtol=1e-12, atol=1e-15)
    assert plan["bytes_read"] > 0 and lay.src.revision == "pinned"


def test_a_server_that_ignores_the_range_is_refused(monkeypatch):
    """A 200 with the whole file to a range request is retried and then refused, never sliced."""
    import functools
    import http.server
    import threading
    from conftest import _Range
    from dmipy_replay.replay import columnar

    class Whole(_Range):
        def do_GET(self):
            if self.headers.get("Range") and not self.path.endswith(".json"):
                del self.headers["Range"]
            super().do_GET()
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(Whole, directory=str(DATA)))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        monkeypatch.setattr(hub, "revision_of", lambda repo, revision="main": "pinned")
        monkeypatch.setattr(hub, "_resolve_url", lambda repo, path, revision: f"{url}/{path}")
        monkeypatch.setattr(columnar, "RETRIES", 2)
        monkeypatch.setattr(columnar.time, "sleep", lambda s: None)
        lay = open_columnar("hf://fixture/local/columnar_quant")
        with pytest.raises(IOError, match="status 200 to a range request"):
            lay.view(K=4)
    finally:
        srv.shutdown()
