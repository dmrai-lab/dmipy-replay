"""The safetensors container read and written with numpy alone, against the reference library where it is installed,
and the hub's windowed read against a local HTTP server."""
import functools
import http.server
import json
import threading
from pathlib import Path

import numpy as np
import pytest

from dmipy_replay.replay.container import read_safetensors, write_safetensors
from dmipy_replay.replay import hub

DATA = Path(__file__).parent / "data"
ARRAYS = {"a": np.arange(12, dtype=np.float32).reshape(3, 4), "b": np.array([1, -2, 3], np.int8),
          "c": np.zeros((0, 3), np.float64), "d": np.array([True, False]), "e": np.arange(5, dtype=np.uint16)}


def test_round_trip(tmp_path):
    p = tmp_path / "x.safetensors"
    write_safetensors(p, ARRAYS, {"rpk": json.dumps({"k": 1})})
    got, meta = read_safetensors(p)
    assert json.loads(meta["rpk"]) == {"k": 1}
    for k, v in ARRAYS.items():
        assert got[k].dtype == v.dtype and got[k].shape == v.shape
        np.testing.assert_array_equal(got[k], v)


def test_against_the_reference_library(tmp_path):
    st = pytest.importorskip("safetensors.numpy")
    from safetensors import safe_open
    p = tmp_path / "ours.safetensors"
    write_safetensors(p, ARRAYS, {"rpk": "{}"})
    with safe_open(str(p), framework="numpy") as f:
        assert f.metadata() == {"rpk": "{}"}
        for k, v in ARRAYS.items():
            np.testing.assert_array_equal(f.get_tensor(k), v)
    q = tmp_path / "theirs.safetensors"
    st.save_file(ARRAYS, str(q), metadata={"rpk": "{}"})
    got, meta = read_safetensors(q)
    assert meta == {"rpk": "{}"}
    for k, v in ARRAYS.items():
        np.testing.assert_array_equal(got[k], v)


class _Range(http.server.SimpleHTTPRequestHandler):
    """A static file server that honours one byte range per request (what the Hub's CDN does)."""

    def do_GET(self):
        path = self.translate_path(self.path)
        data = Path(path).read_bytes()
        rng = self.headers.get("Range")
        if rng:
            a, b = (int(x) for x in rng.split("=")[1].split("-"))
            body, code = data[a:b + 1], 206
        else:
            body, code = data, 200
        self.send_response(code)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


@pytest.fixture
def server():
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(_Range, directory=str(DATA)))
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def test_windowed_read(server, tmp_path, monkeypatch):
    """The first window of a two-window pack by byte range: the shared tensors and window 0 only, and the replay of
    an acquisition inside it equal to the whole pack's."""
    import dmipy_replay as dr
    monkeypatch.setenv("DMIPY_REPLAY_CACHE", str(tmp_path))
    whole = dr.read_rpk(DATA / "windows.rpk")
    assert whole.n_windows == 2
    local = hub.fetch_windows("hf://fixture/local/windows.rpk", range(1), url=f"{server}/windows.rpk", revision="test")
    part = dr.read_rpk(local)
    assert part.windows_present == 1
    assert not any(k.startswith("s1/") for k in part.arrays)
    seq = dr.pgse([[1, 0, 0], [0, 1, 0]], 0.006, 0.02, bvalues=[1e9, 1e9])
    np.testing.assert_allclose(part.replay(seq), whole.replay(seq), rtol=1e-12)
    late = dr.pgse([[1, 0, 0]], 0.01, 0.12, bvalues=[1e9])
    with pytest.raises(ValueError, match="windows=range"):
        part.replay(late)
