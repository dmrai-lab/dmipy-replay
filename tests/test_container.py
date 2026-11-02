"""The safetensors container read and written with numpy alone, against the reference library where it is installed,
and the hub's windowed read against a local HTTP server."""
import json
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


def test_the_writer_is_byte_for_byte_the_reference_library(tmp_path):
    """Every dtype a pack stores, a 0-d array, an empty one and a big-endian one, with metadata and without: the
    file is the one ``safetensors.numpy.save_file`` writes for the contiguous little-endian arrays."""
    st = pytest.importorskip("safetensors.numpy")
    arrays = dict(ARRAYS, f=np.float16([1.5, -2.0]), g=np.arange(6, dtype=np.int64).reshape(2, 3), h=np.uint32([7]),
                  i=np.float64(3.25), j=np.arange(4, dtype=">i4"), k=np.int16([-3, 3]), l=np.uint64([2 ** 63]),
                  m=np.int32([1, 2]), n=np.uint8([9]))
    plain = {k: np.ascontiguousarray(v, np.asarray(v).dtype.newbyteorder("<")) for k, v in arrays.items()}
    for md in ({"rpk": json.dumps({"k": [1, "\u00e9"]}, ensure_ascii=False)}, None):
        ours, theirs = tmp_path / "ours.safetensors", tmp_path / "theirs.safetensors"
        write_safetensors(ours, arrays, md)
        st.save_file(plain, str(theirs), metadata=md)
        assert ours.read_bytes() == theirs.read_bytes()


def test_a_tensor_in_pieces_is_written_as_their_concatenation(tmp_path):
    pieces = [np.arange(6, dtype=np.float32).reshape(2, 3), np.arange(9, dtype=np.float32).reshape(3, 3)]
    a, b = tmp_path / "pieces.safetensors", tmp_path / "whole.safetensors"
    write_safetensors(a, {"x": pieces, "y": np.int8([1])}, {"rpk": "{}"})
    write_safetensors(b, {"x": np.concatenate(pieces), "y": np.int8([1])}, {"rpk": "{}"})
    assert a.read_bytes() == b.read_bytes()
    with pytest.raises(ValueError, match="differ in dtype or row shape"):
        write_safetensors(a, {"x": [pieces[0], pieces[1].astype(np.float64)]})


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
