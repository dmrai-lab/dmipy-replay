"""The ``bitshuffle-zlib`` codec of a band range (:mod:`dmipy_replay.replay.codec`): the round trip, the decoded integers of
every range of the test packs against the fixed-width container's, the replay bit for bit, a walker prefix from the chunks
that hold it (locally and by byte ranges from a server), and the refusal of a codec the reader does not implement."""
import contextlib
import http.server
import json
import re
import threading
from pathlib import Path

import numpy as np
import pytest

import dmipy_replay as dr
from dmipy_replay.replay import codec, compression as cx, container
from dmipy_replay.replay.remote import RemotePack


DATA = Path(__file__).parent / "data"
PACKS = ("cylinder", "field", "permeable")


class _Ranges(http.server.BaseHTTPRequestHandler):
    """A GET that honours ``Range`` over one in-memory payload."""
    data = b""

    def do_GET(self):
        body = type(self).data
        start_s, _, end_s = self.headers.get("Range").split("=", 1)[1].partition("-")
        start, end = int(start_s), min(int(end_s), len(body) - 1)
        chunk = body[start:end + 1]
        self.send_response(206)
        self.send_header("Content-Range", f"bytes {start}-{end}/{len(body)}")
        self.send_header("Content-Length", str(len(chunk)))
        self.end_headers()
        self.wfile.write(chunk)

    def log_message(self, *a):
        pass


@contextlib.contextmanager
def _serve(data):
    httpd = http.server.HTTPServer(("127.0.0.1", 0), type("H", (_Ranges,), {"data": data}))
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_port}/pack.rpk", httpd
    finally:
        httpd.shutdown()
        t.join()


def _pgse():
    return dr.pgse([[1, 0, 0], [0, 0, 1]], 0.006, 0.02, bvalues=[1e9, 3e9])


def _coded_file(tmp_path, name, chunk_walkers):
    arrays, hdr = container.read_safetensors(DATA / f"{name}.rpk")
    meta = json.loads(hdr["rpk"])
    carr, cmeta = codec.encode_ranges(dict(arrays), meta, codec=codec.CODEC, chunk_walkers=chunk_walkers)
    path = tmp_path / f"{name}.coded.rpk"
    container.write_safetensors(path, carr, {"rpk": json.dumps(cmeta)})
    return path


def _ranges(arrays):
    return {k: v for k, v in arrays.items() if re.match(r"^(pos_[xyz]|blt|susc_path_dct)_b\d+(u\d+)?$", k)}


def _integers(k, t, width):
    """The integers of a fixed-width range tensor ``t`` holding ``width`` bands (``(n_w, [n_ch,] width)``)."""
    m = re.match(r"^(?:pos_[xyz]|blt|susc_path_dct)_b\d+(?:u(\d+))?$", k)
    bits = int(m.group(1)) if m.group(1) else 8 * np.asarray(t).dtype.itemsize
    t = np.asarray(t)
    if k.startswith("susc_path_dct"):
        return np.swapaxes(np.asarray(cx._load_range(np.swapaxes(t, 1, 2), bits, width)), 1, 2)
    return np.asarray(cx._load_range(t, bits, width))


@pytest.mark.parametrize("bits", [3, 5, 8, 9, 12, 16])
@pytest.mark.parametrize("shape", [(1, 5), (7, 3, 4), (5000, 9), (4096, 2, 8)])
def test_chunks_round_trip(bits, shape):
    rng = np.random.default_rng(bits * 7 + shape[0])
    lim = min(2 ** (bits - 1) - 1, 2 ** (codec.lane_bits(bits) - 1) - 1)
    q = rng.integers(-lim, lim + 1, shape)
    q.flat[0] = -(2 ** (codec.lane_bits(bits) - 1)) if bits in (8, 16) else -lim
    stream, chunks = codec.encode_chunks(q, bits, 1024)
    assert chunks.dtype == np.int64 and chunks[0] == 0 and chunks[-1] == len(stream) and len(chunks) == -(-shape[0] // 1024) + 1
    got = codec.decode_chunks(stream, chunks, bits, 1024, shape[0], shape[1:])
    assert got.dtype == codec.lane_dtype(bits) and np.array_equal(got, q)


def test_the_bit_layout_is_the_bitshuffle_blocks():
    """Plane ``p`` of a block holds bit ``p`` of its 4096 elements, element ``8 j + i`` at bit ``i`` of byte ``j``."""
    u = np.zeros(codec.BLOCK, np.uint16)
    u[0], u[9], u[4095] = 0b1, 0b100000000000010, 0xFFFF
    planes = np.frombuffer(codec._transpose(u).tobytes(), np.uint8).reshape(16, 512)
    assert planes[0, 0] == 0b1 and planes[1, 1] == 0b10 and planes[14, 1] == 0b10
    assert planes[:, 511].tolist() == [0b10000000] * 16
    assert np.array_equal(codec._untranspose(planes.tobytes(), 16), u)


def test_zigzag_order():
    assert codec.zigzag(np.array([0, -1, 1, -2, 2]), 6).tolist() == [0, 1, 2, 3, 4]
    assert codec.unzigzag(np.array([0, 1, 2, 3, 4]), 6).tolist() == [0, -1, 1, -2, 2]
    assert codec.zigzag(np.array([-128]), 8).tolist() == [255] and codec.zigzag(np.array([-32768]), 16).tolist() == [65535]


def test_a_value_outside_the_lane_is_refused():
    with pytest.raises(ValueError, match="lane"):
        codec.encode_chunks(np.array([[200]]), 8, 4096)


@pytest.mark.parametrize("name", PACKS)
def test_every_range_decodes_to_the_fixed_width_integers(tmp_path, name):
    fixed = dr.read_rpk(DATA / f"{name}.rpk")
    coded = dr.read_rpk(_coded_file(tmp_path, name, 512))
    want = _ranges(fixed.arrays)
    assert want and set(_ranges(coded.arrays)) == set(want)
    assert not [k for k in coded.arrays if codec.is_stored_range(k)]
    for k, t in want.items():
        got = coded.arrays[k]
        assert isinstance(got, codec.CodecRange) and got.shape[0] == t.shape[0]
        assert np.array_equal(_integers(k, t, got.shape[-1]), np.asarray(got)), k
        assert np.array_equal(got.fixed_width(), np.asarray(t)), k          # re-packed, the container's own bytes


@pytest.mark.parametrize("name", PACKS)
def test_the_replay_is_bit_identical(tmp_path, name):
    fixed = dr.read_rpk(DATA / f"{name}.rpk")
    coded = dr.read_rpk(_coded_file(tmp_path, name, 512))
    kw = dict(tissue=dr.Tissue(chi_iso=-1e-7), scanner=7.0) if name == "field" else {}
    seq = dr.gre(0.006) if name == "field" else _pgse()
    assert np.array_equal(np.asarray(coded.replay(seq, **kw)), np.asarray(fixed.replay(seq, **kw)))
    assert coded.plan(seq, **kw).bytes_stored > 0 and coded.plan(seq, **kw).K == fixed.plan(seq, **kw).K
    a, b = cx.read_position_coeffs(fixed.arrays), cx.read_position_coeffs(coded.arrays)
    assert np.array_equal(a, b)


def test_a_walker_prefix_is_read_from_the_chunks_that_hold_it(tmp_path):
    fixed = dr.read_rpk(DATA / "cylinder.rpk")
    coded = dr.read_rpk(_coded_file(tmp_path, "cylinder", 256))
    from dmipy_replay.replay.compression import walker_rows
    rows = walker_rows(coded.arrays, 300, 900)
    want = walker_rows(fixed.arrays, 300, 900)
    for k, t in _ranges(want).items():
        assert np.array_equal(_integers(k, t, rows[k].shape[-1]), np.asarray(rows[k])), k
    r = next(v for k, v in coded.arrays.items() if isinstance(v, codec.CodecRange))
    p = r.prefix(300)
    assert np.array_equal(np.asarray(p), np.asarray(r)[:300]) and p.stored_nbytes == int(r.chunks[2])
    # only the chunks that cover walkers 300:900 inflate: a stream cut after chunk 3 serves them
    cut = codec.CodecRange(r.stream[:int(r.chunks[4])], r.chunks, bits=r.bits, chunk_walkers=256, n_walkers=r.n_total,
                           tail=r.tail, n_held=2000)
    assert np.array_equal(cut.rows(300, 900), np.asarray(r)[300:900])
    with pytest.raises(ValueError, match="ends at byte"):
        cut.rows(1100, 1200)


def test_a_remote_coded_pack_with_field_modes_replays_as_the_whole_file(tmp_path):
    path = _coded_file(tmp_path, "field", 64)
    whole = dr.read_rpk(DATA / "field.rpk")
    tissue, seq = dr.Tissue(chi_iso=-1e-7), dr.gre(0.006)
    with _serve(path.read_bytes()) as (url, handler):
        remote = RemotePack("hf://owner/coded/field.rpk", url=url, revision="rev0")
        np.testing.assert_array_equal(np.asarray(remote.replay(seq, tissue=tissue, scanner=7.0)),
                                      np.asarray(whole.replay(seq, tissue=tissue, scanner=7.0)))
        assert any(n.startswith("susc_path_dct") for n in remote._coded)


def test_a_remote_coded_pack_replays_bit_for_bit_and_reads_a_prefix_by_chunks(tmp_path):
    path = _coded_file(tmp_path, "cylinder", 256)
    fixed = dr.read_rpk(DATA / "cylinder.rpk")
    seq = _pgse()
    with _serve(path.read_bytes()) as (url, handler):
        remote = RemotePack("hf://owner/coded/cylinder.rpk", url=url, revision="rev0")
        assert remote.plan(seq).K == fixed.plan(seq).K
        S = remote.replay(seq)
        np.testing.assert_array_equal(np.asarray(S), np.asarray(fixed.replay(seq)))
        name = next(n for n in remote._coded if n.startswith("pos_x"))
        c = remote._coded[name]
        before = remote.bytes_fetched
        held_chunks = remote._held[c["chunks"]]
        r = remote._coded_range(name, 300)                   # walkers 0:300 -> chunks 0 and 1
        assert r.n_held == 300 and remote.bytes_fetched - before == int(held_chunks[2])
        assert np.array_equal(np.asarray(r), np.asarray(remote._held[name])[:300])


def test_a_codec_the_reader_does_not_implement_is_refused_by_name(tmp_path):
    arrays, hdr = container.read_safetensors(_coded_file(tmp_path, "field", 256))
    meta = json.loads(hdr["rpk"])
    meta["compression"]["container"][0]["codec"] = "zstd-bitshuffle"
    path = tmp_path / "unknown.rpk"
    container.write_safetensors(path, dict(arrays), {"rpk": json.dumps(meta)})
    with pytest.raises(ValueError, match="zstd-bitshuffle"):
        dr.read_rpk(path)
    with _serve(path.read_bytes()) as (url, _):
        with pytest.raises(ValueError, match="zstd-bitshuffle"):
            RemotePack("hf://owner/coded/field.rpk", url=url, revision="rev0")


def test_a_coded_tensor_the_metadata_does_not_declare_is_refused(tmp_path):
    arrays, hdr = container.read_safetensors(_coded_file(tmp_path, "field", 256))
    meta = json.loads(hdr["rpk"])
    for r in meta["compression"]["container"]:
        r.pop("codec"), r.pop("chunk_walkers")
    path = tmp_path / "undeclared.rpk"
    container.write_safetensors(path, dict(arrays), {"rpk": json.dumps(meta)})
    with pytest.raises(ValueError, match="not the bitshuffle-zlib range"):
        dr.read_rpk(path)


def test_chunk_walkers_must_be_a_power_of_two_and_belongs_to_a_codec():
    meta = {"compression": {"container": [{"bands": [0, 4], "bits": 6, "codec": "bitshuffle-zlib", "chunk_walkers": 1000}]}}
    with pytest.raises(ValueError, match="power of two"):
        codec.refuse_unknown(meta)
    meta = {"compression": {"container": [{"bands": [0, 4], "bits": 6, "chunk_walkers": 1024}]}}
    with pytest.raises(ValueError, match="without a codec"):
        codec.refuse_unknown(meta)
    with pytest.raises(ValueError, match="power of two"):
        codec.encode_chunks(np.zeros((4, 4), int), 6, 1000)


def test_an_embedded_coded_pack_carries_the_fixed_width_container(tmp_path):
    from dmipy_replay.replay.phantom import embed_pack
    coded = dr.read_rpk(_coded_file(tmp_path, "field", 256))
    fixed = dr.read_rpk(DATA / "field.rpk")
    tensors, subs = {}, [{}]
    embed_pack(subs, tensors, 0, coded)
    for k, t in fixed.arrays.items():
        assert np.array_equal(tensors[f"substrate0/{k}"], np.asarray(t)), k
    assert not any("codec" in r for r in subs[0]["pack_meta"]["compression"]["container"])
