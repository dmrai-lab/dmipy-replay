"""A pack on the Hub read by plan (:class:`~dmipy_replay.replay.remote.RemotePack`), driven past a local HTTP server
that honours byte ranges: the plan comes from the header alone, only the tensors it names move, and the replay is bit
for bit the replay of the whole file read at the same depth."""
import contextlib
import http.server
import json
import re
import threading
from pathlib import Path

import numpy as np
import pytest

import dmipy_replay as dr
from dmipy_replay.replay.remote import RemotePack

DATA = Path(__file__).parent / "data"


class _Ranges(http.server.BaseHTTPRequestHandler):
    """A GET that honours ``Range`` over one in-memory payload, recording every range asked (``type(self).asked``)."""
    data = b""
    asked = []

    def do_GET(self):
        body = type(self).data
        start_s, _, end_s = self.headers.get("Range").split("=", 1)[1].partition("-")
        start, end = int(start_s), min(int(end_s), len(body) - 1)
        type(self).asked.append((start, end))
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
    handler = type("H", (_Ranges,), {"data": data, "asked": []})
    httpd = http.server.HTTPServer(("127.0.0.1", 0), handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_port}/pack.rpk", handler
    finally:
        httpd.shutdown()
        t.join()


def _spans(blob):
    n = int.from_bytes(blob[:8], "little")
    head = json.loads(blob[8:8 + n])
    head.pop("__metadata__", None)
    return {k: (8 + n + v["data_offsets"][0], 8 + n + v["data_offsets"][1] - 1) for k, v in head.items()}


def _pgse():
    return dr.pgse([[1, 0, 0], [0, 0, 1]], 0.006, 0.02, bvalues=[1e9, 3e9])


def test_a_remote_pack_plans_from_its_header_and_fetches_only_the_planned_ranges():
    data = (DATA / "cylinder.rpk").read_bytes()
    whole = dr.read_rpk(DATA / "cylinder.rpk")
    seq = _pgse()
    with _serve(data) as (url, handler):
        remote = RemotePack("hf://owner/plan/cylinder.rpk", url=url, revision="rev0")
        header = remote.bytes_fetched
        assert header < 0.05 * len(data)
        plan = remote.plan(seq)
        assert plan == whole.plan(seq) and remote.bytes_fetched == header         # no tensor moved to plan
        S = remote.replay(seq)
        np.testing.assert_array_equal(np.asarray(S), np.asarray(whole.replay(seq)))
        assert S.plan == plan
        spans = _spans(data)
        starts = [int(r["bands"][0]) for r in whole.meta["compression"]["container"]]

        def ranges_from(K):
            """The byte spans of the position band-range tensors whose first band is ``K`` or above."""
            return [spans[k] for k in spans if re.fullmatch(r"pos_[xyz]_b\d+(u\d+)?", k)
                    and starts[int(re.match(r"pos_[xyz]_b(\d+)", k).group(1))] >= K]

        def asked_any(asked, spans_):
            return any(s <= b and a <= e for a, b in asked for s, e in spans_)

        beyond = ranges_from(plan.K)
        assert beyond and not asked_any(handler.asked, beyond)                     # the bands beyond the plan stay put
        assert not asked_any(handler.asked, [v for k, v in spans.items() if k.startswith("blt_")])   # no contact asked for
        assert 0 < remote.bytes_fetched < 0.6 * len(data)
        tissue = dr.Tissue(rho2=1e-6)
        before = remote.bytes_fetched
        S2 = remote.replay(seq, tissue=tissue)                                     # the contact tensors are added
        np.testing.assert_array_equal(np.asarray(S2), np.asarray(whole.replay(seq, tissue=tissue)))
        assert remote.bytes_fetched > before
        again = remote.bytes_fetched
        remote.replay(seq, tissue=tissue)
        assert remote.bytes_fetched == again                                       # held: a repeat moves nothing
        ogse = dr.ogse([[1, 0, 0]], 250.0, 0.008, bvalues=[1e8], slew_rate=np.inf)   # plans more bands than the PGSE
        K_ogse = remote.plan(ogse).K
        assert K_ogse > plan.K
        held, before = list(handler.asked), remote.bytes_fetched
        np.testing.assert_array_equal(np.asarray(remote.replay(ogse)), np.asarray(whole.replay(ogse)))
        assert remote.bytes_fetched > before                                       # a higher band adds its ranges ...
        new = handler.asked[len(held):]
        assert not any(a <= d and c <= b for a, b in held for c, d in new)         # ... never one already held ...
        assert ranges_from(K_ogse) and not asked_any(handler.asked, ranges_from(K_ogse))   # ... nor one beyond its plan


def test_a_remote_pack_with_field_modes_replays_as_the_whole_file():
    data = (DATA / "field.rpk").read_bytes()
    whole = dr.read_rpk(DATA / "field.rpk")
    tissue = dr.Tissue(chi_iso=-1e-7)
    seq = dr.gre(0.006)
    with _serve(data) as (url, _):
        remote = RemotePack("hf://owner/plan/field.rpk", url=url, revision="rev0")
        assert remote.plan(seq, tissue=tissue, scanner=7.0) == whole.plan(seq, tissue=tissue, scanner=7.0)
        np.testing.assert_array_equal(np.asarray(remote.replay(seq, tissue=tissue, scanner=7.0)),
                                      np.asarray(whole.replay(seq, tissue=tissue, scanner=7.0)))
        np.testing.assert_array_equal(np.asarray(remote.replay(seq)), np.asarray(whole.replay(seq)))


def test_a_hub_prefix_without_rpk_is_not_a_remote_pack():
    with pytest.raises(ValueError, match="a pack on the Hub is a .rpk file"):
        RemotePack("hf://owner/name/prefix", url="http://unused")


def test_a_hub_rpk_opens_as_a_remote_pack(monkeypatch):
    """``ReplayPack.open`` reads an ``hf://`` path ending in ``.rpk`` by range (any other uri is the columnar layout)."""
    import dmipy_replay.replay.remote as remote
    seen = []
    monkeypatch.setattr(remote, "RemotePack", lambda uri, **kw: seen.append(uri) or "remote")
    assert dr.ReplayPack.open("hf://owner/name/packs/x.rpk") == "remote" and seen == ["hf://owner/name/packs/x.rpk"]
