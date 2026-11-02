"""Stamp a pack's header with its physical envelope: a header that states a scanner class (``replay_envelope.acquisition``
``bound = "scanner_class"`` with ``scanner``, ``G_max``, ``slew_max``) gains the :class:`~dmipy_replay.Envelope` the
reader derives from it (:data:`~dmipy_replay.replay.envelope.CLASS_HEADER_RULE`): ``bound = "envelope"``, ``name``,
``band_hz``, ``G_max``, ``B0_list``. Only the header changes; every tensor's bytes are copied as they are (a stamp, never
a re-encode), and ``provenance.envelope_stamp`` records the rule and the class fields it replaced. A header that already
states an envelope is left as it is; one that states neither (a battery of b values) is refused, since it states no
guarantee to stamp.

Dry run by default: nothing is written, and each file's old and new acquisition, header sizes and whether the tensor
data has to move is printed. With ``--out DIR`` the stamped files are written there (never over their source).

    python tools/stamp_envelope.py tests/data/cylinder.rpk tests/data/columnar_quant/manifest.json
    python tools/stamp_envelope.py tests/data/*.rpk --out /tmp/stamped
    python tools/stamp_envelope.py --hub SubstrateCommons/canonical-pores           # read-only: byte-range header reads

``--hub`` reads only: the dataset's manifest, and each pack's header by HTTP range (``--limit`` packs, all by default;
``--columnar PREFIX`` for a columnar layout's ``PREFIX/manifest.json``). It reports what a Hub pull request stamping the
dataset would change; it never writes to the Hub.
"""
import argparse
import copy
import json
import os
import struct
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dmipy_replay.replay.envelope import (ENVELOPE_BOUND, CLASS_HEADER_BOUND, CLASS_HEADER_RULE,  # noqa: E402
                                          Envelope)

#: the scanner-class fields a stamp replaces with the envelope's
CLASS_FIELDS = ("bound", "scanner", "G_max", "slew_max")


def stamp_meta(meta):
    """``(meta', change)``: the pack or layout header ``meta`` with its envelope stated, and what changed (None when it
    already states one). ``meta`` is not modified."""
    acq = ((meta.get("replay_envelope") or {}).get("acquisition") or {})
    bound = acq.get("bound")
    if bound == ENVELOPE_BOUND:
        Envelope.from_header(meta)                                   # a stated envelope must read
        return meta, None
    if bound != CLASS_HEADER_BOUND:
        raise ValueError(f"{meta.get('id')!r} states no scanner class (bound = {bound!r}): a battery of b values is no "
                         "guarantee to stamp; the pack is walked again with an envelope")
    env = Envelope.from_header(meta)
    out = copy.deepcopy(meta)
    old = out["replay_envelope"]["acquisition"]
    new = {k: v for k, v in old.items() if k not in CLASS_FIELDS}
    new.update(env.header_acquisition())
    out["replay_envelope"]["acquisition"] = new
    out.setdefault("provenance", {})["envelope_stamp"] = dict(rule=CLASS_HEADER_RULE,
                                                              replaced={k: old[k] for k in CLASS_FIELDS if k in old})
    assert Envelope.from_header(out) == env
    return out, dict(id=meta.get("id"), before={k: old.get(k) for k in CLASS_FIELDS if k in old},
                     after=env.header_acquisition())


def _meta_key(md):
    for key in ("rpk", "json"):
        if key in md:
            return key
    raise ValueError(f"no pack metadata in the header (keys {sorted(md)})")


def stamp_header(blob):
    """``(header', change, old_len, new_len)`` for the safetensors header bytes ``blob`` (its JSON, without the 8-byte
    length): the tensor entries as they are, the metadata stamped (:func:`stamp_meta`), padded with spaces to a multiple
    of 8 bytes. ``header'`` is None when nothing changes."""
    header = json.loads(blob)
    md = header.get("__metadata__") or {}
    key = _meta_key(md)
    meta, change = stamp_meta(json.loads(md[key]))
    if change is None:
        return None, None, len(blob), len(blob)
    header["__metadata__"] = dict(md, **{key: json.dumps(meta)})
    hb = json.dumps(header, separators=(",", ":")).encode()
    if len(hb) <= len(blob):                                         # fits where the old one was: pad to its length
        hb += b" " * (len(blob) - len(hb))
    else:
        hb += b" " * (-len(hb) % 8)
    return hb, change, len(blob), len(hb)


def report_line(where, change, old_len, new_len, size=None):
    moves = new_len != old_len
    d = dict(file=where, id=change["id"] if change else None, stamped=change is not None,
             header_bytes=[old_len, new_len], data_moves=moves,
             rewrite_bytes=(None if not change else (size if moves else 8 + new_len)), file_bytes=size)
    if change:
        d.update(before=change["before"], after=change["after"])
    return d


def stamp_file(src, out_dir=None):
    """Stamp one local ``.rpk`` (or a columnar ``manifest.json``); written to ``out_dir`` when given. A report row."""
    src = Path(src)
    if src.name == "manifest.json":
        manifest = json.loads(src.read_text())
        meta, change = stamp_meta(manifest["meta"])
        text = json.dumps(dict(manifest, meta=meta))
        if out_dir is not None and change is not None:
            dst = Path(out_dir) / src.parent.name / src.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(text)
        n0 = src.stat().st_size
        n1 = len(text.encode()) if change else n0
        return report_line(str(src), change, n0, n1, n1)
    size = src.stat().st_size
    with open(src, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        blob = f.read(n)
        hb, change, n0, n1 = stamp_header(blob)
        if out_dir is not None and hb is not None:
            dst = Path(out_dir) / src.name
            if dst.resolve() == src.resolve():
                raise ValueError("--out is never the source's own directory: a stamp writes a new file")
            dst.parent.mkdir(parents=True, exist_ok=True)
            with open(dst, "wb") as g:
                g.write(struct.pack("<Q", len(hb)))
                g.write(hb)
                while True:                                          # the tensors' bytes, as they are
                    chunk = f.read(1 << 24)
                    if not chunk:
                        break
                    g.write(chunk)
    return report_line(str(src), change, n0, n1, size)


def hub_dry_run(repo, *, revision="main", limit=None, columnar=None, workers=8):
    """What stamping ``repo`` would change, read by range requests only: ``(rows, manifest_change)``."""
    from dmipy_replay.replay import hub
    sha = hub.revision_of(repo, revision)
    if columnar is not None:
        path = f"{columnar.strip('/')}/manifest.json"
        body = hub._get(hub._resolve_url(repo, path, sha))
        manifest = json.loads(body)
        meta, change = stamp_meta(manifest["meta"])
        n1 = len(json.dumps(dict(manifest, meta=meta)).encode()) if change else len(body)
        return sha, [report_line(f"hf://{repo}/{path}", change, len(body), n1, n1)], None
    m = hub.manifest(repo, revision=sha) or {}
    rows = [r for r in (m.get("packs") or []) if str(r.get("path", "")).endswith(".rpk")]
    if limit:
        rows = rows[:int(limit)]

    def one(row):
        url = hub._resolve_url(repo, row["path"], sha)
        n = struct.unpack("<Q", hub._get(url, byte_range=(0, 7)))[0]
        try:
            hb, change, n0, n1 = stamp_header(hub._get(url, byte_range=(8, 8 + n - 1)))
        except ValueError as e:
            return dict(file=f"hf://{repo}/{row['path']}", id=row.get("id"), stamped=False, refused=str(e))
        return report_line(f"hf://{repo}/{row['path']}", change, n0, n1, row.get("bytes"))

    with ThreadPoolExecutor(max_workers=int(workers)) as ex:
        out = list(ex.map(one, rows))
    man_change = None
    acq = (m.get("replay_envelope") or {}).get("acquisition")
    if acq:                                            # the dataset's summary: its acquisition block, as its packs'
        _, man_change = stamp_meta(dict(id=f"{repo}/manifest.json", replay_envelope=dict(acquisition=acq),
                                        compression=dict(temporal_bandwidth_hz=_one_band(out))))
    return sha, out, man_change


def _one_band(rows):
    """The one band the stamped packs of a dataset state, which its manifest's summary states with them."""
    bands = {r["after"]["band_hz"] for r in rows if r.get("stamped")}
    if len(bands) != 1:
        raise ValueError(f"the dataset's stamped packs state {sorted(bands)} bands: its manifest summary needs one")
    return bands.pop()


def summary(rows):
    stamped = [r for r in rows if r.get("stamped")]
    refused = [r for r in rows if r.get("refused")]
    moves = [r for r in stamped if r["data_moves"]]
    total = sum(int(r["rewrite_bytes"] or 0) for r in stamped)
    return dict(files=len(rows), stamped=len(stamped), unchanged=len(rows) - len(stamped) - len(refused),
                refused=len(refused), data_moves=len(moves), rewrite_bytes=total)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("paths", nargs="*", help="local .rpk files or columnar manifest.json files")
    ap.add_argument("--out", help="write the stamped files here (default: dry run, nothing written)")
    ap.add_argument("--hub", help="a Hub dataset to dry-run, read by range requests only")
    ap.add_argument("--revision", default="main")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--columnar", default=None, help="with --hub: the prefix of a columnar layout (its manifest.json)")
    ap.add_argument("--report", help="write the rows as JSON here")
    ap.add_argument("--show", type=int, default=3, help="rows printed in full (the summary covers all)")
    a = ap.parse_args(argv)
    if a.hub and a.out:
        ap.error("--hub is a dry run: this tool never writes to the Hub")
    if a.hub:
        sha, rows, man = hub_dry_run(a.hub, revision=a.revision, limit=a.limit, columnar=a.columnar)
        print(f"hf://{a.hub} @ {sha} (dry run, read-only)")
        if man is not None:
            print("manifest.json replay_envelope.acquisition:", json.dumps(man["before"]), "->", json.dumps(man["after"]))
    else:
        rows = [stamp_file(p, a.out) for p in a.paths]
        print("dry run: nothing written" if a.out is None else f"written to {a.out}")
    for r in rows[:a.show]:
        print(json.dumps(r))
    print("summary:", json.dumps(summary(rows)))
    if a.report:
        Path(a.report).write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
