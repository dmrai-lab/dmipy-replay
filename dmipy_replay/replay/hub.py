"""Packs on the Hugging Face Hub, read with the standard library: ``hf://owner/name/path/to/pack.rpk``.

A dataset holds its packs and a ``manifest.json`` (``{"schema": "substratecommons/1", "packs": [row, ...]}``, one row
per pack: its path, sha256, bytes, licence, band, floor). A read pins the dataset's revision once, so the manifest and
the pack it checks come from the same commit, and caches by that revision: a whole pack is downloaded once and its
sha256 checked against its row; :func:`fetch_windows` reads only a pack's leading storage windows (RPK.md 4.3) by HTTP
byte range. A token is read from ``HF_TOKEN`` or the Hub client's own token file when present; public datasets need none.
"""
import hashlib
import json
import os
import shutil
import tempfile
import urllib.error
import urllib.parse
import urllib.request

import numpy as np

from .container import parse_header, tensor_of, read_safetensors, write_safetensors

ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
MANIFEST = "manifest.json"
TIMEOUT_S = 60

__all__ = ["fetch", "fetch_windows", "fetch_tree", "manifest", "packs", "parse_uri", "is_hub_uri", "revision_of", "cache_dir"]


# ------------------------------- URIs -------------------------------
def is_hub_uri(x):
    """Whether ``x`` names a file on the hub (``hf://owner/name/path``)."""
    return isinstance(x, str) and x.startswith("hf://")


def parse_uri(uri):
    """``hf://owner/name/path/to/file`` -> ``("owner/name", "path/to/file")``; anything else is refused."""
    if not is_hub_uri(uri):
        raise ValueError(f"not a hub URI: {uri!r} (expected hf://owner/name/path/to/file.rpk)")
    parts = uri[len("hf://"):].split("/", 2)
    if len(parts) < 3 or not all(parts):
        raise ValueError(f"a hub URI names a repository and a file, hf://owner/name/path/to/file.rpk: {uri!r}")
    owner, name, path = parts
    return f"{owner}/{name}", path


def cache_dir():
    """Where packs are cached: ``$DMIPY_REPLAY_CACHE``, else ``~/.cache/dmipy-replay``."""
    d = os.environ.get("DMIPY_REPLAY_CACHE") or os.path.join(os.path.expanduser("~"), ".cache", "dmipy-replay")
    os.makedirs(d, exist_ok=True)
    return d


# ------------------------------- HTTP -------------------------------
def _token():
    tok = os.environ.get("HF_TOKEN")
    if tok:
        return tok.strip()
    home = os.environ.get("HF_HOME") or os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
    try:
        with open(os.path.join(home, "token")) as f:
            return f.read().strip() or None
    except OSError:
        return None


def _get(url, *, byte_range=None):
    """The body of ``url`` (bytes), or bytes ``[start, end]`` of it; a server that ignores the Range header and answers
    with the whole body is sliced to the same bytes rather than trusted as the range."""
    headers = {"User-Agent": "dmipy-replay"}
    tok = _token()
    if tok:
        headers["Authorization"] = f"Bearer {tok}"
    if byte_range is not None:
        headers["Range"] = f"bytes={int(byte_range[0])}-{int(byte_range[1])}"
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=TIMEOUT_S) as r:
        body = r.read()
        status = r.status
    if byte_range is not None and status != 206:
        body = body[int(byte_range[0]):int(byte_range[1]) + 1]
    return body


def _resolve_url(repo, path, revision):
    return f"{ENDPOINT}/datasets/{repo}/resolve/{revision}/{urllib.parse.quote(path)}"


def revision_of(repo, revision="main"):
    """The commit sha a dataset's ``revision`` (a branch, a tag or a sha) names now."""
    info = json.loads(_get(f"{ENDPOINT}/api/datasets/{repo}/revision/{urllib.parse.quote(revision, safe='')}"))
    return info["sha"]


def _download(url, dest):
    tmp = tempfile.NamedTemporaryFile(dir=os.path.dirname(dest), delete=False)
    try:
        headers = {"User-Agent": "dmipy-replay"}
        tok = _token()
        if tok:
            headers["Authorization"] = f"Bearer {tok}"
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=TIMEOUT_S) as r:
            shutil.copyfileobj(r, tmp, 1 << 24)
        tmp.close()
        os.replace(tmp.name, dest)
    finally:
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def _cached(repo, revision, path, suffix=""):
    owner, name = repo.split("/", 1)
    d = os.path.join(cache_dir(), f"datasets--{owner}--{name}", revision)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, path.replace("/", "__") + suffix)


# ------------------------------- the manifest -------------------------------
def manifest(repo, *, revision="main"):
    """A dataset's ``manifest.json`` at ``revision``, or ``None`` when it has none."""
    sha = revision_of(repo, revision)
    local = _cached(repo, sha, MANIFEST)
    if not os.path.isfile(local):
        try:
            _download(_resolve_url(repo, MANIFEST, sha), local)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            raise
    with open(local) as f:
        return json.load(f)


def packs(repo, *, revision="main"):
    """The packs a dataset lists: its manifest's rows, each with the ``hf://`` URI it loads by under ``uri``."""
    m = manifest(repo, revision=revision) or {}
    return [dict(row, uri=f"hf://{repo}/{row['path']}") for row in m.get("packs") or []]


# ------------------------------- whole packs -------------------------------
def fetch(uri, *, revision="main"):
    """``hf://owner/name/path.rpk`` -> the file's local cache path, its sha256 checked against the dataset's manifest
    at the same revision when the manifest has a row for it (a mismatch raises). Refuses a URI that is not a ``.rpk``."""
    repo, path = parse_uri(uri)
    if not path.endswith(".rpk"):
        raise ValueError(f"{uri}: a pack is a .rpk file; the manifest and the card are not packs")
    sha = revision_of(repo, revision)
    local = _cached(repo, sha, path)
    if not os.path.isfile(local):
        _download(_resolve_url(repo, path, sha), local)
        m = manifest(repo, revision=sha) or {}
        rows = [r for r in m.get("packs") or [] if r.get("path") == path]
        if rows and rows[0].get("sha256"):
            got = sha256_of(local)
            if got != rows[0]["sha256"]:
                os.unlink(local)
                raise ValueError(f"{uri}: the file's sha256 is {got}, the manifest says {rows[0]['sha256']}; refusing to load it")
    return local


def fetch_tree(repo, paths, *, revision="main", workers=8):
    """The files ``paths`` of the dataset ``repo`` at ``revision`` (resolved to its commit sha once) -> the local
    directory that holds them at their paths in the repository: each downloaded once into the cache under that commit,
    ``workers`` at a time."""
    from concurrent.futures import ThreadPoolExecutor
    sha = revision_of(repo, revision)
    owner, name = repo.split("/", 1)
    root = os.path.join(cache_dir(), f"datasets--{owner}--{name}", sha, "tree")

    def one(path):
        local = os.path.join(root, *path.split("/"))
        if not os.path.isfile(local):
            os.makedirs(os.path.dirname(local), exist_ok=True)
            _download(_resolve_url(repo, path, sha), local)
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as ex:
        list(ex.map(one, list(paths)))
    return root


# ------------------------------- windowed load (RPK.md 4.3) -------------------------------
def window_prefix(windows):
    """``windows`` (a ``range`` or a sequence of window indices) as the plain count ``k`` it must be: a prefix
    ``0, 1, ..., k-1``. The walk is continuous (RPK.md 4.3), so a pack cannot serve window 2 without window 1,
    and any other selection is refused by name rather than silently read as a prefix."""
    seq = list(windows)
    if not seq or seq != list(range(len(seq))):
        raise ValueError(f"windows must be a prefix range(k) (0, 1, ..., k-1), the walk being continuous; got {windows!r}")
    return len(seq)


def _coalesce(ranges):
    """Sorted ``(start, end)`` inclusive byte ranges with every adjacent or overlapping pair merged into one."""
    out = []
    for s, e in sorted(ranges):
        if out and s <= out[-1][1] + 1:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out


def _window_tensor_names(names, i, per_window):
    """The tensor names of window ``i``: window 0's are the unprefixed per-window names, window ``i > 0``'s its
    ``s{i}/`` ones. The shared tensors (weights, a static label, the voxel tables) belong to no window and are
    fetched once."""
    if i == 0:
        return sorted(per_window)
    p = f"s{int(i)}/"
    return [n for n in names if n.startswith(p)]


def fetch_windows_from(url, k, local_path, *, revision):
    """The windowed read behind :func:`fetch_windows` from a plain resolvable ``url`` (a Hub resolve URL, or a test
    server's): the header by two range requests, the shared tensors and windows ``0..k-1`` by coalesced ranges,
    written as a local pack holding just them. A cached file holding fewer windows is extended in place."""
    cached_k = cached = None
    if os.path.isfile(local_path):
        cached, hdr = read_safetensors(local_path)
        cached = {k_: np.array(v) for k_, v in cached.items()}
        cached_k = json.loads(hdr.get("rpk") or "{}").get("windows_present")
        if cached_k is not None and int(cached_k) >= k:
            return local_path
    n = int.from_bytes(_get(url, byte_range=(0, 7)), "little")
    _, header = parse_header(_get(url, byte_range=(0, 8 + n - 1)))
    base = 8 + n
    names = [x for x in header if x != "__metadata__"]
    meta = json.loads((header.get("__metadata__") or {}).get("rpk") or "{}")
    n_total = int(((meta.get("walk_params") or {}).get("windows") or {}).get("n") or 1)
    if k > n_total:
        raise ValueError(f"{url}: this pack stores {n_total} window(s); windows=range({k}) asks for more than it has")
    per_window = {x[3:] for x in names if x.startswith("s1/")}
    start = 0 if cached_k is None else int(cached_k)
    wanted = [x for x in names if "/" not in x and x not in per_window] if start == 0 else []
    for i in range(start, k):
        wanted += _window_tensor_names(names, i, per_window)
    spans = {x: (header[x]["data_offsets"][0] + base, header[x]["data_offsets"][1] + base - 1) for x in wanted}
    merged = _coalesce([s for s in spans.values() if s[1] >= s[0]])
    blobs = {r: _get(url, byte_range=r) for r in merged}
    arrays = dict(cached or {})
    for x, (s, e) in spans.items():
        if e < s:
            arrays[x] = tensor_of(b"", dict(header[x], data_offsets=[0, 0]), x)
            continue
        ms, me = next(r for r in merged if r[0] <= s and e <= r[1])
        entry = dict(header[x], data_offsets=[s - ms, e - ms + 1])
        arrays[x] = tensor_of(blobs[(ms, me)], entry, x)
    out = dict(meta, windows_present=k)
    prov = dict(out.get("provenance") or {})
    # a part of a file has no sha256 of the whole to be checked against: what is checkable is recorded instead
    prov["hub_window_read"] = dict(revision=revision, windows_present=k, bytes_fetched=int(sum(e - s + 1 for s, e in merged)))
    out["provenance"] = prov
    write_safetensors(local_path, arrays, {"rpk": json.dumps(out)})
    return local_path


def fetch_windows(uri, windows, *, revision="main", url=None):
    """``hf://owner/name/path.rpk``'s leading windows (RPK.md 4.3) -> the local path of a pack holding just them.
    ``windows`` is a prefix ``range(k)``; a later call for a larger ``k`` fetches only the windows it lacks. ``url=``
    reads from a plain HTTP server instead of the Hub (the tests)."""
    repo, path = parse_uri(uri)
    if not path.endswith(".rpk"):
        raise ValueError(f"{uri}: a pack is a .rpk file; the manifest and the card are not packs")
    k = window_prefix(windows)
    sha = revision if url is not None else revision_of(repo, revision)
    return fetch_windows_from(url or _resolve_url(repo, path, sha), k, _cached(repo, sha, path, ".windows"), revision=sha)
