"""A replay pack in columns, read by reference (dmipy-sim#290): rows ordered by (block, voxel, pool), one file per
stored tensor in parts, the position bands and the field modes in groups, the pack's meta once in ``manifest.json``
with every column's byte layout, and ``index.json`` with the row range of every (voxel, pool). A consumer opens the
layout where it is -- a directory, or ``hf://owner/name/prefix`` on the Hugging Face Hub, read by HTTP range -- and
gets :class:`~dmipy_replay.replay.replay.ReplayPack` VIEWS of the rows and bands an acquisition needs; nothing is
downloaded whole, and every byte fetched is one the replay uses.

    pack = open_columnar("hf://SubstrateCommons/disco-replay/disco")
    plan = pack.plan(seq, tissue=..., scanner=...)            # bands, modes, tiers, bytes: before any transfer
    plan = pack.plan_for_envelope(envelope_of("prisma"))       # what every waveform a Prisma can play needs
    view = pack.view(K=32, voxels=[(20, 20, 20)])              # one voxel's rows at 32 bands: an ordinary ReplayPack
    S, floor, plan = pack.image([seq], settings=[(None, None), (tissue, 3.0)])   # every setting's volume, one pass

The layout is written by dmipy-sim's ``fill.consolidate``.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
import urllib.error
import urllib.request

import numpy as np

from . import hub
from ..phantom.grid import Grid

__all__ = ["ColumnarPack", "open_columnar", "Source"]

POOL_KEY = "comp_static"
RETRIES = 6
SLAB_ROWS = 1 << 15         #: rows whose exponential is formed at once in the numpy voxel sums


def _grid_of(meta):
    return Grid.from_meta(meta["fidelity"]["per_voxel"]["grid"])


class Source:
    """Bytes of a column file by range: a directory, or a Hub dataset through HTTP range requests at one pinned
    commit (``revision`` resolved to its sha when the layout is opened, so every range of a session reads the same
    files), the CDN location of every file resolved once (afresh after a failed read), the ranges of a view fetched
    concurrently (``workers`` requests in flight), a failed range asked for again. A server that answers a range
    request with anything but the bytes asked for is retried, never sliced: a whole column part is gigabytes."""

    def __init__(self, uri, *, revision="main", workers=8):
        self.uri = uri; self.bytes_read = 0; self.requests = 0; self.workers = workers
        self._resolved = {}; self._lock = threading.Lock()
        self.remote = hub.is_hub_uri(uri)
        if self.remote:
            repo, self.prefix = hub.parse_uri(uri)
            self.repo = repo
            self.revision = hub.revision_of(repo, revision)

    def _url(self, rel):
        return hub._resolve_url(self.repo, f"{self.prefix}/{rel}" if self.prefix else rel, self.revision)

    def _headers(self, **extra):
        h = {"User-Agent": "dmipy-replay", **extra}
        tok = hub._token()
        if tok:
            h["Authorization"] = f"Bearer {tok}"
        return h

    def _range(self, url, start, length):
        req = urllib.request.Request(url, headers=self._headers(Range=f"bytes={start}-{start + length - 1}"))
        with urllib.request.urlopen(req, timeout=hub.TIMEOUT_S) as r:
            if r.status != 206:
                raise IOError(f"status {r.status} to a range request")
            body = r.read(length + 1)
            return body, r.geturl()

    def read(self, rel, start, length):
        with self._lock:
            self.requests += 1; self.bytes_read += length
        if not self.remote:
            with open(os.path.join(self.uri, rel), "rb") as fh:
                fh.seek(start); return fh.read(length)
        err = None
        for attempt in range(RETRIES):
            try:
                with self._lock:
                    loc = self._resolved.get(rel)
                body, final = self._range(loc or self._url(rel), start, length)
                if len(body) == length:
                    with self._lock:
                        self._resolved[rel] = final
                    return body
                err = f"{len(body)} of {length} bytes"
            except (OSError, urllib.error.URLError) as e:
                err = repr(e)
            with self._lock:
                self._resolved.pop(rel, None)
            time.sleep(min(60, 2 * 2 ** attempt))
        raise IOError(f"range read of {rel} [{start}, +{length}) failed after {RETRIES} attempts: {err}")

    def read_many(self, jobs):
        """``[(rel, start, length), ...]`` fetched concurrently, in order."""
        from concurrent.futures import ThreadPoolExecutor
        if not self.remote or len(jobs) == 1:
            return [self.read(*j) for j in jobs]
        with ThreadPoolExecutor(max_workers=self.workers) as ex:
            return list(ex.map(lambda j: self.read(*j), jobs))

    def text(self, rel):
        if not self.remote:
            with open(os.path.join(self.uri, rel)) as fh:
                return fh.read()
        return hub._get(self._url(rel)).decode()


def _numpy_voxel_sums(phi, ew, starts):
    """``(n_seg, n_meas)`` complex: ``sum_rows ew exp(i phi)`` over each run of rows that begins at ``starts``, the
    exponential formed a slab of rows at a time so a row group never holds ``rows x n_meas`` complex numbers."""
    n, n_m = phi.shape
    seg_of = np.searchsorted(starts, np.arange(n), side="right") - 1
    out = np.zeros((len(starts), n_m), complex)
    for a in range(0, n, SLAB_ROWS):
        b = min(n, a + SLAB_ROWS)
        E = np.exp(1j * phi[a:b]) * ew[a:b, None]
        s = seg_of[a:b]
        cut = np.flatnonzero(np.r_[True, s[1:] != s[:-1]])
        out[s[cut]] += np.add.reduceat(E, cut, axis=0)
    return out


class ColumnarPack:
    """A pack in the columnar layout, open by reference. ``meta`` is the merged pack's header, ``columns`` the byte
    layout of every column (its parts), ``index`` the row range of every (voxel, pool), ``grid`` the certificate's
    grid, ``floor`` the certificate's median per-voxel floor.

    :attr:`pack_class` is the class a :meth:`view` is (a producer's subclass of
    :class:`~dmipy_replay.replay.replay.ReplayPack` sets its own), and :attr:`IMAGE_BACKENDS` the reductions
    :meth:`image` can form its voxel sums with, ``"numpy"`` here."""

    IMAGE_BACKENDS = {"numpy": lambda: _numpy_voxel_sums}

    def __init__(self, uri, *, revision="main", workers=8):
        from .replay import ReplayPack, _refuse_predating
        self.pack_class = ReplayPack
        self.uri = uri; self.src = Source(uri, revision=revision, workers=workers)
        m = json.loads(self.src.text("manifest.json")); self.meta, self.columns = m["meta"], m["columns"]
        _refuse_predating(self.meta, uri, tensors=list(self.columns))
        self.index = json.loads(self.src.text("index.json"))
        self.n_rows = int(self.index["n_rows"]); self.band_groups = list(self.index["band_groups"]); self.path_groups = list(self.index.get("path_groups", []))
        self.grid = _grid_of(self.meta)
        self.floor = float(self.meta["fidelity"]["per_voxel"]["floor_median"])
        self.K = int(self.meta["compression"]["K"])
        self.susceptibility_field

    @property
    def pack_license(self):
        """The terms the layout is published under (its header's ``pack_license``), or None if it states none."""
        return self.meta.get("pack_license")

    @property
    def dt(self):
        """The block length ``h`` (s) of the walk the layout stores (``walk_params.block``)."""
        return float(self.meta["walk_params"]["block"]["h"])

    @property
    def n_t(self):
        """The block ends of the walk, ``T_max / h + 1``: the grid a waveform built for the layout is sampled on."""
        return int(round(float(self.meta["walk_params"]["T_max"]) / self.dt)) + 1

    @property
    def T_window(self):
        """One storage window's duration (s): the interval the bands are on."""
        return float(self.meta["walk_params"]["windows"]["T"])

    @property
    def susceptibility_field(self):
        """The layout's susceptibility field, ``"present"`` or ``"absent"``, from its embedded spec's pools with its
        stored field modes as the field channel (:func:`~dmipy_replay.replay.replay.declared_susceptibility_field`);
        a layout without an embedded spec, or one that breaks the equivalence, is refused when it is opened."""
        from .replay import declared_susceptibility_field
        return declared_susceptibility_field(self.meta, has_field=bool(self.path_groups))

    @property
    def truncation_budget(self):
        """What a waveform may cost this layout in band truncation (:func:`~dmipy_replay.replay.replay.truncation_budget`):
        its ``fidelity.target_floor`` less its other certified terms; a layout that states no cap is held to its median
        per-voxel floor."""
        from .replay import truncation_budget
        return truncation_budget(self.meta, floor=self.floor, floor_source="fidelity.per_voxel.floor_median")

    # ---- the plan: what an acquisition needs, before any byte moves
    def bands_for(self, seq, tol=0.25):
        """``(K', error)``: the fewest stored band groups whose truncation costs less than ``tol`` x the pack's median
        floor on every measurement, and that truncation's bound, by the rule of :mod:`~dmipy_replay.replay.planning`
        (:func:`~dmipy_replay.replay.planning.band_tail` and :func:`~dmipy_replay.replay.planning.fewest`): the stored bands
        between ``K'`` and ``K`` at the manifest's per-band variance of the worst populated voxel, and the bands beyond
        the layout's envelope (its ``band_hz``, at most the ``K`` it stores; never read) at the free bridge's variance at
        the walk's diffusivity, in closed form to infinity
        (:func:`~dmipy_replay.replay.compression.band_phase_variances`), which bounds every restricted walk's. A waveform
        whose cost there exceeds :attr:`truncation_budget` is refused by name; the walk is not read."""
        from .envelope import Envelope
        from .planning import band_tail, fewest
        from .compression import band_phase_variances
        env = Envelope.from_header(self.meta)
        K = min(self.K, env.mode_count(self.T_window))
        G = np.asarray(seq.G_eff, np.float64)
        D = self.meta["walk_params"].get("diffusivity")
        if D is None:
            raise ValueError("the layout records no diffusivity, so the truncation beyond its stored bands cannot be bounded")
        pb, total = band_phase_variances(G, float(seq.dt), self.T_window, K, float(D))
        beyond = 0.5 * np.clip(total - pb.sum(axis=1), 0.0, None)                  # the bands never read, per measurement
        budget = self.truncation_budget
        if beyond.max() > budget["budget"]:
            raise ValueError(f"the waveform's gradient reaches beyond this layout's temporal band: the {K} bands of its "
                             f"envelope {env.name!r} ({K / (2.0 * self.T_window):.4g} Hz) leave up to {beyond.max():.2e} of the "
                             f"signal in phase the layout does not carry, against the {budget['budget']:.2e} it may cost (its "
                             f"cap {budget['cap']:.2e} from {budget['cap_source']} less {budget['other_terms']:.2e} of other "
                             "certified terms)")
        var = np.asarray(self.meta["columnar"].get("band_variance_max") or self.meta["columnar"]["band_variance"], np.float64).T   # (K, 3)
        tail = (band_tail(G, seq.dt, [(self.T_window, None)], var[:K]) + beyond[:, None]).max(axis=0)
        return fewest(tail, [g for g in self.band_groups if g <= K] or [K], tol * self.floor)

    def bands_for_envelope(self, envelope):
        """``K'``: the fewest stored band groups holding the band of ``envelope`` (an
        :class:`~dmipy_replay.replay.envelope.Envelope`, :func:`~dmipy_replay.replay.envelope.envelope_of` for a scanner
        class) over one window -- what every waveform inside it needs, whatever the waveform. An envelope whose band is
        wider than the layout's own is refused by name."""
        from .envelope import Envelope
        if not isinstance(envelope, Envelope):
            raise TypeError(f"bands_for_envelope takes an Envelope (envelope_of(scanner) for a scanner class); got "
                            f"{type(envelope).__name__}")
        own = Envelope.from_header(self.meta)
        need = envelope.mode_count(self.T_window)
        have = min(self.K, own.mode_count(self.T_window))
        if need > have:
            raise ValueError(f"the envelope {envelope.name!r} needs {need} bands per {self.T_window * 1e3:.4g} ms window "
                             f"({envelope.band_hz:.4g} Hz) and this layout is certified for {have} (its envelope "
                             f"{own.name!r}, {own.band_hz:.4g} Hz): it was built for a narrower band")
        return int(next(g for g in self.band_groups if g >= need))

    def modes_for(self, seq, scanner, tissue, tol=0.25):
        """``(M, error)``: the fewest stored field modes for a replay at the scanner's field with the tissue's chi (0
        when :func:`~dmipy_replay.replay.replay.field_term` resolves the pair to no field term), by the rule of
        :mod:`~dmipy_replay.replay.planning` (:func:`~dmipy_replay.replay.planning.mode_tail`) on the manifest's per-mode variance."""
        from .replay import field_term
        from .planning import mode_tail, fewest
        f = field_term(self.susceptibility_field, tissue, scanner)
        if not f.active:
            return 0, 0.0
        if not self.path_groups:
            raise ValueError("this pair has a field term and the layout stores no field modes (C3 path channel)")
        gate = seq.effective_gate if seq.gate is None else seq.gate
        var = np.asarray(self.meta["columnar"]["path_mode_variance"], np.float64).sum(axis=0)
        tail = mode_tail(gate, seq.dt, [(self.T_window, None)], self.path_groups[-1], var, f.B0, abs(f.chi_iso) + abs(f.chi_aniso))
        return fewest(tail, self.path_groups, tol * self.floor)

    def _tiers_of(self, tissue):
        contact = tissue is not None and (tissue.rho2 is not None or tissue.rho1 is not None)
        relax = tissue is not None and (tissue.T2 is not None or tissue.T1 is not None)
        return contact, relax

    def _plan(self, K, eK, M, eM, contact, relax, voxels):
        ranges = self.rows_of(voxels); rows = sum(e - s for s, e in ranges)
        names = self._names(K, M, contact, relax)
        per_row = sum(int(np.prod(self.columns[n]["shape"][1:], dtype=np.int64)) * np.dtype(self.columns[n]["dtype"]).itemsize for n in names)
        return dict(K=K, band_error=eK, modes=M, mode_error=eM, contact=contact, relaxation=relax, rows=rows,
                    bytes=per_row * rows, bytes_per_row=per_row, columns=names)

    def plan(self, seq, *, tissue=None, scanner=None, tol=0.25, voxels=None):
        """What a replay of ``seq`` reads: bands, field modes, tiers, rows and bytes -- before any byte moves. A
        waveform outside the envelope the layout is certified for is refused first
        (:func:`~dmipy_replay.replay.replay.check_acquisition_envelope`)."""
        from .replay import check_acquisition_envelope, field_envelope_of, field_term
        f = field_term(self.susceptibility_field, tissue, scanner, field_envelope_of(self.meta))
        check_acquisition_envelope(self.meta, seq, field_active=f.active)
        K, eK = self.bands_for(seq, tol); M, eM = self.modes_for(seq, scanner, tissue, tol)
        return self._plan(K, eK, M, eM, *self._tiers_of(tissue), voxels)

    def plan_for_envelope(self, envelope, *, tissue=None, scanner=None, voxels=None):
        """What a replay of ANY waveform inside ``envelope`` reads (an :class:`~dmipy_replay.replay.envelope.Envelope`;
        :func:`~dmipy_replay.replay.envelope.envelope_of` for a scanner class): its bands (:meth:`bands_for_envelope`),
        every stored field mode when the tissue and the field make a field term (the gate of a waveform not yet named
        cannot be projected), the tiers the tissue asks for, rows and bytes."""
        from .replay import field_term
        K = self.bands_for_envelope(envelope)
        M = int(self.path_groups[-1]) if field_term(self.susceptibility_field, tissue, scanner).active else 0
        return self._plan(K, None, M, None, *self._tiers_of(tissue), voxels)

    def _names(self, K, M, contact, relaxation):
        """The per-row columns a replay at ``K`` bands, ``M`` field modes and the tiers asked for reads: every
        column of the manifest except the band groups beyond ``K``, the mode groups beyond ``M``, and the contact
        channel unless asked for."""
        n_b = len([g for g in self.band_groups if g <= K]); n_m = len([g for g in self.path_groups if g <= M]) if M else 0
        names = []
        for n, c in self.columns.items():
            if not c["per_row"]:
                continue
            mb = re.fullmatch(r"pos_[xyz]_b(\d+)(?:u\d+)?", n)
            mp = re.fullmatch(r"susc_path_dct_b(\d+)(?:u\d+)?", n)
            if mb:
                if int(mb.group(1)) >= n_b:
                    continue
            elif mp:
                if int(mp.group(1)) >= n_m:
                    continue
            elif n.startswith("susc_path_m"):
                if int(n[len("susc_path_m"):]) >= n_m:
                    continue
            elif n.startswith("blt_") and not contact:
                continue
            names.append(n)
        return names

    # ---- rows
    def rows_of(self, voxels=None, pools=None):
        """Contiguous row ranges ``(start, end)`` of the voxels (ijk triples; all when None) and pools asked for,
        adjacent ranges merged; a (voxel, pool) walked in two passes has two."""
        rs = []
        want = None if voxels is None else {tuple(int(x) for x in v) for v in voxels}
        for r in self.index["rows"]:
            if (want is None or tuple(r["ijk"]) in want) and (pools is None or r["pool"] in pools):
                if rs and rs[-1][1] == r["start"]:
                    rs[-1] = (rs[-1][0], r["end"])
                else:
                    rs.append((r["start"], r["end"]))
        return rs

    def _parts(self, c):
        if "parts" in c:
            return c["parts"]
        return [{"file": c["file"], "rows": ([0, c["shape"][0]] if c["per_row"] else None), "data_offset": c["data_offset"], "nbytes": c["nbytes"]}]

    def _jobs(self, name, ranges):
        c = self.columns[name]; dt = np.dtype(c["dtype"]); shape = c["shape"]; parts = self._parts(c)
        if not c["per_row"]:
            p = parts[0]
            return [(p["file"], p["data_offset"], p["nbytes"])], dt, shape, [None]
        rowbytes = int(np.prod(shape[1:], dtype=np.int64)) * dt.itemsize
        jobs, rs = [], []
        for s, e in ranges:
            for p in parts:
                ps, pe = p["rows"]; lo, hi = max(s, ps), min(e, pe)
                if lo < hi:
                    jobs.append((p["file"], p["data_offset"] + (lo - ps) * rowbytes, (hi - lo) * rowbytes)); rs.append((lo, hi))
        return jobs, dt, shape, rs

    def columns_over(self, names, ranges):
        """The columns as arrays over ``ranges``, every byte range of every column fetched concurrently."""
        plan = {n: self._jobs(n, ranges) for n in names}
        jobs = [j for n in names for j in plan[n][0]]
        raws = iter(self.src.read_many(jobs)); out = {}
        for n in names:
            js, dt, shape, rs = plan[n]
            parts = []
            for j, r in zip(js, rs):
                raw = next(raws)
                parts.append(np.frombuffer(raw, dt).reshape(shape if r is None else (r[1] - r[0],) + tuple(shape[1:])))
            out[n] = np.concatenate(parts, axis=0) if len(parts) > 1 else parts[0]
        return out

    # ---- views
    def view(self, *, K=None, modes=0, voxels=None, pools=None, contact=False, relaxation=True, ranges=None):
        """A :attr:`pack_class` pack of the rows and bands asked for: ``K`` bands (rounded up to a stored group; the
        pack's K when None), ``modes`` field modes (0: the field tier left out), the contact tier with ``contact``."""
        ranges = self.rows_of(voxels, pools) if ranges is None else ranges
        K = self.K if K is None else int(K)
        K_read = next((g for g in self.band_groups if g >= K), self.band_groups[-1])
        M_read = 0 if not modes else next((g for g in self.path_groups if g >= modes), self.path_groups[-1])
        tables = [n for n, c in self.columns.items() if not c["per_row"] and (contact or not n.startswith("blt_")) and (M_read or n != "susc_path_scale")]
        names = self._names(K_read, M_read, contact, relaxation) + tables
        arrays = self.columns_over(names, ranges)
        if "pos_band_scale" in arrays:
            arrays["pos_band_scale"] = np.ascontiguousarray(arrays["pos_band_scale"][..., :K_read])
        else:                                                              # the float coefficients: the ends and the groups as one array
            n_b = len([g for g in self.band_groups if g <= K_read])
            for a in "xyz":
                arrays[f"pos_{a}"] = np.concatenate([arrays.pop(f"pos_{a}_ends")] + [arrays.pop(f"pos_{a}_b{i}") for i in range(n_b)], axis=1)
        meta = json.loads(json.dumps(self.meta)); meta["compression"]["K"] = int(K_read)
        if meta["compression"].get("container"):                           # the ranges read: a prefix of the container
            meta["compression"]["container"] = meta["compression"]["container"][:len([g for g in self.band_groups if g <= K_read])]
        pmeta = meta["compression"]["channels"].get("susc_field_basis") or {}
        if M_read:
            n_m = len([g for g in self.path_groups if g <= M_read])
            if pmeta.get("container"):                                      # the container's ranges, kept as stored: a prefix of them
                pmeta["container"] = pmeta["container"][:n_m]
            else:
                arrays["susc_path_dct"] = np.concatenate([arrays.pop(f"susc_path_m{i}") for i in range(n_m)], axis=2)
            arrays["susc_path_scale"] = np.ascontiguousarray(arrays["susc_path_scale"][..., :M_read])
            meta["compression"]["channels"]["susc_field_basis"]["K"] = int(M_read)
        else:
            meta["compression"]["channels"].pop("susc_field_basis", None); meta["compression"]["channels"].pop("susceptibility_grid", None)
            if self.path_groups:
                meta["view"] = dict(field="unread")                    # the layout holds C3; this view reads none of it
        if not contact:
            meta["compression"]["channels"].pop("boundary_local_time", None)
        return self.pack_class(arrays, meta)

    def row_groups(self, chunk_rows=2_000_000):
        """The index rows in groups of about ``chunk_rows`` rows, cut on (voxel, pool) boundaries: the groups
        :meth:`iter_views` reads, each a list of the index's ``{ijk, pool, start, end}`` rows in row order."""
        groups, cur, n = [], [], 0
        for r in self.index["rows"]:
            cur.append(r); n += r["end"] - r["start"]
            if n >= chunk_rows:
                groups.append(cur); cur, n = [], 0
        if cur:
            groups.append(cur)
        return groups

    @staticmethod
    def group_ranges(rows):
        """The contiguous row ranges ``(start, end)`` of a group of index rows, adjacent ranges merged."""
        rs = []
        for r in rows:
            if rs and rs[-1][1] == r["start"]:
                rs[-1] = (rs[-1][0], r["end"])
            else:
                rs.append((r["start"], r["end"]))
        return rs

    def iter_views(self, chunk_rows=2_000_000, *, voxels=None, **kw):
        """The pack in row groups of about ``chunk_rows`` on (voxel, pool) boundaries (:meth:`row_groups`; only the
        ``voxels`` asked for when given), each a view (``**kw`` as :meth:`view`), the next group's bytes fetched
        while the current one is contracted; a failed fetch reaches the consumer."""
        import queue
        want = None if voxels is None else {tuple(int(x) for x in v) for v in voxels}
        rows = [r for r in self.index["rows"] if want is None or tuple(r["ijk"]) in want]
        groups, cur, n = [], [], 0
        for r in rows:
            cur.append(r); n += r["end"] - r["start"]
            if n >= chunk_rows:
                groups.append(self.group_ranges(cur)); cur, n = [], 0
        if cur:
            groups.append(self.group_ranges(cur))
        q = queue.Queue(maxsize=1)

        def fetch():
            try:
                for g in groups:
                    q.put(self.view(ranges=g, **kw))
                q.put(None)
            except BaseException as e:
                q.put(e)
        threading.Thread(target=fetch, daemon=True).start()
        while True:
            v = q.get()
            if v is None:
                return
            if isinstance(v, BaseException):
                raise v
            yield v

    # ---- the image loop
    def image(self, seq, *, tissue=None, scanner=None, settings=None, tol=0.25, voxels=None, chunk_rows=2_000_000,
              backend="numpy", progress=None):
        """``(S, floor, plan)``: the signal of every voxel of the grid under ``seq`` (one sequence or a list, their
        measurements side by side; NaN where the pack has none, and outside ``voxels`` when given), the
        certificate's floor beside it (the worst pool's), and the plan with what was read. One pass over the rows:
        each walker's phase and weight (:meth:`ReplayPack.walker_phases`) on the host, the exponential and the sum
        over each voxel's rows by ``backend`` (:attr:`IMAGE_BACKENDS`). With ``settings`` (a list of ``(tissue,
        scanner)``) every setting's volume comes from the same pass -- the settings differ only in what is applied
        to the rows -- and ``S`` gains a leading axis. ``progress(rows, bytes, seconds)`` is called after every
        row group."""
        sums_of = self.IMAGE_BACKENDS[backend]()
        seqs = list(seq) if isinstance(seq, (list, tuple)) else [seq]
        multi = settings is not None; settings = list(settings) if multi else [(tissue, scanner)]
        t0 = time.time(); plans = [self.plan(s, tissue=t_, scanner=B_, tol=tol, voxels=voxels) for s in seqs for t_, B_ in settings]
        plan = max(plans, key=lambda p: (p["K"], p["modes"]))
        plan = dict(plan, K=max(p["K"] for p in plans), modes=max(p["modes"] for p in plans), contact=any(p["contact"] for p in plans), settings=len(settings))
        self.src.bytes_read = self.src.requests = 0
        n_meas = sum(int(np.asarray(s.G_eff).shape[0]) for s in seqs); n_vox = int(np.prod(self.grid.shape))
        num = np.zeros((len(settings), n_vox, n_meas), complex); den = np.zeros(n_vox); rows = 0
        for pk in self.iter_views(chunk_rows=chunk_rows, voxels=voxels, K=plan["K"], modes=plan["modes"], contact=plan["contact"]):
            n = pk.n_walkers
            ijk, _ = self.grid.bin(pk.r0); v = np.ravel_multi_index(ijk.T, self.grid.shape)
            starts = np.flatnonzero(np.r_[True, v[1:] != v[:-1]])            # runs of one voxel
            w = None
            for si, (t_, B_) in enumerate(settings):
                sums = []
                for s in seqs:                                                # each sequence's own relaxation and contact weights
                    w, ew, p = pk.walker_phases(s, tissue=t_, scanner=B_, tol=None)
                    sums.append(np.asarray(sums_of(np.asarray(p, np.float64), np.asarray(ew, np.float64), starts)))
                np.add.at(num[si], v[starts], np.concatenate(sums, axis=1))
            np.add.at(den, v[starts], np.add.reduceat(np.asarray(w), starts))   # a voxel may sit in two row groups
            rows += n
            if progress:
                progress(rows, self.src.bytes_read, time.time() - t0)
        S = np.full((len(settings), n_vox, n_meas), np.nan); m = den > 0; S[:, m] = np.abs(num[:, m] / den[m][None, :, None])
        floor = np.full(n_vox, np.nan); cert = self.columns_over(["voxel_ijk", "voxel_certificate"], [(0, 0)])
        vi = np.ravel_multi_index(np.asarray(cert["voxel_ijk"]).T, self.grid.shape)
        fl = np.asarray(cert["voxel_certificate"])[:, :, 1]; known = np.isfinite(fl).any(axis=1)
        floor[vi[known]] = np.nanmax(fl[known], axis=1)
        plan.update(bytes_read=self.src.bytes_read, requests=self.src.requests, seconds=time.time() - t0, rows=rows)
        S = S.reshape((len(settings),) + tuple(self.grid.shape) + (n_meas,))
        return (S if multi else S[0]), floor.reshape(self.grid.shape), plan


def open_columnar(uri, *, revision="main", workers=8):
    """The columnar layout at ``uri`` (a directory, or ``hf://owner/name/prefix``) as a :class:`ColumnarPack`."""
    return ColumnarPack(uri, revision=revision, workers=workers)
