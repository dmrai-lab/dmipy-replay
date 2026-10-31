"""The **time-segment layout**, read: a walk contracted once against the time segments of its timing classes, so that
every acquisition built on a stored partition -- at any amplitude, in any direction on each segment -- is a few
products per walker.

**Terms, not a shape.** A layout stores PARTITIONS (:class:`Partition`): TERMS, each an interval ``[a_s, b_s]`` of the
walk's clock and a temporal shape ``psi_s`` on it (:class:`Shape`, a stated record normalised to ``max |psi| = 1``), with
the sign of its class's effective gate on the interval (the coherence sign times the transverse gate of the RF schedule:
+1, -1, or 0 while the magnetisation is stored). A constant segment is the term ``psi = 1`` -- a PGSE timing class's two
lobes, a stimulated echo's lobe, mixing time and lobe, a double diffusion encoding's four lobes; a trapezoid with its
ramps, a cosine OGSE block or a designed waveform is a sampled shape; a tensor encoding is several shapes on ONE
interval. Per walker and term it holds the **term vector** ``m_s = gamma int psi_s(t) 1_[a_s, b_s](t) (r(t) - r(0))
dt`` (rad per T/m): the walk's bands contracted against ``psi_s 1_[a_s, b_s]``
(:func:`~dmipy_replay.replay.compression.segment_projection`). An acquisition whose effective gradient is ``sum_s
psi_s(t) a_s`` with the vector ``a_s`` (T/m, the term's **level**) has the phase ``sum_s a_s . m_s`` per walker, the
start dropping out because a refocused encoding has no net area. PGSE, PGSTE, DDE, an unequal pair of one timing
class, a trapezoid at any amplitude and direction, a cosine OGSE, a tensor encoding at any rotation and weighting are
each a linear combination of the stored vectors formed in the image; no vector per acquisition is stored. The gate
(the refocusing schedule and the coherence sign), not the shape, fixes the relaxation, contact and field tiers, so
partitions of one gate share their tier columns.

**The fit.** :meth:`Partition.levels` finds the levels of a sequence, and a sequence fits only when its effective
gradient IS ``sum_s psi_s(t) a_s`` on its grid to the float32 rounding of its samples: a shape's vector serves that shape
alone, so a different waveform is refused, never served with an error no certificate states. Disjoint constant segments
take the gradient's value on each; shaped or overlapping terms take the least squares fit of the gradient on the terms'
shapes, and its residual is held to the same tolerance, :data:`_RTOL` of the peak. A sequence that plays a gradient
outside the terms or on a term its gate stores, is off the shapes (the partition, the row, the term the residual lies
in and the residual's relative norm are named), or leaves a net moment, is refused by name. A level beyond the envelope
the layout's widths are derived at (``G_max``, ``b_max``, the manifest's ``source.envelope``) is refused too.

**The words.** Each component of a term vector is a two's-complement bit field of a width the manifest states, a
walker's fields of one partition placed first-fit into 32-bit words, no field across a word; :func:`unpack_words`
decodes them (a shift up and an arithmetic shift down per field, a scale). The manifest records per term ``{"t",
"gate", "shape", "q_max", "floor_target", "step", "bits", "offsets", "word", "scale"}`` (``shape`` the term's
:meth:`Shape.to_meta` record, required: a term with none is refused by name) and its ``certificate``.

**The tiers.** Per sequence group (the partitions that share a grid, an RF schedule and a readout: the gate the tiers
are read under) the layout holds each walker's pool (``uint8``), its gated contact and its gated field channels
(``t_<g>.npy``, one word array), and optionally its background vector ``gamma int e(t) r(t) dt`` over the acquisition
(``bg_<g>.npy``; ``e`` the effective gate, what a magnet's constant gradient ``g0`` encodes as ``g0 . n``). The contact
enters as the weight ``exp((rho2 / D) c)`` and the field as ``B0 (chi_iso A + chi_aniso B)`` with ``A, B`` the channels
contracted for the field's direction (RPK.md 5.3: ``iso_local``, ``iso_P`` xx, yy, xy, xz, yz, ``aniso_G`` (6);
``iso_P_zz`` is never stored, the reader forms ``3 iso_local - iso_P_xx - iso_P_yy`` before the contraction). A
channel within half its step of zero on every row is not stored and contributes nothing.

**Files.** Rows in tiles of :data:`TILE`, every tile the rows of one voxel half (a voxel's rows alternate between its
two split halves in order), the last tile of a half padded with zero weight: ``w.npy`` ``(n_tiles, TILE)`` float32,
``tiles.npy`` ``(n_tiles,)`` (``2 voxel + half``), ``p_<partition>.npy`` ``(n_tiles, TILE, n_words)`` uint32, with
tiers ``pool.npy`` and ``t_<g>.npy``, with the background ``bg_<g>.npy``, and ``manifest.json`` (the partitions, the
groups, the widths, the floor, the source, the walk's identity, the certificate).

**The image.** :meth:`TimeSegmentLayout.image` takes a sequence (any rows, each fitting a stored partition) and forms
``|sum_w w exp(i sum_s G_s . m_s)| / sum_w w`` per voxel with its split-half floor, a chunk of tiles at a time: each
word decoded where it is read, float32 phases written out as products, complex64 exponentials, complex128 tile sums
and accumulator. ``tissue=`` / ``scanner=`` / ``b0_direction=`` turn the tiers into exactly
:meth:`~dmipy_replay.replay.primitives.Primitives.reduction_terms`: the row weight ``amp exp(-tau_2 / T2_pool -
tau_1 / T1_pool + (rho2 / D) c)`` and the phase ``B0 (chi_iso A + chi_aniso B)``. The arithmetic runs on a backend of
:data:`BACKENDS` (``"numpy"`` here); a backend is the same arithmetic on another array library and device.
"""
from __future__ import annotations

import contextlib
import functools
import json
import os
import itertools
from dataclasses import dataclass, field

import numpy as np

MANIFEST = "manifest.json"
FORMAT = "time_segments/2"
TILE = 128                                                            # rows per tile
WORD_BITS = 32                                                        # the word a walker's fields are packed into
_RTOL = 1e-5                                                          # a sequence's gradient is float32: the tolerance of every fit, of the peak
_AXES = np.eye(3)


def read_manifest(path):
    """The manifest of the layout in the directory ``path``; a manifest of another format is refused by name."""
    with open(os.path.join(path, MANIFEST)) as f:
        manifest = json.load(f)
    _check_format(manifest, path)
    return manifest


def _check_format(manifest, where):
    if manifest.get("format") != FORMAT:
        raise ValueError(f"{where} is not a {FORMAT} layout ({manifest.get('format')!r}); a layout of another format is "
                         "written again from its walk's segment rows (segment_rows, write_time_segment_layout)")


def _decode(words, scale, layout):
    """The fields of int32 ``words`` ``(..., n_words)`` on any array library (numpy, JAX, torch): each shifted to the top
    of its word and arithmetically back down, times its scale; ``None`` for a 0-bit field (it is zero)."""
    out = []
    for f, (w, off, b) in enumerate(layout):
        if b == 0:
            out.append(None); continue
        x = words[..., w]
        out.append(((x << (WORD_BITS - off - b)) >> (WORD_BITS - b)) * scale[f])
    return out


def unpack_words(words, scale, layout):
    """``(..., n_fields)`` float32: the values :func:`pack_words` stored (0 for a 0-bit field)."""
    w = np.asarray(words, np.uint32).view(np.int32)
    sc = np.asarray(scale, np.float32)
    vals = _decode(w, sc, layout)
    zero = np.zeros(w.shape[:-1], np.float32)
    return np.stack([zero if v is None else v.astype(np.float32) for v in vals], axis=-1)


def _sequence(seq):
    seq = seq.waveform if hasattr(seq, "waveform") else seq
    from ..acquisition.scanner_sequence import ScannerSequence
    if not isinstance(seq, ScannerSequence):
        raise TypeError(f"a time-segment layout reads a ScannerSequence; got {type(seq).__name__}")
    return seq


def _lobes(G, dt):
    """The maximal runs of samples on which every row's effective gradient is one non-zero constant, as ``[(a, b),
    ...]`` seconds: a sample-and-hold waveform's lobes."""
    n_t = G.shape[1]
    tol = _RTOL * (float(np.abs(G).max()) or 1.0)
    on = np.abs(G).max(axis=(0, 2)) > tol
    out, i = [], 0
    while i < n_t:
        if not on[i]:
            i += 1; continue
        j = i + 1
        while j < n_t and on[j] and np.abs(G[:, j] - G[:, i]).max() <= tol:
            j += 1
        out.append((i * float(dt), j * float(dt))); i = j
    return out


def _group_key(seq):
    """What the tiers of a sequence are read under: its grid, RF schedule and readout."""
    return json.dumps(dict(dt=float(seq.dt), n_t=int(seq.n_t), rf=seq.rf.to_dicts(), readout=[int(i) for i in seq.readout]),
                      sort_keys=True, default=float)


@dataclass(frozen=True)
class Shape:
    """The temporal shape ``psi(t)`` of a term, a stated record and nothing else. ``kind`` is ``"constant"`` (``psi`` is
    1 on the term: a segment) or ``"sampled"`` (``psi`` is held on ``len(values)`` equal cells of the term, as a
    :class:`~dmipy_replay.acquisition.scanner_sequence.ScannerSequence` holds its samples: a trapezoid with its ramps, a
    cosine, a designed waveform). A shape is normalised, ``max |psi| = 1``, so that the amplitude (the term's level,
    T/m) bounds the gradient and the stored vector's scale means what a segment's does; ``values`` are the normalised
    cell values. The projection of ``psi`` on the bands is exact for either kind
    (:func:`~dmipy_replay.replay.compression.segment_projection`): no quadrature stands between the stored shape and the
    stored vector. Shapes read and write the manifest as :meth:`to_meta` / :meth:`from_meta`."""
    kind: str
    values: tuple = ()

    @classmethod
    def constant(cls):
        """``psi = 1``."""
        return cls("constant", ())

    @classmethod
    def sampled(cls, values):
        """``psi`` held on ``len(values)`` equal cells, ``values`` divided by their largest magnitude."""
        v = np.asarray(values, np.float64).ravel()
        if v.size == 0 or not np.all(np.isfinite(v)) or not np.any(v != 0.0):
            raise ValueError("a sampled shape holds finite values, not all zero")
        return cls("sampled", tuple((v / np.abs(v).max()).tolist()))

    @classmethod
    def of_samples(cls, values):
        """The shape of the waveform samples ``values`` (one per cell): :meth:`constant` when they are one value to
        :data:`_RTOL`, else :meth:`sampled`."""
        v = np.asarray(values, np.float64).ravel()
        if v.size and np.abs(v / (v[np.abs(v).argmax()] or 1.0) - 1.0).max() <= _RTOL:
            return cls.constant()
        return cls.sampled(v)

    @property
    def is_constant(self):
        return self.kind == "constant"

    @property
    def cells(self):
        return 1 if self.is_constant else len(self.values)

    def projection_values(self):
        """The cell values :func:`~dmipy_replay.replay.compression.segment_projection` weights by (``None``: the indicator)."""
        return None if self.is_constant else np.asarray(self.values, np.float64)

    def weight(self, tau):
        """``int |psi| dt`` over a term of duration ``tau``: what bounds a vector's magnitude (``tau`` for a constant)."""
        return float(tau) if self.is_constant else float(tau) * float(np.abs(self.values).mean())

    def area(self, tau):
        """``int psi dt`` over a term of duration ``tau``: the moment the term's level leaves (``tau`` for a constant).
        A sum within the rounding of its own ``int |psi|`` (``cells`` additions at the float64 unit roundoff) is zero."""
        if self.is_constant:
            return float(tau)
        v = np.asarray(self.values, np.float64)
        area, w = float(tau) * float(v.mean()), float(tau) * float(np.abs(v).mean())
        return 0.0 if abs(area) <= v.size * np.finfo(np.float64).eps * w else area

    def edges(self, a, b):
        """The cell edges of the term ``[a, b]``, ``(cells + 1,)`` seconds."""
        if self.is_constant:
            return np.array([a, b], np.float64)
        e = a + (b - a) * np.arange(self.cells + 1) / self.cells
        e[-1] = b
        return e

    def running(self, t, a, b):
        """``int_a^t psi dt`` at the times ``t`` for the term ``[a, b]`` (0 before it, its area after): piecewise linear
        between the cell edges."""
        t = np.asarray(t, np.float64)
        if self.is_constant:
            return np.minimum(np.clip(t - a, 0.0, None), b - a)
        e = self.edges(a, b)
        return np.interp(t, e, np.concatenate([[0.0], np.cumsum(np.asarray(self.values) * np.diff(e))]))

    def on_grid(self, a, b, dt, n_t, where):
        """``(i0, i1, psi)`` of the term ``[a, b]`` on a sequence's grid: its sample range and ``psi`` held on each
        sample; a term or cell edge off the grid, or a cell of a fraction of a sample, is refused naming ``where``."""
        i0, i1 = a / dt, b / dt
        if abs(i0 - round(i0)) > 1e-6 or abs(i1 - round(i1)) > 1e-6 or round(i1) > n_t:
            raise ValueError(f"{where} [{a:.6g}, {b:.6g}] s is not on the sequence's grid (dt = {dt:.6g} s, {n_t} samples)")
        i0, i1 = int(round(i0)), int(round(i1))
        if self.is_constant:
            return i0, i1, np.ones(i1 - i0)
        per, rem = divmod(i1 - i0, self.cells)
        if rem or per < 1:
            raise ValueError(f"the {self.cells} cells of {where} [{a:.6g}, {b:.6g}] s are not on the sequence's grid "
                             f"(dt = {dt:.6g} s, {i1 - i0} samples): a cell is a whole number of samples")
        return i0, i1, np.repeat(np.asarray(self.values, np.float64), per)

    def to_meta(self):
        return dict(kind="constant") if self.is_constant else dict(kind="sampled", normalisation="max|psi| = 1",
                                                                    cells=self.cells, values=list(self.values))

    @classmethod
    def from_meta(cls, rec, where):
        """The shape of the manifest record ``rec`` of ``where`` (a term); a record that is missing, names no known kind,
        lacks a field its kind needs or is not normalised is refused naming the field."""
        if not isinstance(rec, dict) or "kind" not in rec:
            raise ValueError(f"{where} has no shape record ('shape': {{'kind': 'constant'}} or {{'kind': 'sampled', "
                             "'cells': n, 'values': [...]}): every term records its shape explicitly")
        if rec["kind"] == "constant":
            return cls.constant()
        if rec["kind"] != "sampled":
            raise ValueError(f"{where} has the shape kind {rec['kind']!r}; a shape is 'constant' or 'sampled'")
        for k in ("cells", "values"):
            if k not in rec:
                raise ValueError(f"the sampled shape of {where} has no {k!r} field")
        v = np.asarray(rec["values"], np.float64)
        if int(rec["cells"]) != v.size or v.size == 0 or abs(float(np.abs(v).max()) - 1.0) > 1e-12:
            raise ValueError(f"the sampled shape of {where} is not {int(rec['cells'])} normalised cell values "
                             "(max |psi| = 1)")
        return cls("sampled", tuple(v.tolist()))


@dataclass(frozen=True)
class Partition:
    """Terms of the walk's clock, ``segments = ((a, b), ...)`` seconds in order of their start, each with a
    :class:`Shape` ``shapes[s]`` (``psi_s`` on ``[a_s, b_s]``; a constant segment is ``psi = 1``) and ``gate`` the sign
    of the effective gate on it (+1, -1, or 0 while the magnetisation is stored, where the term carries no phase);
    ``template`` the sequence the gate was read from (its grid, RF schedule and readout are the tiers' gate). Terms
    may overlap in time (a tensor encoding is three shapes on one interval); a partition of disjoint constant
    segments is the common case. :meth:`of` reads one from a sequence's lobes, :meth:`from_sequence` shapes from a
    sequence's waveforms."""
    name: str
    segments: tuple
    gate: tuple
    shapes: tuple
    template: object = field(default=None, compare=False, repr=False)

    def __post_init__(self):
        if not (len(self.segments) == len(self.gate) == len(self.shapes)) or not all(isinstance(s, Shape) for s in self.shapes):
            raise ValueError(f"partition {self.name!r} is one Shape and one gate per segment")

    @classmethod
    def of(cls, name, sequence, segments=None):
        """The partition of ``sequence``: its lobes (the maximal runs of one non-zero effective gradient), or the stated
        ``segments`` (``[(a, b), ...]`` s, in time order and disjoint), each a constant term with the effective gate's
        sign on it; a segment the gate changes sign inside (a refocusing pulse) is refused by name."""
        seq = _sequence(sequence)
        if segments is None:
            segments = _lobes(np.asarray(seq.G_eff, np.float64), seq.dt)
        return cls._make(name, seq, segments, None, disjoint=True)

    @classmethod
    def from_sequence(cls, name, sequence, *, terms=None, rows=0):
        """The partition of ``sequence`` whose terms take the shapes of its waveforms: term ``s`` is the interval
        ``terms[s]`` (``(a, b)`` s on the sequence's grid; by default the maximal runs of row ``rows[0]``'s non-zero
        effective gradient within one effective-gate sign) and its shape the normalised waveform of row ``rows[s]``
        (``rows`` an int for every term) along its own direction on the interval, one value per sample --
        :meth:`Shape.of_samples`, so a plateau is a constant term. Terms may repeat an interval with different rows
        (a tensor encoding's three shapes), and their shapes must be linearly independent on the grid. A row that is
        not along one direction on its term is refused by name. The level of the term in a fitted sequence
        (:meth:`levels`) is the amplitude and direction that waveform is played at."""
        seq = _sequence(sequence)
        G = np.asarray(seq.G_eff, np.float64); dt = float(seq.dt)
        rws = [int(rows)] * (len(terms) if terms is not None else 1) if np.ndim(rows) == 0 else [int(r) for r in rows]
        if any(not 0 <= r < G.shape[0] for r in rws):
            raise ValueError(f"rows {rws} are not measurements of the sequence ({G.shape[0]} rows)")
        if terms is None:
            e = np.asarray(seq.effective_gate, np.float64)
            tol = _RTOL * (float(np.abs(G[rws[0]]).max()) or 1.0)
            on = np.abs(G[rws[0]]).max(axis=1) > tol
            terms, i = [], 0
            while i < G.shape[1]:
                if not on[i]:
                    i += 1; continue
                j = i + 1
                while j < G.shape[1] and on[j] and e[j] == e[i]:
                    j += 1
                terms.append((i * dt, j * dt)); i = j
            rws = rws[:1] * len(terms)
        terms = [(float(a), float(b)) for a, b in terms]
        if len(rws) != len(terms):
            raise ValueError(f"rows is one measurement per term: {len(rws)} rows for {len(terms)} terms")
        shapes = []
        for s, ((a, b), r) in enumerate(zip(terms, rws)):
            i0, i1, _ = Shape.constant().on_grid(a, b, dt, G.shape[1], f"term {s} of partition {name!r}")
            v = G[r, i0:i1, :]
            peak = float(np.abs(v).max())
            if peak == 0.0:
                raise ValueError(f"row {r} plays no gradient on term {s} [{a:.6g}, {b:.6g}] s of partition {name!r}")
            g_hat = v[np.linalg.norm(v, axis=1).argmax()]; g_hat = g_hat / np.linalg.norm(g_hat)
            psi = v @ g_hat
            if np.abs(v - psi[:, None] * g_hat[None, :]).max() > _RTOL * peak:
                raise ValueError(f"row {r} is not along one direction on term {s} [{a:.6g}, {b:.6g}] s of partition "
                                 f"{name!r}: a term's shape is a scalar waveform (give each axis its own term)")
            shapes.append(Shape.of_samples(psi))
        return cls._make(name, seq, terms, shapes, disjoint=False)

    @classmethod
    def _make(cls, name, seq, segments, shapes, *, disjoint):
        segs = tuple((float(a), float(b)) for a, b in segments)
        if not segs:
            raise ValueError(f"partition {name!r} has no segment: give segments= or a sequence with a gradient")
        shapes = tuple(Shape.constant() for _ in segs) if shapes is None else tuple(shapes)
        for s, (a, b) in enumerate(segs):
            if not 0.0 <= a < b:
                raise ValueError(f"segment {s} of partition {name!r} is [{a:.6g}, {b:.6g}] s: a segment is 0 <= a < b")
            if s and a < segs[s - 1][(1 if disjoint else 0)] - 1e-12:
                raise ValueError(f"the segments of partition {name!r} are in time order" + (" and disjoint" if disjoint else ""))
        if max(b for _, b in segs) > seq.n_t * seq.dt * (1.0 + 1e-9):
            raise ValueError(f"partition {name!r} ends at {max(b for _, b in segs):.6g} s, beyond its sequence's grid")
        e = np.asarray(seq.effective_gate, np.float64)
        gate = []
        for s, (a, b) in enumerate(segs):
            i0 = int(np.floor(a / seq.dt + 1e-6)); i1 = max(int(np.ceil(b / seq.dt - 1e-6)), i0 + 1)
            vals = np.unique(e[i0:i1])
            if len(vals) != 1:
                raise ValueError(f"the effective gate changes sign inside segment {s} [{a:.6g}, {b:.6g}] s of partition "
                                 f"{name!r} (a pulse acts there): split the segment at the pulse")
            gate.append(int(np.rint(vals[0])))
        out = cls(str(name), segs, tuple(gate), shapes, seq)
        if not disjoint:
            out.design(seq.dt, seq.n_t)
        return out

    @property
    def is_constant(self):
        return all(sh.is_constant for sh in self.shapes)

    @property
    def is_indicator(self):
        """Whether every term is a constant segment disjoint from the others: the sequences of one level per segment."""
        return self.is_constant and all(self.segments[s][0] >= self.segments[s - 1][1] - 1e-12 for s in range(1, len(self.segments)))

    @property
    def durations(self):
        return np.array([b - a for a, b in self.segments], np.float64)

    @property
    def areas(self):
        """``(n_term,)``: ``int psi_s dt``, the moment a unit level on term ``s`` leaves (its duration for a segment)."""
        return np.array([sh.area(b - a) for sh, (a, b) in zip(self.shapes, self.segments)], np.float64)

    @property
    def weights(self):
        """``(n_term,)``: ``int |psi_s| dt``, what bounds the magnitude of a unit level's moment (its duration for a
        segment)."""
        return np.array([sh.weight(b - a) for sh, (a, b) in zip(self.shapes, self.segments)], np.float64)

    @property
    def carrying(self):
        """Per segment whether its gate lets a phase accrue there (``gate != 0``)."""
        return np.array([g != 0 for g in self.gate], bool)

    def projection_shapes(self, idx=None):
        """The ``shapes`` argument of :func:`~dmipy_replay.replay.compression.segment_projection` for the terms ``idx``
        (all when None); ``None`` when every one is a segment."""
        sel = range(len(self.shapes)) if idx is None else idx
        vals = [self.shapes[s].projection_values() for s in sel]
        return None if all(v is None for v in vals) else vals

    def design(self, dt, n_t):
        """``(n_t, n_term)``: ``psi_s`` held on each sample of the grid (0 outside term ``s``), and ``covered`` ``(n_t,)``
        the samples inside some term; a term off the grid is refused by name, and terms whose shapes are linearly
        dependent on the grid (their levels would not be determined) likewise."""
        D = np.zeros((int(n_t), len(self.segments))); covered = np.zeros(int(n_t), bool)
        for s, ((a, b), sh) in enumerate(zip(self.segments, self.shapes)):
            i0, i1, psi = sh.on_grid(a, b, float(dt), int(n_t), f"term {s} of partition {self.name!r}")
            D[i0:i1, s] = psi; covered[i0:i1] = True
        if not self.is_indicator:
            sv = np.linalg.svd(D, compute_uv=False)
            if sv.size and sv[-1] <= sv[0] * max(D.shape) * np.finfo(np.float64).eps:
                raise ValueError(f"the shapes of partition {self.name!r} are linearly dependent on the sequence's grid: "
                                 "the levels of its terms are not determined by a gradient")
        return D, covered

    def b_matrix(self):
        """``(n_seg, n_seg)``: ``B_st = gamma^2 int Q_s Q_t dt`` with ``Q_s(t) = int_a^t psi_s`` the running integral of
        term ``s``, so that an effective encoding with levels ``l`` on the terms has ``b = l^T B l`` (s/m^2) when it is
        refocused; exact (each ``Q_s`` is piecewise linear between the cell edges)."""
        from ..constants import GAMMA
        pts = np.unique(np.concatenate([[0.0]] + [sh.edges(a, b) for sh, (a, b) in zip(self.shapes, self.segments)]))
        Q = np.stack([sh.running(pts, a, b) for sh, (a, b) in zip(self.shapes, self.segments)])
        L = np.diff(pts); qa, qb = Q[:, :-1], Q[:, 1:]
        B = (L[None, None, :] * (2.0 * qa[:, None] * qa[None] + qa[:, None] * qb[None] + qb[:, None] * qa[None]
                                 + 2.0 * qb[:, None] * qb[None]) / 6.0).sum(-1)
        return GAMMA ** 2 * B

    def refocused_basis(self):
        """``(idx, N)``: the terms that carry a phase and an orthonormal basis ``N`` ``(len(idx), len(idx) - 1)`` of
        the levels on them that leave no net moment (``sum_s l_s A_s = 0``, ``A_s`` the term's :attr:`areas`); terms of
        zero area (a whole number of periods of a cosine) each leave none and ``N`` is the identity."""
        idx = np.flatnonzero(self.carrying)
        a = self.areas[idx]
        if idx.size and not np.any(a):
            return idx, np.eye(idx.size)
        if idx.size < 2:
            return idx, np.zeros((idx.size, 0))
        _, _, Vt = np.linalg.svd(a[None, :])
        return idx, Vt[1:].T

    def class_vertices(self, G_max):
        """``(idx, V)``: the terms that carry a phase and the vertices ``V`` ``(n_v, len(idx))`` of the refocused
        levels a scanner class plays on one axis, ``{|l_s| <= G_max, sum_s l_s A_s = 0}`` (``A_s`` the :attr:`areas`):
        a box cut by a plane, whose vertices have every level but one at ``+-G_max`` (the box's own, when every area
        is zero). A convex function of the levels (a phase variance, a b) is largest over them at one of these."""
        idx = np.flatnonzero(self.carrying)
        d = self.areas[idx]
        n = idx.size
        G = float(G_max)
        out = []
        if n and not np.any(d):
            out = [G * np.asarray(signs) for signs in itertools.product((-1.0, 1.0), repeat=n)]
        elif n >= 2:
            for i in np.flatnonzero(d):
                rest = [j for j in range(n) if j != i]
                for signs in itertools.product((-1.0, 1.0), repeat=n - 1):
                    v = np.zeros(n)
                    v[rest] = G * np.asarray(signs)
                    v[i] = -float(d[rest] @ v[rest]) / float(d[i])
                    if abs(v[i]) <= G * (1.0 + 1e-12):
                        out.append(v)
        if not out:
            return idx, np.zeros((0, n))
        return idx, np.unique(np.round(np.asarray(out) / G, 12), axis=0) * G

    def reach(self, G_max, b_max=None):
        """``(n_seg,)``: the largest level ``|G_s|`` (T/m, the vector's) an encoding within the envelope puts on each
        segment (0 on a segment the gate stores): over a scanner class's refocused levels (``b_max`` None,
        :meth:`class_vertices`) ``sqrt(3)`` times the largest per-axis ``|l_s|``, every axis at its own, else
        ``min(G_max, sqrt(b_max [N (N^T B N)^-1 N^T]_ss))`` over those within ``b_max``."""
        out = np.zeros(len(self.segments))
        if b_max is None:
            idx, V = self.class_vertices(G_max)
            if V.shape[0]:
                out[idx] = np.sqrt(3.0) * np.abs(V).max(axis=0)
            return out
        idx, N = self.refocused_basis()
        if N.shape[1] == 0:
            return out
        M = N.T @ self.b_matrix()[np.ix_(idx, idx)] @ N
        P = N @ np.linalg.solve(M, N.T)
        out[idx] = np.minimum(float(G_max), np.sqrt(float(b_max) * np.clip(np.diag(P), 0.0, None)))
        return out

    def levels(self, sequence):
        """``(n_meas, n_seg, 3)``: each row's effective gradient on each term (T/m), the vectors ``a_s`` with
        ``G(t) = sum_s psi_s(t) a_s`` to :data:`_RTOL` of the peak.

        Disjoint constant segments take the gradient's value on them, which must be constant on a segment. Otherwise (a
        shape, or terms that overlap) the levels are the least squares fit of ``G(t)`` on the terms' shapes on the
        sequence's grid, and the residual is held to the same tolerance. A sequence refused by name: off the terms'
        grid, a gradient outside every term, off the shapes (the row, the term the residual is largest in and its
        relative norm named), on a term its gate stores, or leaving a net moment."""
        seq = _sequence(sequence)
        G = np.asarray(seq.G_eff, np.float64); dt = float(seq.dt)
        n_meas, n_t, _ = G.shape
        peak = float(np.abs(G).max()) or 1.0; tol = _RTOL * peak
        design, covered = self.design(dt, n_t)
        if np.abs(G[:, ~covered, :]).max(initial=0.0) > tol:
            raise ValueError(f"the sequence plays a gradient outside the segments of partition {self.name!r}: it does not "
                             "fit the partition")
        if self.is_indicator:
            L = np.zeros((n_meas, len(self.segments), 3))
            for s, (a, b) in enumerate(self.segments):
                i0, i1, _ = self.shapes[s].on_grid(a, b, dt, n_t, f"segment {s} of partition {self.name!r}")
                lev = G[:, i0, :]
                if np.abs(G[:, i0:i1, :] - lev[:, None, :]).max() > tol:
                    raise ValueError(f"the sequence's effective gradient is not constant on segment {s} [{a:.6g}, {b:.6g}] s "
                                     f"of partition {self.name!r}: it does not fit the partition")
                L[:, s] = lev
        else:
            X = np.moveaxis(G, 1, 0).reshape(n_t, -1)
            sol = np.linalg.lstsq(design, X, rcond=None)[0]
            L = np.moveaxis(sol.reshape(len(self.segments), n_meas, 3), 0, 1)
            R = np.moveaxis((X - design @ sol).reshape(n_t, n_meas, 3), 0, 1)
            if np.abs(R).max(initial=0.0) > tol:
                self._refuse_residual(R, G, design)
        if np.abs(L[:, ~self.carrying]).max(initial=0.0) > tol:
            raise ValueError(f"the sequence plays an effective gradient on a segment of partition {self.name!r} its gate "
                             "stores: the partition holds no vector there")
        net = np.einsum("msc,s->mc", L, self.areas)
        if np.abs(net).max(initial=0.0) > _RTOL * peak * float(self.weights.sum()):
            raise ValueError(f"the sequence leaves a net moment at the readout on partition {self.name!r} (an unbalanced "
                             "encoding): a segment vector is the path relative to the walker's start, which only a "
                             "refocused encoding does not see")
        return L

    def _refuse_residual(self, R, G, design):
        """Refuse a least squares fit whose residual ``R`` ``(n_meas, n_t, 3)`` of ``G`` is beyond the tolerance, naming
        the row with the largest residual, the term it lies in and its relative norm."""
        m = int(np.abs(R).max(axis=(1, 2)).argmax())
        energy = ((R[m] ** 2).sum(1)[:, None] * (design != 0)).sum(0)
        term = int(energy.argmax())
        rel = float(np.sqrt((R[m] ** 2).sum() / max((G[m] ** 2).sum(), 1e-300)))
        a, b = self.segments[term]
        raise ValueError(f"the sequence's effective gradient does not fit the shapes of partition {self.name!r}: row {m} "
                         f"leaves a residual of relative norm {rel:.3g} (largest on term {term} [{a:.6g}, {b:.6g}] s), "
                         f"above the {_RTOL:.0e} of the peak a sequence is held to")

    def to_meta(self):
        return dict(segments=[list(s) for s in self.segments], gate=list(self.gate), shapes=[sh.to_meta() for sh in self.shapes])


def _layout_of(rec):
    """``(layout, scale)`` of a partition's or a background's record: the fields' places and scales in field order."""
    if "segments" in rec:
        items = [(w, o, b, s) for seg in rec["segments"] for w, o, b, s in zip(seg["word"], seg["offsets"], seg["bits"], seg["scale"])]
    elif "fields" in rec:
        items = [(f["word"], f["offset"], f["bits"], f["scale"]) for f in rec["fields"]]
    else:
        items = list(zip(rec["word"], rec["offsets"], rec["bits"], rec["scale"]))
    return tuple((int(w), int(o) if b else 0, int(b)) if b else (0, 0, 0) for w, o, b, _ in items), \
        np.asarray([s for *_, s in items], np.float32)


def _field_of_tiers(pack_id, tiers, groups):
    """The susceptibility field of a tier layout: its embedded spec's, with the groups' field columns as the stored
    field channel (:func:`~dmipy_replay.replay.replay.declared_susceptibility_field`)."""
    from .replay import declared_susceptibility_field
    return declared_susceptibility_field(dict(id=pack_id, substrate=tiers.get("substrate")),
                                         has_field=any(grp.get("field") for grp in groups.values()))


class TimeSegmentLayout:
    """The time-segment layout at ``path``, tiles memory-mapped from disk: ``manifest``, ``grid``, ``partitions`` (the
    stored :class:`Partition` objects by name), and :meth:`image` -- a whole grid under any sequence whose effective gradient
    is a sum of the shapes of a stored partition's terms (piecewise constant on its segments, a trapezoid, a cosine, a
    tensor encoding), on the device. Open a Hub layout with :meth:`open`."""

    TILES = 4096                                                        # tiles per device chunk (2^19 rows)
    MEAS = 64                                                           # measurements padded to a multiple
    #: the backends :meth:`image` and :meth:`tier_maps` run on, by name, and the one ``backend=None`` names
    BACKENDS = None                                                     # set below: {"numpy": NumpyBackend()}
    DEFAULT_BACKEND = "numpy"

    def _backend(self, name):
        """The backend ``name`` names (:attr:`DEFAULT_BACKEND` for None); a name this layout has no backend for is refused
        by name, naming those it has."""
        name = self.DEFAULT_BACKEND if name is None else name
        if name not in self.BACKENDS:
            raise ValueError(f"backend is one of {sorted(self.BACKENDS)}, got {name!r}")
        return self.BACKENDS[name]

    def __init__(self, path):
        from ..phantom.grid import Grid
        self.path = path
        self.manifest = read_manifest(path)
        self.grid = Grid.from_meta(self.manifest["grid"])
        self.n_rows = int(self.manifest["n_rows"]); self.n_tiles = int(self.manifest["n_tiles"]); self.tile = int(self.manifest["tile"])
        self.n_vox = int(np.prod(self.grid.shape))
        self.partitions = {n: Partition(n, tuple(tuple(s["t"]) for s in p["segments"]), tuple(int(s["gate"]) for s in p["segments"]),
                                        tuple(Shape.from_meta(s.get("shape"), f"term {i} of partition {n!r} in the manifest")
                                              for i, s in enumerate(p["segments"])))
                           for n, p in self.manifest["partitions"].items()}
        self.tiers = self.manifest.get("tiers")
        self.susceptibility_field = (None if not self.tiers else
                                     _field_of_tiers((self.manifest["source"].get("pack") or {}).get("id"), self.tiers,
                                                     self.manifest["groups"]))
        self._m = {}; self._device = {}; self._host = {}

    @classmethod
    def open(cls, uri, *, partitions=None, revision="main", workers=8):
        """The layout at ``uri``: a directory, or ``hf://owner/name/prefix`` fetched into the cache at ``revision``
        (resolved to its commit once; :func:`~dmipy_replay.replay.hub.fetch_tree`): the manifest, the shared columns, the
        tiers and the words of ``partitions`` (every partition when None)."""
        from . import hub
        if not hub.is_hub_uri(uri):
            return cls(uri)
        repo, prefix = hub.parse_uri(uri)
        sha = hub.revision_of(repo, revision)
        root = hub.fetch_tree(repo, [f"{prefix}/{MANIFEST}"], revision=sha)
        with open(os.path.join(root, prefix, MANIFEST)) as f:
            manifest = json.load(f)
        _check_format(manifest, uri)
        hub.fetch_tree(repo, [f"{prefix}/{f}" for f in cls.files(manifest, partitions)], revision=sha, workers=workers)
        return cls(os.path.join(root, prefix))

    @staticmethod
    def columns(manifest):
        """Every column of a layout by name -> file: ``w``, ``tiles``, ``p_<partition>``, with tiers ``pool`` and
        ``t_<g>`` per sequence group, with the background ``bg_<g>``."""
        cols = dict(manifest["columns"])
        cols.update({f"p_{n}": p["column"] for n, p in manifest["partitions"].items() if p.get("column")})
        if manifest.get("tiers"):
            cols["pool"] = manifest["tiers"]["pool_column"]
        for g, grp in manifest["groups"].items():
            if grp.get("column"):
                cols[f"t_{g}"] = grp["column"]
            if (grp.get("background") or {}).get("column"):
                cols[f"bg_{g}"] = grp["background"]["column"]
        return cols

    @staticmethod
    def files(manifest, partitions=None):
        """The files a layout needs for ``partitions`` (every partition when None): the manifest, the shared columns, the
        tier and background columns and the words of those partitions."""
        keep = set(manifest["partitions"]) if partitions is None else set(partitions)
        return [MANIFEST] + [f for name, f in TimeSegmentLayout.columns(manifest).items()
                             if not name.startswith("p_") or name[2:] in keep]

    def _column(self, name):
        if name not in self._m:
            self._m[name] = np.load(os.path.join(self.path, self.columns(self.manifest)[name]), mmap_mode="r")
        return self._m[name]

    def _group(self, partition):
        """``(g, group record)`` of ``partition``'s sequence group."""
        g = self.manifest["partitions"][partition]["group"]
        return g, self.manifest["groups"][g]

    def _code(self, partition):
        """``(scale, layout, n_words)`` of a partition's words."""
        rec = self.manifest["partitions"][partition]
        layout, scale = _layout_of(rec)
        return scale, layout, int(rec["n_words"])

    def _tier_code(self, g):
        """``(contact field's (layout, scale) or None, channel fields)`` of group ``g``'s tier words."""
        tw = self.manifest["groups"][g].get("tier_words")
        if tw is None:
            return None
        layout, scale = _layout_of(tw)
        return layout, scale, [f["name"] for f in tw["fields"]]

    def fit(self, sequence, partition=None):
        """``(partition, levels)``: the stored partition ``sequence`` fits (``partition`` names one, else the first that
        fits among those whose group plays the sequence's gate, then among the rest) and its rows' effective gradients
        on the segments ``(n_meas, n_seg, 3)``; refused by name when none fits."""
        if partition is None:
            key = _group_key(_sequence(sequence))
            own = [n for n in self.partitions if self._gate_key(n) == key]
            names = own + [n for n in self.partitions if n not in own]
        else:
            names = [partition]
        reasons = []
        for name in names:
            if name not in self.partitions:
                raise KeyError(f"no partition {name!r} in this layout; it holds {list(self.partitions)}")
            try:
                return name, self.partitions[name].levels(sequence)
            except ValueError as e:
                reasons.append(f"{name}: {e}")
        raise ValueError("the sequence fits no stored partition -- " + "; ".join(reasons))

    def _certified(self, partition, L):
        """Refuse levels beyond the envelope the partition's widths are derived at: a segment above ``G_max`` or an
        encoding above ``b_max``; for a scanner class's layout (``b_max`` None) a level above ``G_max`` on any axis."""
        env = self.manifest["source"]["envelope"]
        B = self.partitions[partition].b_matrix()
        b = np.einsum("msc,st,mtc->m", L, B, L)
        per_axis = env.get("b_max") is None
        g = (np.abs(L).max(initial=0.0) if per_axis else np.linalg.norm(L, axis=2).max(initial=0.0))
        if g > env["G_max"] * (1 + 1e-6):
            raise ValueError(f"the sequence plays {g:.4g} T/m{' on one axis' if per_axis else ''} on partition {partition!r} "
                             f"and its vectors are derived for at most G_max = {env['G_max']:.4g} T/m")
        if b.size and env.get("b_max") is not None and b.max() > env["b_max"] * (1 + 1e-6):
            raise ValueError(f"the sequence reaches b = {b.max():.4g} s/m^2 on partition {partition!r} and its vectors are "
                             f"derived for at most b_max = {env['b_max']:.4g} s/m^2 (the source's declared envelope)")
        return b

    def vectors(self, partition):
        """``(n_tiles, tile, n_seg, 3)`` float32: every row's segment vectors (0 on padding), decoded from the words."""
        scale, layout, _ = self._code(partition)
        n_seg = len(self.partitions[partition].segments)
        words = (self._column(f"p_{partition}") if f"p_{partition}" in self.columns(self.manifest)
                 else np.zeros((self.n_tiles, self.tile, 1), np.uint32))
        v = unpack_words(words, scale, layout)
        return v.reshape(v.shape[:2] + (n_seg, 3))

    def tier_columns(self, g):
        """``{name: (n_tiles, tile) float32}``: group ``g``'s decoded tier fields (the contact and the stored field
        channels)."""
        code = self._tier_code(g)
        if code is None:
            return {}
        layout, scale, names = code
        v = unpack_words(self._column(f"t_{g}"), scale, layout)
        return {nm: v[..., i] for i, nm in enumerate(names)}

    def terms(self, partition, tissue=None, scanner=None):
        """The reduction terms of a (tissue, scanner) pair on ``partition``'s gate -- exactly
        :meth:`~dmipy_replay.replay.study.Primitives.reduction_terms` on a descriptor of what the layout carries: ``None``
        for the bare image (no tissue, no scanner), else ``(logw_pool (n_pools,), rho2_over_D, a_iso, a_aniso,
        amplitude)``. A tier the pair needs that the layout lacks is refused by name, and a relaxivity beyond the
        ``rho2 / D`` the contact's widths are derived at (the source's declared envelope) likewise."""
        if tissue is None and scanner is None:
            return None
        if not self.tiers:
            raise ValueError("this layout holds the bare diffusion phase only (written without tiers=True); a tissue "
                             "or a scanner needs the tiers")
        from .primitives import Primitives
        from ..spec.substrate import SubstrateSpec
        from ..spec.tissue import _by_pool_id
        t = self.tiers; g, grp = self._group(partition)
        env = self.manifest["source"]["envelope"]
        spec = SubstrateSpec.from_dict(t["substrate"]) if t.get("substrate") else None

        def by_pool(values, what):
            if values is None:
                return None
            if spec is None:
                raise ValueError(f"{what} was given by pool but the layout embeds no substrate spec")
            if not isinstance(values, dict):
                raise ValueError(f"{what} is {{pool name: seconds}} over every pool of the spec {[p.name for p in spec.pools]}")
            return [float(v) for v in _by_pool_id(spec, values, what)]
        n_pools = int(t["pools"]) if t.get("relaxation") else None
        has_field = bool(grp.get("field"))
        desc = Primitives(w=np.ones(1), phi=np.zeros((1, 1)),
                          field_iso=np.zeros(1) if has_field else None, field_aniso=np.zeros(1) if has_field else None,
                          exposure_t2=np.zeros((1, n_pools)) if n_pools else None, exposure_t1=np.zeros((1, n_pools)) if n_pools else None,
                          contact=np.zeros(1) if grp.get("contact") else None, D_walk=t.get("D_walk"), voxel=np.ones(1),
                          pathway=float(grp["pathway"]), by_pool=by_pool, susceptibility_field=self.susceptibility_field,
                          rho2_over_D_max=env["rho2_over_D_max"],
                          field_envelope=(dict(B0_max=env["B0_max_T"], B0_chi_max=env["B0_chi_max_T"]) if has_field else None))
        rt = desc.reduction_terms(tissue, scanner)
        logw = -(float(grp["tau_t2"] or 0.0) * np.asarray(rt["invT2"], np.float64) + float(grp["tau_t1"] or 0.0) * np.asarray(rt["invT1"], np.float64))
        return (logw, float(rt["rho2_over_D"]), float(rt["a_iso"]), float(rt["a_aniso"]), float(rt["amplitude"]))

    def _gate_key(self, partition):
        """The gate ``partition``'s group was read under, spelled as :func:`_group_key` spells a sequence's."""
        _, grp = self._group(partition)
        return json.dumps(dict(dt=float(grp["dt"]), n_t=int(grp["n_t"]), rf=grp["rf"], readout=grp["readout"]), sort_keys=True, default=float)

    def _check_gate(self, partition, sequence):
        """Refuse a tiered image of a sequence whose gate (grid, RF schedule, readout) is not the one ``partition``'s
        tiers were read under."""
        g, grp = self._group(partition)
        if _group_key(_sequence(sequence)) != self._gate_key(partition):
            raise ValueError(f"the tiers of partition {partition!r} were read under group {g}'s gate (its grid, RF schedule "
                             "and readout), and this sequence plays another")

    @functools.cached_property
    def weights(self):
        """The per-voxel weight sums ``(n_vox, 2)`` (the two split halves), the denominator of every image (read once)."""
        w = np.asarray(self._column("w"), np.float64).sum(1); tiles = np.asarray(self._column("tiles"))
        return np.bincount(tiles, w, minlength=2 * self.n_vox).reshape(self.n_vox, 2)

    @functools.cached_property
    def live_voxels(self):
        """The flat indices of the voxels the layout has rows in, in order: the voxels a delivery is computed at."""
        return np.flatnonzero(self.weights.sum(1) > 0)

    # ---- the field direction's contraction
    def _field_spec(self, g, b0_direction, a_iso, a_aniso):
        """What the kernel needs to form group ``g``'s field phase from its tier words for ``b0_direction`` (a unit vector
        in the substrate frame), or ``None`` when the group stores no field or the pair's field scales are both zero:
        ``(layout, scale, coef, n_iso)``, the stored channels of the basis's twelve (RPK.md 5.3: ``iso_local``,
        ``iso_P`` xx, yy, xy, xz, yz, ``aniso_G`` (6)) in that order, their places in the words, their scales and their
        contraction weights in float64 -- :func:`~dmipy_replay.fields.hollow_cylinder.iso_weights` on the six isotropic
        channels (``iso_P_zz`` folded in by the trace identity, never formed), :func:`~dmipy_replay.fields.hollow_cylinder.q_of_H`
        on the six anisotropic ones -- and how many of them are isotropic. A channel that rounds to zero on every row
        (not stored) contributes nothing."""
        from .field import iso_weights, q_of_H
        b0 = np.asarray(b0_direction, np.float64)
        if b0.shape != (3,) or abs(np.linalg.norm(b0) - 1.0) > 1e-6:
            raise ValueError("b0_direction is a unit vector")
        f = self.manifest["groups"][g].get("field")
        if not f or not f.get("stored") or (float(a_iso) == 0.0 and float(a_aniso) == 0.0):
            return None
        layout, scale, names = self._tier_code(g)
        weight = dict(zip(f["channels"][:6], iso_weights(b0)))
        weight.update(zip(f["channels"][6:12], q_of_H(b0)))
        iso = [nm for nm in f["channels"][:6] if nm in names]
        aniso = [nm for nm in f["channels"][6:12] if nm in names]
        order = iso + aniso
        return (tuple(layout[names.index(nm)] for nm in order),
                np.asarray([scale[names.index(nm)] for nm in order], np.float64),
                np.asarray([weight[nm] for nm in order], np.float64), len(iso))

    def tier_names(self, partition):
        """The tier columns of ``partition``'s group the kernel reads: the pool and the group's tier words (the contact
        and the field channels, decoded in the kernel)."""
        g, _ = self._group(partition)
        return ("pool", f"t_{g}")

    def release(self, keep=()):
        """Drop the device copies of every partition's words but ``keep``'s (the shared columns stay), on every backend."""
        for name in [n for n in self._device if n[1].startswith("p_") and n[1][2:] not in keep]:
            del self._device[name]

    def preload(self, partitions=None, background=True):
        """The padded host arrays of ``partitions``' words (every partition when None), the shared columns and the tiers
        kept in this process's memory, so that a device copy built later -- in a forked worker of a shared pool, which
        inherits them for free -- costs a host-to-device transfer and not a read of the layout; with ``background`` each
        partition's group's background words too, where the layout carries them."""
        parts = list(self.partitions if partitions is None else partitions)
        names = ["w", "tiles"] + [f"p_{p}" for p in parts]
        if self.tiers:
            for p in parts:
                names += list(self.tier_names(p))
        if background:
            names += [f"bg_{self._group(p)[0]}" for p in parts if self.manifest["groups"][self._group(p)[0]].get("background")]
        for name in dict.fromkeys(names):
            if name not in self._host:
                self._host[name] = self._padded(name)

    @functools.cached_property
    def _compact(self):
        """``(n_vox + 1,)`` int32: each voxel's index into :attr:`live_voxels` (the dump slot ``len(live)`` for a voxel
        without rows): a tile's slot in a delivery."""
        live = self.live_voxels
        compact = np.full(self.n_vox + 1, len(live), np.int32); compact[live] = np.arange(len(live), dtype=np.int32)
        return compact

    @property
    def n_pad(self):
        """The tiles padded to whole chunks of :attr:`TILES`."""
        return -(-self.n_tiles // self.TILES) * self.TILES

    def _host_chunk(self, name, s, e):
        """Tiles ``s:e`` of the column ``name`` on the host, padded past the layout's last tile (padding rows weigh
        nothing and scatter into the dump slot), words as their int32 view: a slice of the preloaded padded copy when
        there is one, else read from the memory-mapped column. ``slot`` is each tile's index into a delivery's live
        voxels."""
        if name in self._host:
            return self._host[name][s:e]
        n, real = e - s, max(0, min(e, self.n_tiles) - s)
        if name == "slot":
            a = np.full(n, len(self.live_voxels), np.int32); a[:real] = self._compact[np.asarray(self._column("tiles")[s:s + real]) // 2]
        elif name == "tiles":
            a = np.full(n, 2 * self.n_vox, np.int32); a[:real] = self._column("tiles")[s:s + real]
        elif name in ("w", "pool"):
            a = np.zeros((n, self.tile), np.uint8 if name == "pool" else np.float32); a[:real] = self._column(name)[s:s + real]
        elif name not in self.columns(self.manifest):           # a word array with no words (every field zero): one zero word
            a = np.zeros((n, self.tile, 1), np.int32)
        else:
            words = self._column(name)
            a = np.zeros((n,) + tuple(words.shape[1:]), np.uint32); a[:real] = words[s:s + real]
            a = a.view(np.int32)
        return a

    def _padded(self, name):
        """The column ``name`` on the host, padded to whole chunks of :attr:`TILES` (:meth:`_host_chunk`)."""
        return self._host[name] if name in self._host else self._host_chunk(name, 0, self.n_pad)

    def _resident(self, backend, device, names):
        """The columns ``names`` whole on the backend's device, kept across calls."""
        put = backend.put(device)
        out = []
        for name in names:
            key = (backend.name, name, str(device))
            if key not in self._device:
                self._device[key] = put(self._padded(name), name)
            out.append(self._device[key])
        return tuple(out)

    def _chunks(self, backend, device, resident, names):
        """The columns ``names`` chunk by chunk of :attr:`TILES` tiles, a tuple of device arrays per chunk: slices of
        the device-resident columns (``resident``, :meth:`_resident`), or each chunk's tiles read from the layout (or
        the preloaded host copy), padded and uploaded on its own, so that one chunk of every column is on the device
        at a time and the words are decoded in the kernel step that reads them."""
        T = self.TILES
        if resident:
            cols = self._resident(backend, device, names)
            for s in range(0, self.n_pad, T):
                yield tuple(c[s:s + T] for c in cols)
            return
        put = backend.put(device)
        for s in range(0, self.n_pad, T):
            yield tuple(put(self._host_chunk(name, s, s + T), name) for name in names)

    def background_group(self, partition):
        """The sequence group whose background words ``partition`` reads, refused for a layout without them."""
        g, grp = self._group(partition)
        if not grp.get("background"):
            raise ValueError("this layout holds no background vectors: a magnet's own gradient needs them "
                             "(segment_rows(..., background=True))")
        return g

    def tier_maps(self, partition, tissue=None, scanner=None, *, backend=None, device=None, resident=True,
                  b0_direction=(0.0, 0.0, 1.0)):
        """Per-voxel means over the walkers, at the walk's weights, of what each tier multiplies into :meth:`image` on
        ``partition``'s gate, each on the grid (NaN where the layout has no rows): ``pool`` (the weight fraction per
        pool, ``{name: grid}``), ``contact`` (the gated contact term; None without the contact tier),
        ``contact_weight`` (``exp(rho2_over_D contact)`` at the pair's rho2; None without a tissue), ``phase`` and
        ``phase_std`` (the mean and the spread of the field's phase offset at the pair's field and ``b0_direction``;
        None without a scanner). The tier words are decoded chunk by chunk on the device, as in :meth:`image`
        (``resident`` as there)."""
        be = self._backend(backend)
        if not self.tiers:
            raise ValueError("this layout holds the bare diffusion phase only (written without tiers=True)")
        tm = self.terms(partition, tissue, scanner)
        g, grp = self._group(partition)
        pools = [(q["name"], int(q["id"])) for q in self.tiers["substrate"]["pools"]]
        device = be.device(device)
        rho_D, a_iso, a_aniso = (0.0, 0.0, 0.0) if tm is None else (float(tm[1]), float(tm[2]), float(tm[3]))
        fspec = self._field_spec(g, b0_direction, a_iso, a_aniso) if tm is not None else None
        code = self._tier_code(g)
        names = [("pool", name, pid) for name, pid in pools]
        if grp.get("contact"):
            names.append(("contact",))
            if tm is not None:
                names.append(("contact_weight",))
        if grp.get("field") and tm is not None:
            names += [("phase",), ("phase_sq",)]
        on_torch = be.torch

        def rows(w, pool, tw, xp):
            out = []
            c = 0.0 * w
            if code is not None and "contact" in code[2]:
                layout, scale, fnames = code
                sc = xp.asarray(scale) if not on_torch else xp.as_tensor(scale, device=tw.device)
                c = _decode(tw, sc, layout)[fnames.index("contact")]
            ph = 0.0 * w
            if fspec is not None:
                ph = _field_offset(tw, fspec, a_iso, a_aniso, w, on_torch)
            for q in names:
                if q[0] == "pool":
                    out.append((pool == q[2]).to(w.dtype) if on_torch else (pool == q[2]).astype(w.dtype))
                elif q[0] == "contact":
                    out.append(c)
                elif q[0] == "contact_weight":
                    out.append(xp.exp(rho_D * c))
                else:
                    out.append(ph if q[0] == "phase" else ph * ph)
            return out

        n_halves = 2 * self.n_vox + 1
        chunks = self._chunks(be, device, resident, ("w", "tiles") + self.tier_names(partition))
        with be.context():
            if fspec is not None:
                fspec = be.field_spec(fspec, device)
            acc = be.voxel_sums(rows, len(names), chunks, n_halves)
        num = acc[:, :2 * self.n_vox].reshape(len(names), self.n_vox, 2).sum(2)
        den = self.weights.sum(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            means = np.where(den > 0, num / den, np.nan).reshape((len(names),) + tuple(self.grid.shape))
        by = {q[0] if q[0] != "pool" else ("pool", q[1]): m for q, m in zip(names, means)}
        maps = dict(pool={name: by[("pool", name)] for name, _ in pools}, contact=by.get("contact"), contact_weight=by.get("contact_weight"),
                    phase=by.get("phase"), phase_std=None)
        if "phase" in by:
            maps["phase_std"] = np.sqrt(np.maximum(by["phase_sq"] - by["phase"] ** 2, 0.0))
        return maps


    def image(self, sequence, *, partition=None, backend=None, device=None, resident=True, tissue=None, scanner=None,
              b0_direction=(0.0, 0.0, 1.0), delivered=None):
        """``(S, floor)`` of the grid under ``sequence`` (a :class:`~dmipy_replay.acquisition.scanner_sequence.ScannerSequence`
        whose rows are sums of the shapes of a stored partition's terms; ``partition`` names one, else the first it fits): ``S``
        ``grid.shape + (n_meas,)``, the weighted ensemble magnitude per voxel (NaN where the layout has no rows),
        ``floor`` ``grid.shape`` its split-half floor ``max_m |S_a - S_b| / 2``. A row beyond the envelope the widths are
        derived at (``G_max``, ``b_max``) is refused by name. ``backend`` names the arithmetic's backend
        (:attr:`BACKENDS`; :attr:`DEFAULT_BACKEND` when None) and ``device`` its device. The packed words -- the partition's, the
        tiers', the background's -- reach the device as they are stored and are decoded in the kernel step that reads
        them, a chunk of :attr:`TILES` tiles at a time; ``resident`` keeps every column the image reads whole on the
        device across calls, ``resident=False`` reads and uploads each chunk on its own (one chunk of each column on the
        device at a time, nothing kept), to the same bits. ``tissue`` and ``scanner`` turn the stored tiers into the weights and the field phase of
        :meth:`terms` (the sequence must play the partition's group's gate); both None is the bare image, scaled by the
        sequence's pathway amplitude. ``b0_direction`` is the field's direction in the substrate frame.

        ``delivered`` (:meth:`delivery`, for this sequence) is what a machine plays at each voxel in place of the
        commanded gradient: each row's phase is ``sum_s q_vs . m_s + g0_v . n`` with ``q_vs`` the voxel's delivered
        vector on segment ``s``, ``n`` the row's background vector, and each voxel's signal is multiplied by its pathway
        amplitude at its transmit scale over the nominal one
        (:func:`~dmipy_replay.acquisition.epg.transmit_amplitude`)."""
        from ..acquisition.epg import pathway_weight
        be = self._backend(backend)
        name, L = self.fit(sequence, partition)
        self._certified(name, L)
        tm = self.terms(name, tissue, scanner)
        if tm is not None:
            self._check_gate(name, sequence)
        scale, layout, _ = self._code(name)
        n_seg = len(self.partitions[name].segments)
        M = L.shape[0]; M_pad = -(-M // self.MEAS) * self.MEAS
        n_halves = 2 * self.n_vox + 1
        device = be.device(device)
        g, grp = self._group(name)
        tier_names = self.tier_names(name) if tm is not None else ()
        tier = None
        if tm is not None:
            contact = None
            t_code = self._tier_code(g)
            if t_code is not None and "contact" in t_code[2]:
                t_layout, t_scale, t_names = t_code
                i = t_names.index("contact")
                contact = ((t_layout[i],), t_scale[i:i + 1])
            fspec = self._field_spec(g, b0_direction, tm[2], tm[3])
            tier = (contact, fspec, np.asarray(tm[0], np.float32), float(tm[1]), float(tm[2]), float(tm[3]), float(tm[4]))
        else:
            self._field_spec(g, b0_direction, 0.0, 0.0)             # the direction checked whatever the tiers
        factor = None; bg = None
        names = (f"p_{name}", "w", "tiles")
        if delivered is not None:
            live = self._check_delivered(delivered, name, L)
            names += ("slot",)
            Q = np.zeros((len(live) + 1, M_pad, n_seg, 3), np.float32); Q[:len(live), :M] = delivered.q
            if delivered.g0 is not None:
                gb = self.background_group(name)
                b_layout, b_scale = _layout_of(self.manifest["groups"][gb]["background"])
                G0 = np.zeros((len(live) + 1, 3), np.float32); G0[:len(live)] = delivered.g0
                bg = (b_layout, b_scale, G0)
            if delivered.kappa is not None:
                from ..acquisition.epg import transmit_amplitude
                factor = np.ones(self.n_vox)
                factor[live] = transmit_amplitude(_sequence(sequence), delivered.kappa) / transmit_amplitude(_sequence(sequence), [1.0])[0]
            levels = Q
        else:
            levels = np.zeros((M_pad, n_seg, 3), np.float32); levels[:M] = L
        names += tuple(tier_names) + ((f"bg_{gb}",) if bg is not None else ())
        chunks = self._chunks(be, device, resident, names)
        with be.context():
            acc = be.sums(chunks, scale, layout, levels, delivered is not None, tier, bg, n_halves, device)
        num = acc[:2 * self.n_vox, :M].reshape(self.n_vox, 2, M)
        den = self.weights
        S = np.full((self.n_vox, M), np.nan); floor = np.full(self.n_vox, np.nan)
        any_ = den.sum(1) > 0; both = (den > 0).all(1)
        pw = float(pathway_weight(_sequence(sequence))) if tm is None else 1.0   # the tiered path applied it as amp
        pv = pw if factor is None else pw * factor
        S[any_] = (pv if np.ndim(pv) == 0 else pv[any_, None]) * np.abs(num[any_].sum(1) / den[any_].sum(1)[:, None])
        pb = pv if np.ndim(pv) == 0 else pv[both, None]
        Sa = pb * np.abs(num[both, 0] / den[both, 0][:, None]); Sb = pb * np.abs(num[both, 1] / den[both, 1][:, None])
        floor[both] = 0.5 * np.abs(Sa - Sb).max(1)
        return S.reshape(tuple(self.grid.shape) + (M,)), floor.reshape(self.grid.shape)

    def _check_delivered(self, delivered, partition, L):
        """The live voxels, once ``delivered`` is checked to be this layout's, this partition's and these levels'."""
        live = self.live_voxels
        if delivered.partition != partition:
            raise ValueError(f"the delivery was computed for the partition {delivered.partition!r}, not {partition!r}")
        if not np.array_equal(np.asarray(delivered.voxels), live):
            raise ValueError("the delivery was computed for other voxels than the layout's")
        if delivered.q.shape != (len(live),) + L.shape or not np.allclose(delivered.levels, L, rtol=1e-9, atol=0.0):
            raise ValueError("the delivery was computed for another sequence than the image's")
        if delivered.g0 is not None:
            amax = float(self.manifest["source"]["background_amplitude"])
            if float(np.linalg.norm(delivered.g0, axis=1).max(initial=0.0)) > amax * (1 + 1e-9):
                raise ValueError(f"the magnet's gradient reaches {np.linalg.norm(delivered.g0, axis=1).max():.3g} T/m and the "
                                 f"layout's background vectors are certified to {amax:g} T/m")
        return live


@dataclass(frozen=True)
class Delivered:
    """:meth:`TimeSegmentLayout.delivery`: what a machine plays at each of a layout's voxels with rows (``voxels``, flat
    indices) for a sequence on ``partition`` whose commanded effective levels are ``levels`` ``(n_meas, n_seg, 3)``:
    the delivered effective vector on every segment ``q`` ``(n_voxels, n_meas, n_seg, 3)``, the magnet's own gradient
    ``g0`` ``(n_voxels, 3)`` (None without one) and the transmit scale ``kappa`` ``(n_voxels,)`` (None without one)
    (:class:`~dmipy_replay.phantom.bore.VectorDelivery`)."""
    partition: str
    voxels: np.ndarray
    levels: np.ndarray
    q: np.ndarray
    g0: np.ndarray = None
    kappa: np.ndarray = None


def _field_terms_of_words(tw, fspec, on_torch):
    """``(iso, aniso)`` float64 of every row from its int32 tier words ``tw`` ``(chunk, tile, n_words)``: each stored
    channel's integer shifted out of its word, made float64 and times its scale (exact: a code of at most 24 bits
    times a float32 scale), then weighted by the direction's contraction coefficients and summed in channel order, the
    products written out (a float32 matmul on a GPU may round its inputs to TF32); ``None`` for a term with no stored channel."""
    layout, scale, coef, n_iso = fspec
    iso = aniso = None
    for j, (w, off, b) in enumerate(layout):
        x = tw[..., w]
        k = (x << (WORD_BITS - off - b)) >> (WORD_BITS - b)
        if on_torch:
            import torch
            c = k.to(torch.float64) * scale[j]
        else:
            c = k.astype(np.float64) * scale[j]
        t = c * coef[j]
        if j < n_iso:
            iso = t if iso is None else iso + t
        else:
            aniso = t if aniso is None else aniso + t
    return iso, aniso


def _field_offset(tw, fspec, a_iso, a_aniso, w, on_torch):
    """The field's phase offset of every row, float32: ``a_iso iso + a_aniso aniso`` with the terms of
    :func:`_field_terms_of_words` rounded to float32 first."""
    iso, aniso = _field_terms_of_words(tw, fspec, on_torch)
    f32 = w.dtype
    fi = iso.to(f32) if on_torch and iso is not None else (iso.astype(f32) if iso is not None else 0.0 * w)
    fa = aniso.to(f32) if on_torch and aniso is not None else (aniso.astype(f32) if aniso is not None else 0.0 * w)
    return a_iso * fi + a_aniso * fa


def _phase(comps, levels, per_tile):
    """``sum_{s,c} m_sc L_sc`` ``(chunk, tile, M_pad)`` from the decoded components ``comps`` (segment-major, ``None``
    for a 0-bit field) and the levels: ``(M_pad, n_seg, 3)``, or with ``per_tile`` ``(chunk, M_pad, n_seg, 3)`` (a
    delivery's per voxel); the products written out (a float32 matmul on a GPU may round its inputs to TF32)."""
    ph = None
    for f, c in enumerate(comps):
        if c is None:
            continue
        s, ax = divmod(f, 3)
        term = c[:, :, None] * (levels[:, None, :, s, ax] if per_tile else levels[None, None, :, s, ax])
        ph = term if ph is None else ph + term
    return ph


def _row_terms(ph, w, pool, tw, nb, slot, tier, bg, on_torch):
    """The phase and the weight of each row with the background (``g0_v . n``, the voxel's ``g0`` by ``slot``, ``n``
    decoded from the background words ``nb``) and the tiers (the weight ``w amp exp(logw_pool[pool] + rho_D c)`` with
    the contact ``c`` decoded from the tier words ``tw``, the field offset :func:`_field_offset` from the same words)
    folded in, on numpy or JAX arrays (their own array namespace) or (``on_torch``) torch tensors."""
    if bg is not None:
        b_layout, b_scale, G0 = bg
        g0 = G0[slot]
        off = None
        for c, v in enumerate(_decode(nb, b_scale, b_layout)):
            if v is not None:
                t = v * g0[:, None, c]
                off = t if off is None else off + t
        if off is not None:
            ph = ph + off[:, :, None]
    if tier is None:
        return ph, w
    contact, fspec, logw_pool, rho_D, a_iso, a_aniso, amp = tier
    if fspec is not None:
        ph = ph + _field_offset(tw, fspec, a_iso, a_aniso, w, on_torch)[:, :, None]
    if on_torch:
        import torch
        lw, exp = logw_pool[pool.long()], torch.exp
    else:
        lw, exp = logw_pool[pool.astype(np.int32)], w.__array_namespace__().exp
    if contact is not None:
        lw = lw + rho_D * _decode(tw, contact[1], contact[0])[0]
    return ph, w * amp * exp(lw)


def _sums_numpy(chunks, scale, layout, levels, per_tile, tier, bg, n_halves, device=None):
    """The per-voxel-half sums ``(n_halves, M_pad)`` complex128 on numpy, chunk by chunk of :meth:`TimeSegmentLayout._chunks`
    (the partition's words, the weights, the tiles, then the slot with a delivery, the pool and the tier words with the
    tiers, the background words with a background): the words decoded, the phase in float32 with the tiers and the
    background folded in, ``w exp(i ph)`` in complex64 summed over each tile's rows, the tile sums added into their
    voxel halves in complex128."""
    M = levels.shape[-3]
    acc = np.zeros((n_halves, M), np.complex128)
    lev = np.asarray(levels, np.float32)
    sc = np.asarray(scale, np.float32)
    t = None
    if tier is not None:
        ct, fspec, lw, rho_D, a_iso, a_aniso, amp = tier
        t = (ct, fspec, np.asarray(lw, np.float32), np.float32(rho_D), np.float32(a_iso), np.float32(a_aniso), np.float32(amp))
    for cols in chunks:
        words, w, tiles = cols[:3]; k = 3
        slot = cols[k] if per_tile else None
        k += 1 if per_tile else 0
        pool = tw = nb = None
        if tier is not None:
            pool, tw = cols[k:k + 2]; k += 2
        if bg is not None:
            nb = cols[k]
        comps = _decode(words, sc, layout)
        ph = _phase(comps, lev[slot] if per_tile else lev, per_tile)
        if ph is None:
            ph = np.zeros(words.shape[:2] + (M,), np.float32)
        ph, wf = _row_terms(ph, w, pool, tw, nb, slot, t, bg, False)
        E = np.exp(np.complex64(1j) * ph.astype(np.complex64)) * wf[:, :, None].astype(np.float32)
        np.add.at(acc, np.asarray(tiles), E.sum(1).astype(np.complex128))
    return acc


def _voxel_sums_numpy(rows, n_q, chunks, n_halves):
    """The weighted per-voxel-half sums ``(n_q, n_halves)`` float64 of the ``n_q`` per-row quantities ``rows(w, pool, tw,
    numpy)`` gives for each chunk ``(w, tiles, pool, tw)`` of ``chunks``."""
    acc = np.zeros((n_q, n_halves), np.float64)
    for w, tiles, pool, tw in chunks:
        ts = np.stack([(q * w).sum(1) for q in rows(w, pool, tw, np)]).astype(np.float64)
        for i in range(n_q):
            np.add.at(acc[i], np.asarray(tiles), ts[i])
    return acc


class NumpyBackend:
    """The image's arithmetic on numpy, on the host: what :meth:`TimeSegmentLayout.image` and
    :meth:`TimeSegmentLayout.tier_maps` call. Another backend (another array library, another device) is an object
    with the same attributes in :attr:`TimeSegmentLayout.BACKENDS`: ``name``, ``torch`` (whether its rows are torch
    tensors), ``device(device)`` (the device a call runs on), ``put(device)`` (``(host array, column name) -> array`` on
    it), ``context()`` (around the arithmetic), ``field_spec(fspec, device)`` (the field contraction's scales and
    weights there), ``sums(...)`` (:func:`_sums_numpy`'s) and ``voxel_sums(...)`` (:func:`_voxel_sums_numpy`'s)."""
    name = "numpy"
    torch = False

    def device(self, device):
        if device not in (None, "cpu"):
            raise ValueError(f"the numpy backend runs on the host; device={device!r}")
        return None

    def put(self, device):
        return lambda a, name: a

    def context(self):
        return contextlib.nullcontext()

    def field_spec(self, fspec, device):
        return fspec

    sums = staticmethod(_sums_numpy)
    voxel_sums = staticmethod(_voxel_sums_numpy)


TimeSegmentLayout.BACKENDS = {"numpy": NumpyBackend()}
