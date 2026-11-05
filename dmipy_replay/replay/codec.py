"""The ``bitshuffle-zlib`` codec of a band range, and the array that reads it.

A band range of the integer band container (positions, boundary local time, field modes) may be stored under a ``codec``
named in its ``container`` entry (replay-pack-spec, CODEC_REGISTRY.md, "Container option: codec"). ``"bitshuffle-zlib"`` holds
the range's integers ``q`` -- the values of the fixed-width container, without the offset -- as zigzag integers in an 8-bit
lane (``bits <= 8``) or a 16-bit lane, bit-transposed in blocks of 4096 elements and deflated per chunk of
``chunk_walkers`` walkers, with a table of the chunks' byte offsets, so that a walker prefix is the chunks that cover it.
The decoded integers are the fixed-width container's, bit for bit.

:func:`encode_chunks` and :func:`decode_chunks` are the codec; :class:`CodecRange` is a stored range read as an array of
those integers (decoded when asked, a row range from the chunks that hold it); :func:`expand` turns a pack's stored tensors
into the arrays the readers take. Standard library ``zlib`` and NumPy only.
"""
from __future__ import annotations

import re
import zlib

import numpy as np

CODEC = "bitshuffle-zlib"
CODECS = (CODEC,)
BLOCK = 4096
PLANE_BYTES = BLOCK // 8
DEFAULT_LEVEL = 6

#: the tensors of a coded range: ``<prefix><base>_b<i>z<bits>`` (the chunk streams) and ``..._chunks`` (their byte offsets)
_STREAM = re.compile(r"^(?P<prefix>(?:s\d+/)?)(?P<base>pos_[xyz]|blt|susc_path_dct)_b(?P<i>\d+)z(?P<bits>\d+)$")


def lane_bits(bits):
    """The lane of a range of ``bits`` bits: 8 up to 8 bits, else 16."""
    return 8 if int(bits) <= 8 else 16


def lane_dtype(bits):
    """The signed integer type that holds the range's values: int8 up to 8 bits, else int16."""
    return np.dtype(np.int8 if lane_bits(bits) == 8 else np.int16)


def refuse_unknown(meta):
    """Raise unless every band range ``meta`` declares with a ``codec`` names one this module implements."""
    for where, ranges in container_ranges(meta):
        for i, r in enumerate(ranges):
            c = r.get("codec")
            if c is not None and c not in CODECS:
                raise ValueError(f"{where}[{i}].codec {c!r} is not a container codec this reader implements "
                                 f"({', '.join(CODECS)}); refusing the pack by that name")
            if c is not None:
                cw = r.get("chunk_walkers")
                if not isinstance(cw, int) or cw < 1 or cw & (cw - 1):
                    raise ValueError(f"{where}[{i}].chunk_walkers must be a power of two, got {cw!r}")
            elif "chunk_walkers" in r:
                raise ValueError(f"{where}[{i}] states chunk_walkers without a codec")


def container_ranges(meta):
    """``[(where, ranges), ...]``: the band containers a pack's metadata declares (positions, boundary local time, field
    modes), each the list of its range entries."""
    cx = (meta or {}).get("compression") or {}
    ch = cx.get("channels") or {}
    out = []
    if cx.get("container"):
        out.append(("compression.container", cx["container"]))
    if (ch.get("boundary_local_time") or {}).get("container"):
        out.append(("boundary_local_time.container", ch["boundary_local_time"]["container"]))
    if (ch.get("susc_field_basis") or {}).get("container"):
        out.append(("susc_field_basis.container", ch["susc_field_basis"]["container"]))
    return out


def without_codec(meta):
    """``meta`` with the ``codec`` and ``chunk_walkers`` of every band range removed: the metadata of the same pack held
    as decoded (fixed-width) ranges."""
    import copy
    out = copy.deepcopy(meta)
    for _, ranges in container_ranges(out):
        for r in ranges:
            r.pop("codec", None)
            r.pop("chunk_walkers", None)
    return out


def zigzag(q, bits):
    """The unsigned integers of the signed ``q`` in the lane of ``bits``: ``0, -1, 1, -2, ...`` -> ``0, 1, 2, 3, ...``."""
    x = np.asarray(q).astype(np.int32)
    return ((x << 1) ^ (x >> 31)).astype(np.uint8 if lane_bits(bits) == 8 else np.uint16)


def unzigzag(u, bits):
    """The inverse of :func:`zigzag`, as the signed type of the lane."""
    x = np.asarray(u).astype(np.int32)
    return ((x >> 1) ^ -(x & 1)).astype(lane_dtype(bits))


def _transpose(u):
    """The bit planes of the lane array ``u`` (``(n,)``, a multiple of :data:`BLOCK` long): per block, plane ``p`` (bit ``p``
    of every element) packed little-endian, planes in order of ``p``; ``(blocks * lane * 512,)`` uint8."""
    w = 8 * u.dtype.itemsize
    b = np.unpackbits(u.astype(u.dtype.newbyteorder("<")).view(np.uint8).reshape(-1, BLOCK, w // 8), axis=2, bitorder="little")
    return np.packbits(b.transpose(0, 2, 1), axis=2, bitorder="little").reshape(-1)


def _untranspose(buf, w):
    """The lane array of the planes ``buf`` of lane width ``w``: the inverse of :func:`_transpose`."""
    planes = np.unpackbits(np.frombuffer(buf, np.uint8).reshape(-1, w, PLANE_BYTES), axis=2, bitorder="little")
    dt = np.uint8 if w == 8 else np.uint16
    u = planes[:, 0].astype(dt)
    for p in range(1, w):
        u |= planes[:, p].astype(dt) << dt(p)
    return u.reshape(-1)


def encode_chunks(q, bits, chunk_walkers, *, level=DEFAULT_LEVEL):
    """The stored form of a band range: ``(stream, chunks)``, the deflated chunks one after the other (uint8 ``(S,)``) and
    their byte offsets (int64 ``(n_chunks + 1,)``). ``q`` is the range's signed integers, walker-leading
    (``(n_w, m)`` or ``(n_w, n_ch, m)``); a chunk is ``chunk_walkers`` walkers, their elements in C order."""
    q = np.asarray(q)
    cw = int(chunk_walkers)
    if cw < 1 or cw & (cw - 1):
        raise ValueError(f"chunk_walkers must be a power of two, got {chunk_walkers}")
    n = q.shape[0]
    flat = q.reshape(n, -1)
    lo = -(1 << (lane_bits(bits) - 1))
    if q.size and (flat.min() < lo or flat.max() >= -lo):
        raise ValueError(f"a value of the range does not fit its {lane_bits(bits)}-bit lane")
    parts, off = [], [0]
    for c in range(0, n, cw):
        u = zigzag(flat[c:c + cw].reshape(-1), bits)
        pad = (-len(u)) % BLOCK
        if pad:
            u = np.concatenate([u, np.zeros(pad, u.dtype)])
        parts.append(zlib.compress(_transpose(u).tobytes(), level))
        off.append(off[-1] + len(parts[-1]))
    stream = np.frombuffer(b"".join(parts), np.uint8) if parts else np.zeros(0, np.uint8)
    return stream, np.asarray(off, np.int64)


def decode_chunks(stream, chunks, bits, chunk_walkers, n_walkers, tail, start=0, stop=None):
    """The integers of walkers ``start:stop`` of a coded range of ``n_walkers`` walkers, ``(stop - start,) + tail``, int8 or int16 by the lane: the
    chunks that cover those walkers are inflated, the others not read. ``stream`` may hold only the leading bytes of
    the stored stream, as many as the chunks asked for need."""
    stop = n_walkers if stop is None else min(int(stop), n_walkers)
    start = max(int(start), 0)
    tail = tuple(int(t) for t in tail)
    per = int(np.prod(tail, dtype=np.int64))
    out = np.empty((max(stop - start, 0),) + tail, lane_dtype(bits))
    if stop <= start:
        return out
    cw = int(chunk_walkers)
    chunks = np.asarray(chunks, np.int64)
    buf = memoryview(np.asarray(stream, np.uint8))
    w = lane_bits(bits)
    for c in range(start // cw, -(-stop // cw)):
        rows = min(cw, n_walkers - c * cw)
        n_el = rows * per
        want = w * PLANE_BYTES * -(-n_el // BLOCK)
        a, b = int(chunks[c]), int(chunks[c + 1])
        if b > len(buf):
            raise ValueError(f"chunk {c} ends at byte {b}, the stream holds {len(buf)}")
        try:
            raw = zlib.decompress(buf[a:b])
        except zlib.error as e:
            raise ValueError(f"chunk {c} of a {CODEC} range does not inflate: {e}") from e
        if len(raw) != want:
            raise ValueError(f"chunk {c} of a {CODEC} range inflates to {len(raw)} bytes, {want} expected")
        q = unzigzag(_untranspose(raw, w)[:n_el], bits).reshape((rows,) + tail)
        lo, hi = max(start, c * cw), min(stop, c * cw + rows)
        out[lo - start:hi - start] = q[lo - c * cw:hi - c * cw]
    return out


def pack_fixed(q, bits):
    """The fixed-width container's tensor of the integers ``q`` (walker-leading, last axis the bands): int8 / int16 for 8 / 16
    bits, else the bit-packed uint8 rows of ``q + 2**(bits - 1)``, most significant bit first (the inverse of
    :func:`~dmipy_replay.replay.compression.unpack_bits`)."""
    q = np.asarray(q)
    if bits in (8, 16):
        return q.astype(np.int8 if bits == 8 else np.int16)
    lead, m = q.shape[:-1], q.shape[-1]
    u = (q.astype(np.int64) + (1 << (bits - 1))).reshape(-1, m)
    bitsmat = ((u[:, :, None] >> np.arange(bits - 1, -1, -1)[None, None, :]) & 1).astype(np.uint8)
    pad = (-m) % 8
    if pad:
        bitsmat = np.concatenate([bitsmat, np.zeros((len(u), pad, bits), np.uint8)], axis=1)
    packed = np.packbits(bitsmat.reshape(len(u), -1), axis=1)[:, :-(-m * bits // 8)]
    return packed.reshape(lead + (packed.shape[1],))


class CodecRange:
    """A band range stored under the ``bitshuffle-zlib`` codec, read as the array of its integers (int8 or int16,
    ``(n_walkers, [n_ch,] m)``): the tensor the fixed-width container's unpacked range is. Decoded when asked; a row
    slice inflates only the chunks that hold it, and ``np.asarray`` decodes it whole, once. ``stream`` may be the leading
    bytes of the stored stream when only a walker prefix is held (``n_held`` walkers)."""

    def __init__(self, stream, chunks, *, bits, chunk_walkers, n_walkers, tail, n_held=None):
        self.stream = np.asarray(stream, np.uint8)
        self.chunks = np.asarray(chunks, np.int64)
        self.bits, self.chunk_walkers = int(bits), int(chunk_walkers)
        self.n_total = int(n_walkers)
        self.n_held = self.n_total if n_held is None else int(n_held)
        self.tail = tuple(int(t) for t in tail)
        self.shape = (self.n_held,) + self.tail
        self.ndim = len(self.shape)
        self.dtype = lane_dtype(bits)
        self._full = None

    def __len__(self):
        return self.n_held

    @property
    def nbytes(self):
        """Bytes of the decoded integers."""
        return int(np.prod(self.shape, dtype=np.int64)) * self.dtype.itemsize

    @property
    def stored_nbytes(self):
        """Bytes the stored chunks of the held walkers take."""
        return int(self.chunks[-(-self.n_held // self.chunk_walkers)])

    def rows(self, start, stop):
        """The integers of walkers ``start:stop`` (a view of the whole decode once that is held)."""
        if self._full is not None:
            return self._full[start:stop]
        return decode_chunks(self.stream, self.chunks, self.bits, self.chunk_walkers, self.n_total, self.tail, start,
                             min(stop, self.n_held))

    def __array__(self, dtype=None, copy=None):
        if self._full is None:
            self._full = self.rows(0, self.n_held)
            self._full.flags.writeable = False
        return self._full if dtype is None else self._full.astype(dtype)

    def __getitem__(self, idx):
        idx = idx if isinstance(idx, tuple) else (idx,)
        first, rest = idx[0], idx[1:]
        if isinstance(first, slice) and first.step in (None, 1):
            start, stop, _ = first.indices(self.n_held)
            block = self.rows(start, stop)
            return block[(slice(None),) + rest] if rest else block
        if isinstance(first, (int, np.integer)):
            r = int(first) % self.n_held if first < 0 else int(first)
            block = self.rows(r, r + 1)[0]
            return block[rest] if rest else block
        return np.asarray(self)[idx]

    def prefix(self, n):
        """The range of the first ``n`` walkers (a view of the same stored bytes)."""
        return CodecRange(self.stream, self.chunks, bits=self.bits, chunk_walkers=self.chunk_walkers, n_walkers=self.n_total,
                          tail=self.tail, n_held=min(int(n), self.n_held))

    def fixed_width(self):
        """The same integers in the fixed-width container's layout (:func:`pack_fixed`)."""
        return pack_fixed(np.asarray(self), self.bits)

    def __reduce__(self):
        return (_rebuild, (self.stream, self.chunks, self.bits, self.chunk_walkers, self.n_total, self.tail, self.n_held))


def _rebuild(stream, chunks, bits, chunk_walkers, n_walkers, tail, n_held):
    return CodecRange(stream, chunks, bits=bits, chunk_walkers=chunk_walkers, n_walkers=n_walkers, tail=tail, n_held=n_held)


def is_stored_range(name):
    """Whether ``name`` is a chunk stream or chunk table of a coded range."""
    return bool(_STREAM.match(name) or (name.endswith("_chunks") and _STREAM.match(name[:-len("_chunks")])))


def _range_entry(meta, base, i):
    cx = (meta or {}).get("compression") or {}
    ch = cx.get("channels") or {}
    if base.startswith("pos_"):
        ranges, n_ch = cx.get("container"), 1
    elif base == "blt":
        ranges, n_ch = (ch.get("boundary_local_time") or {}).get("container"), 1
    else:
        f = ch.get("susc_field_basis") or {}
        ranges, n_ch = f.get("container"), int(f.get("n_ch", 12))
    if not ranges or i >= len(ranges):
        raise ValueError(f"the tensor {base}_b{i}z.. names a band range the metadata does not declare")
    r = ranges[i]
    k0, k1 = r["bands"] if "bands" in r else r["modes"]
    return r, n_ch, int(k1 - k0)


def expand(arrays, meta, n_walkers=None):
    """``arrays`` with every coded range's chunk stream and chunk table replaced by a :class:`CodecRange` under the name the
    fixed-width container gives the range (:func:`~dmipy_replay.replay.compression.range_name`), the others as they are;
    ``arrays`` itself when it holds no coded range. A tensor of a coded range the metadata does not declare as
    ``bitshuffle-zlib``, or at another depth, is refused. ``n_walkers`` is the pack's walkers (read from its per-walker
    tensors when absent)."""
    names = [k for k in arrays if _STREAM.match(k)]
    if not names:
        refuse_unknown(meta)
        return arrays
    refuse_unknown(meta)
    from .compression import range_name
    out = {k: v for k, v in arrays.items() if not is_stored_range(k)}
    for k in names:
        m = _STREAM.match(k)
        base, i, bits = m["base"], int(m["i"]), int(m["bits"])
        r, n_ch, width = _range_entry(meta, base, i)
        if r.get("codec") != CODEC or int(r["bits"]) != bits:
            raise ValueError(f"tensor {k!r} is not the {CODEC} range the metadata declares ({r})")
        if f"{k}_chunks" not in arrays:
            raise ValueError(f"tensor {k!r} has no chunk table {k + '_chunks'!r}")
        n_w = n_walkers if n_walkers is not None else _walkers_of(arrays, m["prefix"], meta)
        chunks = np.asarray(arrays[f"{k}_chunks"])
        cw = int(r["chunk_walkers"])
        if chunks.dtype != np.int64 or chunks.shape != (-(-n_w // cw) + 1,):
            raise ValueError(f"{k}_chunks must be int64 ({-(-n_w // cw) + 1},) for {n_w} walkers in chunks of {cw}, "
                             f"got {chunks.dtype} {chunks.shape}")
        tail = (n_ch, width) if base == "susc_path_dct" else (width,)
        out[m["prefix"] + range_name(base, i, bits)] = CodecRange(arrays[k], chunks, bits=bits, chunk_walkers=cw,
                                                                  n_walkers=n_w, tail=tail)
    return out


def _walkers_of(arrays, prefix, meta):
    for k in ("pos_x_ends", "pos_y_ends", "pos_z_ends", "pos_x", "blt_start", "spin_weights", "comp_static"):
        if prefix + k in arrays:
            return int(np.shape(arrays[prefix + k])[0])
    n = ((meta or {}).get("walk_params") or {}).get("n_walkers")
    if n is None:
        raise ValueError("a coded range needs the pack's walker count: no per-walker tensor and no walk_params.n_walkers")
    return int(n)


_FIXED = re.compile(r"^(?P<prefix>(?:s\d+/)?)(?P<base>pos_[xyz]|blt|susc_path_dct)_b(?P<i>\d+)(?:u(?P<bits>\d+))?$")


def encode_ranges(arrays, meta, *, codec, chunk_walkers, level=DEFAULT_LEVEL):
    """``(arrays, meta)`` of the same pack with every fixed-width band range stored under ``codec`` in chunks of
    ``chunk_walkers`` walkers: each range's tensor replaced by its chunk stream and chunk table, its ``container`` entry naming
    the codec and the chunk size. The decoded integers, the scales and every other tensor are unchanged. A range that is
    already coded is refused."""
    import copy
    from .compression import _load_range, range_name
    if codec != CODEC:
        raise ValueError(f"codec must be one of {CODECS}, got {codec!r}")
    if any(_STREAM.match(k) for k in arrays):
        raise ValueError("the pack already stores a band range under a codec")
    meta = copy.deepcopy(meta)
    for _, ranges in container_ranges(meta):
        for r in ranges:
            r["codec"], r["chunk_walkers"] = codec, int(chunk_walkers)
    out, n_coded = {}, 0
    for k, v in arrays.items():
        m = _FIXED.match(k)
        if not m:
            out[k] = v
            continue
        n_coded += 1
        t = np.asarray(v)
        bits = int(m["bits"]) if m["bits"] else 8 * t.dtype.itemsize
        r, n_ch, width = _range_entry(meta, m["base"], int(m["i"]))
        if int(r["bits"]) != bits:
            raise ValueError(f"tensor {k!r} holds {bits}-bit bands, the metadata declares {r['bits']}")
        if m["base"] == "susc_path_dct":
            q = np.swapaxes(np.asarray(_load_range(np.swapaxes(t, 1, 2), bits, width)), 1, 2)
        else:
            q = np.asarray(_load_range(t, bits, width))
        stream, chunks = encode_chunks(q, bits, chunk_walkers, level=level)
        name = f"{m['prefix']}{m['base']}_b{int(m['i'])}z{bits}"
        out[name], out[f"{name}_chunks"] = stream, chunks
    if not n_coded:
        raise ValueError("the pack holds no integer band range to store under a codec (its bands are in the float container)")
    return out, meta
