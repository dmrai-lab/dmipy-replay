"""The ``.rpk`` container (RPK.md 12): a safetensors file, read and written with numpy alone.

A safetensors file is an 8-byte little-endian header length ``n``, ``n`` bytes of JSON -- per tensor its ``dtype``,
``shape`` and ``data_offsets`` into the buffer that follows, and the string map ``__metadata__`` -- then the buffer.
There is no code in it to execute, which is why the format was chosen; reading it needs no library. A read maps the
file and returns each tensor as a read-only view of the mapping, so a pack of many gigabytes costs the pages a replay
touches.
"""
import json
import mmap
import os

import numpy as np

#: safetensors dtype names -> numpy (little-endian; BF16 has no numpy dtype and is refused by name)
DTYPES = {"BOOL": "bool", "U8": "<u1", "I8": "<i1", "I16": "<i2", "U16": "<u2", "I32": "<i4", "U32": "<u4",
          "I64": "<i8", "U64": "<u8", "F16": "<f2", "F32": "<f4", "F64": "<f8"}
_NAMES = {np.dtype(v).str: k for k, v in DTYPES.items()}
_NAMES[np.dtype("bool").str] = "BOOL"

#: the largest header a reader accepts (the format's own limit is 100 MB)
MAX_HEADER = 100_000_000


def parse_header(blob):
    """The header JSON of a safetensors file from its first bytes: ``(n, header)``, ``n`` the header's length. Refuses
    a length beyond :data:`MAX_HEADER` or past the bytes given."""
    if len(blob) < 8:
        raise ValueError("not a safetensors file: fewer than 8 bytes")
    n = int.from_bytes(blob[:8], "little")
    if n > MAX_HEADER:
        raise ValueError(f"not a safetensors file: a header of {n} bytes")
    if len(blob) < 8 + n:
        raise ValueError(f"truncated safetensors header: {len(blob) - 8} of {n} bytes")
    return n, json.loads(bytes(blob[8:8 + n]))


def tensor_of(buf, entry, name="tensor"):
    """One tensor from its header ``entry`` over the data buffer ``buf`` (bytes-like, starting at the buffer's
    offset 0): a read-only view."""
    dtype = entry["dtype"]
    if dtype not in DTYPES:
        raise NotImplementedError(f"{name}: safetensors dtype {dtype!r} has no numpy dtype")
    a, b = (int(x) for x in entry["data_offsets"])
    dt = np.dtype(DTYPES[dtype])
    shape = tuple(int(s) for s in entry["shape"])
    count = int(np.prod(shape, dtype=np.int64)) if shape else 1
    if b - a != count * dt.itemsize:
        raise ValueError(f"{name}: {b - a} bytes for {shape} of {dtype}")
    return np.frombuffer(buf, dtype=dt, count=count, offset=a).reshape(shape)


def read_safetensors(path):
    """``(arrays, metadata)`` of the safetensors file at ``path``: every tensor as a read-only view of the mapped file,
    and the ``__metadata__`` string map (``{}`` when the file has none)."""
    with open(os.fspath(path), "rb") as f:
        size = os.fstat(f.fileno()).st_size
        if size < 8:
            raise ValueError(f"{path}: not a safetensors file ({size} bytes)")
        m = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    n, header = parse_header(m[:8 + min(int.from_bytes(m[:8], "little"), MAX_HEADER)])
    meta = header.pop("__metadata__", None) or {}
    buf = memoryview(m)[8 + n:]
    arrays = {k: tensor_of(buf, v, k) for k, v in header.items()}
    return arrays, meta


def write_safetensors(path, arrays, metadata=None):
    """Write ``arrays`` (name -> ndarray) and the string map ``metadata`` as a safetensors file at ``path``: tensors in
    name order, each C-contiguous little-endian, the header padded with spaces to a multiple of 8 bytes."""
    names = sorted(arrays)
    tens = {}
    for k in names:
        a = np.ascontiguousarray(arrays[k])
        if a.dtype.byteorder == ">":
            a = a.astype(a.dtype.newbyteorder("<"))
        if a.dtype.str not in _NAMES:
            raise TypeError(f"{k}: dtype {a.dtype} has no safetensors name")
        tens[k] = a
    header, off = {}, 0
    for k in names:
        nb = tens[k].nbytes
        header[k] = {"dtype": _NAMES[tens[k].dtype.str], "shape": list(tens[k].shape), "data_offsets": [off, off + nb]}
        off += nb
    if metadata:
        header["__metadata__"] = {str(k): str(v) for k, v in metadata.items()}
    hb = json.dumps(header, separators=(",", ":")).encode()
    hb += b" " * (-len(hb) % 8)
    with open(os.fspath(path), "wb") as f:
        f.write(len(hb).to_bytes(8, "little"))
        f.write(hb)
        for k in names:
            f.write(tens[k].tobytes())
