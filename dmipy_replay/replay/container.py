"""The ``.rpk`` container (RPK.md 12): a safetensors file, read and written with numpy alone.

A safetensors file is an 8-byte little-endian header length ``n``, ``n`` bytes of JSON -- per tensor its ``dtype``,
``shape`` and ``data_offsets`` into the buffer that follows, and the string map ``__metadata__`` -- then the buffer.
There is no code in it to execute, which is why the format was chosen; reading it needs no library. A read maps the
file copy-on-write and returns each tensor as a view of the mapping, so a pack of many gigabytes costs the pages a
replay touches, and a reader that writes into a tensor changes its own copy, never the file.
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
    offset 0): a view of it, writable when the buffer is."""
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
    """``(arrays, metadata)`` of the safetensors file at ``path``: every tensor as a view of the file mapped
    copy-on-write, and the ``__metadata__`` string map (``{}`` when the file has none)."""
    with open(os.fspath(path), "rb") as f:
        size = os.fstat(f.fileno()).st_size
        if size < 8:
            raise ValueError(f"{path}: not a safetensors file ({size} bytes)")
        m = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_COPY)
    n, header = parse_header(m[:8 + min(int.from_bytes(m[:8], "little"), MAX_HEADER)])
    meta = header.pop("__metadata__", None) or {}
    buf = memoryview(m)[8 + n:]
    arrays = {k: tensor_of(buf, v, k) for k, v in header.items()}
    return arrays, meta


#: the order safetensors lays tensors out in, by dtype (then by name): the reference writer's (``safetensors.serialize``)
_ST_ORDER = ("U64", "I64", "F64", "F32", "U32", "I32", "F16", "U16", "I16", "I8", "U8", "BOOL")
_WRITE_BYTES = 1 << 26                  #: the largest piece of a tensor written at once (bytes)


def write_safetensors(path, arrays, metadata=None):
    """Write ``arrays`` (name -> ndarray, or a list of ndarrays stored as their concatenation along the first axis, as a
    merge stores its shards' tensors) and the string map ``metadata`` as a safetensors file at ``path``. The file is
    byte for byte what ``safetensors.numpy.save_file`` writes for the contiguous arrays (a 0-d array stored as one
    element, as ``np.ascontiguousarray`` makes it): the header's ``__metadata__`` first and the tensors in their data
    order -- by dtype as :data:`_ST_ORDER`, then by name -- compact, padded with spaces to eight bytes, the data
    little-endian. The tensors are streamed in pieces of at most :data:`_WRITE_BYTES` into a sibling file that replaces
    ``path`` once complete: a pack mapped from ``path`` (:func:`read_safetensors`) keeps reading the file it was mapped
    from."""
    import struct
    parts = {k: [np.ascontiguousarray(x) for x in (v if isinstance(v, (list, tuple)) else [v])] for k, v in arrays.items()}
    head = {}
    for k, ps in parts.items():
        dt = ps[0].dtype
        if any(x.dtype.name != dt.name or x.shape[1:] != ps[0].shape[1:] for x in ps):
            raise ValueError(f"the pieces of tensor {k!r} differ in dtype or row shape")
        name = _NAMES.get(dt.newbyteorder("<").str)
        if name is None:
            raise TypeError(f"tensor {k!r} is {dt}, which safetensors does not store")
        head[k] = dict(dtype=name, shape=[sum(int(x.shape[0]) for x in ps)] + [int(d) for d in ps[0].shape[1:]])
    order = sorted(head, key=lambda k: (_ST_ORDER.index(head[k]["dtype"]), k))
    off = 0
    for k in order:
        nb = sum(int(x.nbytes) for x in parts[k])
        head[k]["data_offsets"] = [off, off + nb]; off += nb
    md = {"__metadata__": {str(k): str(v) for k, v in metadata.items()}} if metadata else {}
    h = json.dumps(dict(**md, **{k: head[k] for k in order}), separators=(",", ":"), ensure_ascii=False).encode()
    h += b" " * (-len(h) % 8)
    tmp = f"{os.fspath(path)}.{os.getpid()}.part"
    try:
        with open(tmp, "wb") as f:
            f.write(struct.pack("<Q", len(h))); f.write(h)
            for k in order:
                for x in parts[k]:
                    step = max(1, _WRITE_BYTES // max(1, x.nbytes // max(1, len(x))))
                    for i in range(0, len(x), step):
                        f.write(np.ascontiguousarray(x[i:i + step], x.dtype.newbyteorder("<")).tobytes())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
