"""Every band range of whole replay packs, stored under the ``bitshuffle-zlib`` codec and read back, against the fixed-width
container's integers. The packs are named by the environment variable ``DMIPY_REPLAY_REAL_PACKS`` (``.rpk`` paths separated by
``:``); without it the test is skipped. Memory: about three times the largest pack."""
import json
import os
import re

import numpy as np
import pytest

from dmipy_replay.replay import codec, compression as cx, container

PACKS = [p for p in os.environ.get("DMIPY_REPLAY_REAL_PACKS", "").split(":") if p]


@pytest.mark.parametrize("path", PACKS or [pytest.param(None, marks=pytest.mark.skip("DMIPY_REPLAY_REAL_PACKS is not set"))])
def test_every_range_of_a_whole_pack_round_trips(path):
    arrays, hdr = container.read_safetensors(path)
    meta = json.loads(hdr["rpk"])
    coded, cmeta = codec.encode_ranges(dict(arrays), meta, codec=codec.CODEC, chunk_walkers=4096)
    expanded = codec.expand(coded, cmeta)
    assert not [k for k in expanded if codec.is_stored_range(k)]
    n_checked = 0
    for k, t in arrays.items():
        m = re.match(r"^(pos_[xyz]|blt|susc_path_dct)_b(\d+)(?:u(\d+))?$", k)
        if not m:
            assert k in expanded and np.array_equal(expanded[k], t), k
            continue
        bits = int(m.group(3)) if m.group(3) else 8 * t.dtype.itemsize
        got = np.asarray(expanded[k])
        _, n_ch, width = codec._range_entry(meta, m.group(1), int(m.group(2)))
        if m.group(1) == "susc_path_dct":
            want = np.swapaxes(np.asarray(cx._load_range(np.swapaxes(t, 1, 2), bits, width)), 1, 2)
        else:
            want = np.asarray(cx._load_range(t, bits, width))
        assert got.shape == want.shape and np.array_equal(got, want), k
        n_checked += 1
    assert n_checked
