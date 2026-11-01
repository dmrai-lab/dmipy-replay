"""The package is numpy and the standard library: importing it, reading a pack, replaying it, reading a ``.seq`` and
building from a b-value table, imaging a columnar
layout and importing the thread cap and the timing record load nothing else."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HEAVY = ("jax", "scipy", "torch", "safetensors", "huggingface_hub", "pypulseq", "requests", "matplotlib", "dipy")


def test_nothing_heavy_is_imported():
    code = f"""
import sys
sys.path.insert(0, {str(ROOT)!r})
import dmipy_replay as dr
from dmipy_replay.sequences.pulseq import from_pulseq
p = dr.read_rpk({str(ROOT / 'tests' / 'data' / 'cylinder.rpk')!r})
p.replay(dr.pgse([[1, 0, 0]], 0.006, 0.02, bvalues=[1e9]))
p.replay_bloch(dr.pgse([[1, 0, 0]], 0.006, 0.02, bvalues=[1e9]), b1_scale=0.9)
from_pulseq({str(ROOT / 'tests' / 'data' / 'seq' / 'foreign_sinc_se.seq')!r})
from dmipy_replay import from_btable
lay = dr.open_columnar({str(ROOT / 'tests' / 'data' / 'columnar_quant')!r})
lay.image(dr.pgse([[1, 0, 0]], 0.6e-3, 1.8e-3, bvalues=[1e9], n_t=401, slew_rate=float("inf"), TE=4e-3))
import dmipy_replay._blas, dmipy_replay.replay.phases, dmipy_replay.replay.pose, dmipy_replay.replay.pose_device
from dmipy_replay.replay.compression import resolve_device
resolve_device("numpy")
print(sorted({{m.split('.')[0] for m in sys.modules}} & set({HEAVY!r})))
"""
    out = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, check=True).stdout
    assert out.strip().splitlines()[-1] == "[]"
