"""Walk a myelinated cylinder with its field, write its time-segment layout with dmipy-sim, and record sim's own reads
and images of it.

Dev-only: needs dmipy-sim (and its JAX) installed; dmipy-replay's tests read what this writes and need neither. One
walk, one layout with tiers and the background, six partitions on one 0.1 ms grid: a square PGSE (``box``), a PGSE
with ramps (``trap``, a shaped term), a cosine OGSE (``cos``), a tensor encoding's three shapes on one interval
(``tensor``), a stimulated echo's lobes and mixing time (``pgste``) and a double diffusion encoding (``dde``). Writes
``tests/data/time_segments/`` and ``tests/data/segment_reference.npz`` / ``.json``. Run from the repository root with an
untouched sim checkout first on the path:

    JAX_PLATFORMS=cpu PYTHONPATH=/path/to/pristine/dmipy-sim python tools/make_segment_fixtures.py
"""
import json
import shutil
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
from segment_scenarios import TISSUE_KW, acquisitions, partitions  # noqa: E402

import dmipy_sim as d  # noqa: E402
from dmipy_sim.fields.hollow_cylinder import HollowCylinderField  # noqa: E402
from dmipy_sim.phantom.grid import Grid  # noqa: E402
from dmipy_sim.replay.bank import build_replay_pack  # noqa: E402
from dmipy_sim.replay.time_segments import Partition, TimeSegmentLayout, segment_rows, write_time_segment_layout  # noqa: E402
from dmipy_sim.spec.tissue import Tissue  # noqa: E402

NS = type("NS", (), dict(pgse=staticmethod(d.pgse), pgste=staticmethod(d.pgste), ogse=staticmethod(d.ogse),
                         ScannerSequence=d.ScannerSequence, Partition=Partition, Tissue=Tissue))
OUT = ROOT / "tests" / "data"
UM, D0, G_MAX = 1e-6, 2e-9, 1.0


def main():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        g = d.MyelinatedCylinder(2.0 * UM, 2.8 * UM, (0, 0, 1), D0, D0)
        w = d.simulate_trajectories(600, D0, g, 0.05, seed=7, window_T=0.05, field_basis=HollowCylinderField.of(g),
                                    require_gpu=False)
        r0 = np.asarray(w.position_coeffs)[:, 0, 0, :]
        lo, hi = r0.min(0), r0.max(0)
        vs = np.array([(hi[0] - lo[0]) / 2 * 1.001, (hi[1] - lo[1]) / 2 * 1.001, 1e-5])
        grid = Grid(shape=(2, 2, 1), voxel_size_m=tuple(vs), origin_m=(lo[0] - 1e-9 + vs[0] / 2, lo[1] - 1e-9 + vs[1] / 2, 0.0))
        pk = build_replay_pack(w, id="fixture/segments", license="CC-BY-4.0", citation="dmipy-replay fixture",
                               sigma_star=0.02, voxel_grid=grid)
        parts = partitions(NS)
        out = OUT / "time_segments"
        shutil.rmtree(out, ignore_errors=True)
        with tempfile.TemporaryDirectory() as t:
            segment_rows(w, parts, str(Path(t) / "rows"), grid=grid, pack=pk, g_max=G_MAX, tiers=True, background=True)
            write_time_segment_layout([str(Path(t) / "rows")], str(out))
    lay = TimeSegmentLayout(str(out))
    tissue = Tissue(**TISSUE_KW)
    ref, info = {}, {"sim_commit": subprocess.run(["git", "-C", str(Path(d.__file__).parents[1]), "rev-parse", "HEAD"],
                                                 capture_output=True, text=True).stdout.strip(), "failed": {}}
    for name in lay.partitions:
        ref[f"vectors/{name}"] = lay.vectors(name)
    for g in lay.manifest["groups"]:
        for k, v in lay.tier_columns(g).items():
            ref[f"tiers/{g}/{k}"] = v
    for key, (part, seq) in acquisitions(NS).items():
        name, L = lay.fit(seq)
        ref[f"{key}/levels"] = L
        info[f"{key}/partition"] = name
        S, floor = lay.image(seq, backend="jax")
        ref[f"{key}/S"], ref[f"{key}/floor"] = S, floor
        try:
            S2, _ = lay.image(seq, backend="jax", tissue=tissue, scanner=3.0, b0_direction=(0.6, 0.0, 0.8))
            ref[f"{key}/S_tiers"] = S2
        except ValueError as e:                                    # recorded: replay must refuse the same way
            info["failed"][f"{key}/S_tiers"] = f"{type(e).__name__}: {e}"
    maps = lay.tier_maps("box", tissue, 3.0, backend="jax", b0_direction=(0.6, 0.0, 0.8))
    for k, v in maps.items():
        if isinstance(v, dict):
            for p, m in v.items():
                ref[f"maps/{k}/{p}"] = m
        elif v is not None:
            ref[f"maps/{k}"] = v
    np.savez(OUT / "segment_reference.npz", **ref)
    (OUT / "segment_reference.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1)[:3000])


if __name__ == "__main__":
    main()
