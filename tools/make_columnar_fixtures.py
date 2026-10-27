"""Walk a small two-block fill with dmipy-sim, consolidate it into the columnar layout, and record sim's own plan and
image of it.

Dev-only: needs dmipy-sim (and its JAX) installed; dmipy-replay's tests read what this writes and need neither. Two
layouts of one walk: ``columnar_float`` (the float coefficients, band groups 16 / 32 / 48) and ``columnar_quant``
(the banded container). Writes ``tests/data/<layout>/`` and ``tests/data/columnar_reference.npz`` / ``.json``. Run
from the repository root with an untouched sim checkout first on the path:

    JAX_PLATFORMS=cpu PYTHONPATH=/path/to/pristine/dmipy-sim python tools/make_columnar_fixtures.py
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
from columnar_scenarios import seqs, tissue_of  # noqa: E402

import dmipy_sim as d  # noqa: E402
from dmipy_sim.fill.consolidate import consolidate  # noqa: E402
from dmipy_sim.phantom.grid import Grid  # noqa: E402
from dmipy_sim.replay import ReplayPack  # noqa: E402
from dmipy_sim.replay.bank import build_replay_pack  # noqa: E402
from dmipy_sim.spec import Tissue  # noqa: E402

NS = type("NS", (), dict(pgse=staticmethod(d.pgse), set_b=staticmethod(d.set_b), Tissue=Tissue))

OUT = ROOT / "tests" / "data"
T_MAX, H, K = 4e-3, 5e-5, 48


def _walk(tmp, name, **container):
    g = d.PackedCylinders([1.5e-6, 2e-6], [[-2.5e-6, 0.0], [2.5e-6, 0.0]], 10e-6)
    grid = Grid(shape=(2, 2, 1), voxel_size_m=(5e-6, 5e-6, 1e-5), origin_m=(-2.5e-6, -2.5e-6, 0.0))
    (tmp / name).mkdir()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        starts = np.asarray(d.simulate_trajectories(800, 2e-9, g, 2 * H, H, seed=30, require_gpu=False, K=1).position_coeffs[:, 0, 0])
        for block, sel in ((0, starts[:, 0] < 0), (1, starts[:, 0] >= 0)):
            r0 = np.ascontiguousarray(starts[sel], np.float32)
            w = d.simulate_trajectories(len(r0), 2e-9, g, T_MAX, H, seed=31 + block, r0=r0, require_gpu=False, K=K)
            pk = build_replay_pack(w, id=f"t/columnar-{block}", license="CC-BY-4.0", citation="dmipy-replay fixture", K=K,
                                   voxel_grid=grid, out_path=str(tmp / name / f"block-{block:04d}.p1.rpk"), **container)
            if not container:                    # one walk's shards share one container, as a fill's recipe pins it
                c = pk.meta["compression"]
                container = dict(position_container=c["container"],
                                 blt_container=c["channels"]["boundary_local_time"]["container"])
    out = OUT / name
    shutil.rmtree(out, ignore_errors=True)
    consolidate(str(tmp / name), str(out), blocks=[0, 1], id=f"t/{name}")
    return ReplayPack.open(str(out))


def main():
    ref, info = {}, {"sim_commit": subprocess.run(["git", "-C", str(Path(d.__file__).parents[1]), "rev-parse", "HEAD"],
                                                 capture_output=True, text=True).stdout.strip(), "layouts": {}}
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        for name, kw in (("columnar_float", dict(position_container=None, blt_container=None)), ("columnar_quant", {})):
            lay = _walk(tmp, name, **kw)
            info["layouts"][name] = dict(band_groups=lay.band_groups, K=lay.K, n_rows=lay.n_rows)
            for key, seq in seqs(NS).items():
                for tol in (0.25, 1e-12):
                    p = lay.plan(seq, tol=tol)
                    info["layouts"][name][f"{key}/plan/{tol:g}"] = {k: p[k] for k in ("K", "modes", "rows", "bytes")}
                S, floor, plan = lay.image(seq, tol=1e-12)
                ref[f"{name}/{key}/S"] = S
                ref[f"{name}/{key}/floor"] = floor
                tis = tissue_of(NS, lay.meta)
                S2, _, _ = lay.image(seq, settings=[(None, None), (tis, 3.0)], tol=1e-12)
                ref[f"{name}/{key}/settings"] = S2
    np.savez(OUT / "columnar_reference.npz", **ref)
    (OUT / "columnar_reference.json").write_text(json.dumps(info, indent=1, default=list))
    print(json.dumps(info, indent=1, default=list)[:2000])


if __name__ == "__main__":
    main()
