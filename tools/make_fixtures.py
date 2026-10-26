"""Walk the parity packs with dmipy-sim and record its replay of every scenario of ``tests/scenarios.py``.

Dev-only: needs dmipy-sim (and its JAX) installed; dmipy-replay's own tests read what this writes and need neither.
Writes ``tests/data/<pack>.rpk``, ``tests/data/reference.npz`` (one array per scenario) and ``tests/data/reference.json``
(the sim commit, the scenarios, what failed on sim's side). Run from the repository root:

    JAX_PLATFORMS=cpu python tools/make_fixtures.py
"""
import json
import os
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
from scenarios import SCENARIOS  # noqa: E402

import dmipy_sim as d  # noqa: E402
from dmipy_sim.replay.bank import build_replay_pack  # noqa: E402
from dmipy_sim.replay.replay import read_rpk  # noqa: E402
from dmipy_sim.spec import Tissue  # noqa: E402

OUT = ROOT / "tests" / "data"


def _three_strands(tmp):
    """The three-strand DiSCo-format substrate of sim's own field tests, a sheath on each: its spec."""
    from dmipy_sim.io.strands import write_tck, concat_centerlines
    from dmipy_sim.spec import disco_spec
    cls_ = [np.array([[x, 0, -12e-6], [x, 0.5e-6, 0], [x, 0, 12e-6]]) + 10e-6 for x in (-5e-6, 0, 5e-6)]
    tck, dia = str(tmp / "t.tck"), str(tmp / "d.txt")
    write_tck(tck, *concat_centerlines(cls_), coordinate_unit_m=25e-6)
    np.savetxt(dia, np.array([2 * r for r in (1.5e-6, 1.0e-6, 2.0e-6)]) / 1e-3)
    return disco_spec(tck, dia, side_m=20e-6)


def build_packs():
    """Each pack as ``(name, pack)``, walked only when ``tests/data/<name>.rpk`` is not there yet (``--rebuild``
    walks them all)."""
    have = lambda name: (OUT / f"{name}.rpk").exists() and "--rebuild" not in sys.argv
    for name, make in _PACKS:
        if not have(name):
            yield name, make()


def _cylinder():
    w = d.simulate_trajectories(2000, 2e-9, d.Cylinder(2e-6, (0, 0, 1)), 0.04, seed=1, require_gpu=False, K=64)
    return build_replay_pack(w, id="fixture/cylinder", license="CC-BY-4.0", citation="dmipy-replay fixture")


def _permeable():
    perm = d.Cylinder(2e-6, (0, 0, 1), permeability=5e-5)
    w = d.simulate_trajectories(2000, 2e-9, perm, 0.04, seed=2, require_gpu=False, K=64)
    return build_replay_pack(w, id="fixture/permeable", license="CC-BY-4.0", citation="dmipy-replay fixture")


def _windows():
    w = d.simulate_trajectories(600, 2e-9, d.Cylinder(2e-6, (0, 0, 1)), 0.2, seed=3, require_gpu=False, K=32, window_T=0.1)
    return build_replay_pack(w, id="fixture/windows", license="CC-BY-4.0", citation="dmipy-replay fixture")


def _field():
    from dmipy_sim.spec import walk_spec
    spec = _three_strands(Path(tempfile.mkdtemp()))
    w = walk_spec(spec, 150, 7e-3, 2e-4, seed=4, n_probe=20_000, require_gpu=False, adaptive_steps=True, K=16)
    return build_replay_pack(w, id="fixture/field", license="CC-BY-4.0", citation="dmipy-replay fixture", K=16, susc_path_K=16)


_PACKS = [("cylinder", _cylinder), ("permeable", _permeable), ("windows", _windows), ("field", _field)]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, pack in build_packs():
        pack.save(OUT / f"{name}.rpk")
        print(f"{name}: {pack.n_walkers} walkers, K={pack.K}, {os.path.getsize(OUT / f'{name}.rpk')} bytes", flush=True)
    ns = SimpleNamespace(pgse=d.pgse, pgste=d.pgste, ogse=d.ogse, gre=d.gre, cpmg=d.cpmg, Tissue=Tissue)
    packs = {}
    out, failed = {}, {}
    for name, key, fn in SCENARIOS:
        p = packs.get(key) or packs.setdefault(key, read_rpk(OUT / f"{key}.rpk"))
        try:
            out[name] = np.asarray(fn(ns, p))
        except Exception as e:                                       # recorded: replay must refuse the same way
            failed[name] = f"{type(e).__name__}: {e}"
            traceback.print_exc()
        print(name, "ok" if name in out else "FAILED", flush=True)
    np.savez(OUT / "reference.npz", **out)
    commit = subprocess.run(["git", "-C", str(Path(d.__file__).parents[1]), "rev-parse", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    (OUT / "reference.json").write_text(json.dumps(dict(dmipy_sim_commit=commit, scenarios=[s[0] for s in SCENARIOS],
                                                        failed=failed), indent=1))


if __name__ == "__main__":
    main()
