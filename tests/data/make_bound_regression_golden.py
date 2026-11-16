"""Writes ``bound_regression_golden.npz``: what the client returned on the committed ``slab_exchange.rpk`` (a ``comp``
event column and no ``bound`` column) for a spin echo of 50 ms with T2 0.080 s and 0.020 s per pool, before the ``bound``
column was read and before the Bloch route read the occupancy runs between block ends.

The file is a record of that behaviour and is never regenerated: ``tests/test_bound_column.py`` holds the client to it
bit for bit where the specification says the reading did not change: a pack without a ``bound`` column on the scalar
routes (``replay``, ``walker_signals``, the study route's ``Primitives.signals``), and on the Bloch route the walkers whose
``comp`` holds no event in ``(0, TE]``.

Run from the repository root: ``python tests/data/make_bound_regression_golden.py``.
"""
from pathlib import Path

import numpy as np

import dmipy_replay as dr
from dmipy_replay import Tissue
from dmipy_replay.replay.compression import occupancy_events

HERE = Path(__file__).parent
TE = 0.05


def read_all(pack):
    n_t = int(round(TE / 1e-4)) + 1
    seq = dr.pgse([[1.0, 0, 0]], 0.1 * TE, 0.4 * TE, bvalues=[0.0], TE=TE, n_t=n_t)
    tissue = Tissue(T2={"intra": 0.080, "extra": 0.020})
    w, ew, E = pack.walker_signals(seq, tissue=tissue, tol=None)
    scalar_mean = np.asarray(pack.replay(seq, tissue=tissue, complex_signal=True, tol=None))
    prim = pack.walker_primitives(seq, tol=None)                       # the study route: the exposures once, the tissue applied after
    sw, sew, sE = prim.signals(tissue)
    study_mean = np.asarray(prim.signal(tissue, complex_signal=True))
    bloch_walker = np.asarray(pack.replay_bloch(seq, tissue=tissue, per_walker=True, complex_signal=True))
    bloch_mean = np.asarray(pack.replay_bloch(seq, tissue=tissue, complex_signal=True))
    return dict(scalar_w=np.asarray(w), scalar_ew=np.asarray(ew), scalar_E=np.asarray(E), scalar_mean=scalar_mean,
                study_ew=np.asarray(sew), study_E=np.asarray(sE), study_mean=study_mean,
                bloch_walker=bloch_walker, bloch_mean=bloch_mean)


def walkers_with_event_in(pack, t_hi):
    """Boolean per walker: the ``comp`` column has an event at a time in ``(0, t_hi]``."""
    col = next(d for d in pack.meta["compression"]["channels"]["compartment"]["columns"] if d["name"] == "comp")
    start, ev = occupancy_events(pack.arrays, col)
    out = np.zeros(len(start), bool)
    for i in range(len(start)):
        ts = ev.t[ev.offsets[i]:ev.offsets[i + 1]]
        out[i] = bool(np.any((ts > 0.0) & (ts <= t_hi)))
    return out


if __name__ == "__main__":
    pack = dr.read_rpk(HERE / "slab_exchange.rpk")
    out = read_all(pack)
    out["has_event"] = walkers_with_event_in(pack, TE)
    np.savez_compressed(HERE / "bound_regression_golden.npz", **out)
    print({k: v.shape for k, v in out.items()}, "walkers with an event in (0, TE]:", int(out["has_event"].sum()))
