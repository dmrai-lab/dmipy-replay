"""Walk the cross-tier fixture packs of ``tests/test_cross_tier.py`` with dmipy-sim (CPU, minutes).

Two packs, each a fresh seed, each a 1-D substrate whose exact reference is
:class:`dmipy_analytic.bloch_torrey.BlochTorrey1D`:

* ``plane_linear_field.rpk``: two reflecting plates ``PLANE_L`` apart (dmipy-sim's ``Box1D``), the positions and the
  field channel of a prescribed linear field ``dB = PLANE_G0 (x - L/2)`` at ``B0 = PLANE_B0``, ``chi = PLANE_CHI``. The
  field is a grid basis whose ``iso_P_xx`` channel is ``G0 (x - L/2) / (B0 chi)`` and every other channel zero, so the
  contraction ``iso_local - Q . iso_P`` at a main field along z reads it back.
* ``slab_exchange.rpk``: the permeable slab (dmipy-sim's ``PermeableSlab1D``): compartment A ``[0, L/2]`` (pool
  ``intra``, every walker starts there) and B ``[L/2, L]`` (``extra``) across a membrane of permeability
  ``SLAB_KAPPA``, reflecting outer walls; the occupancy events (C1), the boundary local time at the membrane's
  reflections (C2) and the field of a magnetised layer B, ``dB = chi B0 (1_B - 1/2) / 3`` (C3).

The field of neither substrate comes from a magnetic pool's geometry, so neither can be walked by ``dmipy_sim.walk``
(its field route computes the field of a cylinder shell, a mesh, a sphere union or swept polylines): both are walked by
the engine's ``_walk_geometry`` with the prescribed grid as ``field_basis``, and the walk's spec states the magnetic
pool (``SubstrateSpec.replace(susceptibility=...)`` and the ``field`` tier) that ``build_replay_pack`` requires of a
pack storing the field channel. Every band is the one a production walk takes: the position band is the envelope's,
and the field channel's band is the one the walk derives on its pilot (no ``K_c3`` is given).

    python tools/make_xtier_fixtures.py [--n-plane N] [--n-slab N] [--out DIR]
"""
import argparse
import dataclasses
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "tests" / "data"

D0 = 2e-9
PLANE_L, PLANE_T, PLANE_SEED = 8e-6, 0.04, 70311
PLANE_B0, PLANE_CHI, PLANE_G0 = 3.0, 1e-6, 0.1            # T, -, T/m: dB = G0 (x - L/2)
SLAB_L, SLAB_KAPPA, SLAB_T, SLAB_SEED = 4e-6, 1e-5, 0.05, 70312
SLAB_CHI = -1e-7
SLAB_RHO2_MAX = 1e-4
BAND_HZ = 2000.0


def _grid(L, f, n=802):
    """The :class:`FieldGrid` of a field basis that is ``f(x)`` in ``iso_P_xx`` and zero elsewhere, on ``n`` nodes
    reaching half a node beyond each wall (no clamped edge inside the substrate), one voxel across y and z."""
    from dmipy_sim.fields.susceptibility_field import FieldGrid
    h = L / (n - 2)
    x = (np.arange(n) - 0.5) * h
    zero = np.zeros((n, 1, 1))
    basis = dict(iso_local=zero, iso_P=[f(x).reshape(n, 1, 1)] + [zero] * 5, shape=(n, 1, 1),
                 voxel_size=np.array([h, 1.0, 1.0]))
    return FieldGrid(basis, np.array([x[0], -0.5, -0.5]))


def _magnetic(walk, pool, chi):
    """The walk with its spec stating ``pool`` magnetic (``chi_iso = chi``) and the field tier."""
    spec = walk.spec
    tiers = list(spec.validity.tiers) + ([] if "field" in spec.validity.tiers else ["field"])
    spec = spec.replace(susceptibility={pool: {"chi_iso": chi}},
                        validity=dataclasses.replace(spec.validity, tiers=tiers))
    return dataclasses.replace(walk, spec=spec)


def _walk(geometry, n, T, seed, grid, envelope):
    import dmipy_sim as d  # noqa: F401  (the engine's registry)
    from dmipy_sim.engine.core import _walk_geometry
    from dmipy_sim.replay.compression import certificate_settings
    w = _walk_geometry(n, D0, geometry, T_max=T, seed=seed, tiers="all", field_basis=grid, K=envelope.mode_count(T),
                       envelope=certificate_settings(envelope, None), f_band=envelope.band_hz, require_gpu=False)
    return dataclasses.replace(w, geometry=geometry, spec=geometry.spec, envelope=envelope)


def plane(n, out):
    import dmipy_sim as d
    from dmipy_replay import Envelope
    from dmipy_sim.replay.bank import build_replay_pack
    env = Envelope(name="cross-tier fixture", band_hz=BAND_HZ, G_max=1.0, B0_list=(PLANE_B0,), chi_max=PLANE_CHI,
                   rho2_max=None, max_refocusing_pulses=1)
    grid = _grid(PLANE_L, lambda x: PLANE_G0 * (x - PLANE_L / 2) / (PLANE_B0 * PLANE_CHI))
    w = _walk(d.Box1D(PLANE_L), n, PLANE_T, PLANE_SEED, grid, env)
    pk = build_replay_pack(_magnetic(w, w.spec.pool(w.spec.seeding.pools[0]).name, PLANE_CHI),
                           id="fixture/plane-linear-field", license="CC-BY-4.0", citation="dmipy-replay test fixture",
                           envelope=env, diffusivity=D0, out_path=str(out / "plane_linear_field.rpk"),
                           provenance=dict(substrate="Box1D", L_m=PLANE_L, D0=D0, seed=PLANE_SEED, B0=PLANE_B0,
                                           chi=PLANE_CHI, field_gradient_T_per_m=PLANE_G0))
    return pk


def slab(n, out):
    import dmipy_sim as d
    from dmipy_replay import Envelope
    from dmipy_sim.replay.bank import build_replay_pack
    env = Envelope(name="cross-tier fixture", band_hz=BAND_HZ, G_max=1.0, B0_list=(3.0, 7.0), chi_max=abs(SLAB_CHI),
                   rho2_max=SLAB_RHO2_MAX, max_refocusing_pulses=1)
    grid = _grid(SLAB_L, lambda x: ((x > SLAB_L / 2).astype(float) - 0.5) / 3.0)
    w = _walk(d.PermeableSlab1D(SLAB_L, SLAB_KAPPA), n, SLAB_T, SLAB_SEED, grid, env)
    pk = build_replay_pack(_magnetic(w, "extra", SLAB_CHI), id="fixture/slab-exchange", license="CC-BY-4.0",
                           citation="dmipy-replay test fixture", envelope=env, diffusivity=D0,
                           out_path=str(out / "slab_exchange.rpk"),
                           provenance=dict(substrate="PermeableSlab1D", L_m=SLAB_L, kappa=SLAB_KAPPA, D0=D0,
                                           seed=SLAB_SEED, chi_layer=SLAB_CHI))
    return pk


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-plane", type=int, default=6000)
    ap.add_argument("--n-slab", type=int, default=6000)
    ap.add_argument("--out", type=Path, default=DATA)
    ap.add_argument("--only", choices=("plane", "slab"), default=None)
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    for name, fn, n in (("plane", plane, a.n_plane), ("slab", slab, a.n_slab)):
        if a.only not in (None, name):
            continue
        t0 = time.time()
        pk = fn(n, a.out)
        print(f"{name}: {n} walkers, K={pk.K}, {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
