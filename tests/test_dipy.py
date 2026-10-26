"""The DIPY connector against DIPY itself (a test oracle only; no shipped module imports it): the gradient table's
units and timing, ``multi_tensor``'s angle and fraction conventions, and the refusals of what a table cannot say."""
from pathlib import Path

import numpy as np
import pytest

import dmipy_replay as dr
from dmipy_replay.dipy import add_noise, multi_pack, sticks_of
from dmipy_replay.sequences.btable import from_btable

dipy_gt = pytest.importorskip("dipy.core.gradients")
dipy_voxel = pytest.importorskip("dipy.sims.voxel")

PACK = Path(__file__).parent / "data" / "cylinder.rpk"
BVALS = np.array([0.0, 5.0, 1000.0, 1000.0, 1000.0, 2000.0, 3000.0])
BVECS = np.array([[0, 0, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1],
                  [1, 1, 0] / np.sqrt(2), [0.6, 0, 0.8]], dtype=float)
DELTA, BIG_DELTA = 0.006, 0.02


@pytest.fixture(scope="module")
def pack():
    return dr.read_rpk(PACK)


def gtab(**kw):
    return dipy_gt.gradient_table(BVALS, bvecs=BVECS, **kw)


def test_the_table_plays_as_the_si_pgse(pack):
    """A DIPY table (s/mm^2, its small/big delta) is the PGSE of the same numbers in SI; the b <= 50 volume is a b0."""
    S, _ = multi_pack(gtab(big_delta=BIG_DELTA, small_delta=DELTA), [pack], angles=[(0, 0)], fractions=[100])
    b = np.where(BVALS > 50, BVALS, 0.0) * 1e6
    dirs = np.where((BVALS > 50)[:, None], BVECS, [1.0, 0, 0])
    ref = pack.replay(dr.pgse(dirs, DELTA, BIG_DELTA, bvalues=b), orientation=[0, 0, 1])
    np.testing.assert_array_equal(S, ref)
    assert S[0] == S[1] == 1.0


def test_sticks_are_dipys(pack):
    angles = [(0, 0), (90, 0), (60, 45)]
    _, sticks_dipy = dipy_voxel.multi_tensor(gtab(), np.array([[1.7e-3, 3e-4, 3e-4]] * 3), angles=angles,
                                             fractions=[30, 30, 40], snr=None)
    np.testing.assert_allclose(sticks_of(angles), sticks_dipy, atol=1e-15)
    np.testing.assert_allclose(sticks_of(sticks_dipy), sticks_dipy, atol=1e-15)


def test_a_stick_is_the_packs_axis_and_fractions_are_percent(pack):
    g = gtab(big_delta=BIG_DELTA, small_delta=DELTA)
    S, _ = multi_pack(g, [pack, pack], angles=[(90, 0), (0, 0)], fractions=[25, 75], S0=100)
    seq = from_btable(BVALS, BVECS, delta=DELTA, Delta=BIG_DELTA)
    ref = 100 * (0.25 * pack.replay(seq, orientation=[1, 0, 0]) + 0.75 * pack.replay(seq, orientation=[0, 0, 1]))
    np.testing.assert_allclose(S, ref, rtol=1e-13)


def test_rician_noise_is_dipys_formula():
    s = np.linspace(0.1, 1.0, 50)
    rng = np.random.default_rng(3)
    n1, n2 = rng.normal(0, 0.05, s.shape), rng.normal(0, 0.05, s.shape)
    np.testing.assert_array_equal(add_noise(s, 20, 1.0, rng=3), np.sqrt((s + n1) ** 2 + n2 ** 2))
    assert add_noise(s, None, 1.0) is s


def test_a_table_without_its_timing_is_refused(pack):
    with pytest.raises(ValueError, match="carries no timing"):
        multi_pack(gtab(), [pack], angles=[(0, 0)], fractions=[100])


def test_a_tissue_without_an_echo_time_is_refused(pack):
    with pytest.raises(ValueError, match="state TE"):
        multi_pack(gtab(big_delta=BIG_DELTA, small_delta=DELTA), [pack], angles=[(0, 0)], fractions=[100],
                   tissue=dr.Tissue(T2={"intra": 0.08}))


def test_fractions_that_are_not_percent_are_refused(pack):
    with pytest.raises(ValueError, match="percent"):
        multi_pack(gtab(big_delta=BIG_DELTA, small_delta=DELTA), [pack, pack], angles=[(0, 0), (90, 0)],
                   fractions=[0.5, 0.5])


def test_a_direction_that_is_not_unit_is_refused():
    with pytest.raises(ValueError, match="not a unit vector"):
        from_btable([1000.0], [[0.5, 0, 0]], delta=DELTA, Delta=BIG_DELTA)


def test_fsl_bvecs_are_read_either_way():
    a = from_btable(BVALS, BVECS, delta=DELTA, Delta=BIG_DELTA)
    b = from_btable(BVALS, BVECS.T, delta=DELTA, Delta=BIG_DELTA)
    np.testing.assert_array_equal(a.G, b.G)
