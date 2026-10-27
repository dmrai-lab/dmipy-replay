"""A b-value table as the toolkits write it plays as the PGSE of its numbers in SI, and what a table cannot say is
refused rather than defaulted."""
from pathlib import Path

import numpy as np
import pytest

import dmipy_replay as dr

BVALS = np.array([0.0, 5.0, 1000.0, 1000.0, 2000.0, 3000.0])
BVECS = np.array([[0, 0, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0], [1, 1, 0] / np.sqrt(2), [0.6, 0, 0.8]], dtype=float)
DELTA, BIG_DELTA = 0.006, 0.02


def test_the_table_plays_as_the_si_pgse():
    p = dr.read_rpk(Path(__file__).parent / "data" / "cylinder.rpk")
    S = p.replay(dr.from_btable(BVALS, BVECS, delta=DELTA, Delta=BIG_DELTA))
    dirs = np.where((BVALS > 50)[:, None], BVECS, [1.0, 0, 0])
    ref = p.replay(dr.pgse(dirs, DELTA, BIG_DELTA, bvalues=np.where(BVALS > 50, BVALS, 0.0) * 1e6))
    np.testing.assert_array_equal(S, ref)
    assert S[0] == S[1] == 1.0                       # b <= 50 s/mm^2 is a b0


def test_a_table_without_its_timing_is_refused():
    with pytest.raises(ValueError, match="carries no timing"):
        dr.from_btable(BVALS, BVECS, delta=None, Delta=BIG_DELTA)


def test_a_direction_that_is_not_unit_is_refused():
    with pytest.raises(ValueError, match="not a unit vector"):
        dr.from_btable([1000.0], [[0.5, 0, 0]], delta=DELTA, Delta=BIG_DELTA)


def test_fsl_bvecs_are_read_either_way():
    a = dr.from_btable(BVALS, BVECS, delta=DELTA, Delta=BIG_DELTA)
    np.testing.assert_array_equal(a.G, dr.from_btable(BVALS, BVECS.T, delta=DELTA, Delta=BIG_DELTA).G)
