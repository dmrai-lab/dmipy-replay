"""A pack that states no floor is held to a computed one and the plan says so; a certificate that cannot split its error
is refused by name (the reader holds no silent default in the certified path)."""
import numpy as np
import pytest

from dmipy_replay.replay import planning
from dmipy_replay.replay.replay import truncation_budget


def test_the_floor_names_where_it_came_from():
    stated, why = planning.floor_of({"fidelity": {"floor_max": 0.02}}, 10_000)
    assert stated == 0.02 and why == planning.FLOOR_CERTIFICATE
    for meta in ({}, {"fidelity": {}}, {"fidelity": {"floor_max": 0.0}}, {"fidelity": {"floor_max": None}}):
        computed, why = planning.floor_of(meta, 10_000)
        assert computed == pytest.approx(0.01) and why == planning.FLOOR_N_WALKERS


def test_a_certificate_with_err_max_and_no_err_truncation_is_refused_by_name():
    with pytest.raises(ValueError, match="err_truncation"):
        truncation_budget({"fidelity": {"target_floor": 0.05, "err_max": 0.01}}, floor=0.05, floor_source="x")
    ok = truncation_budget({"fidelity": {"target_floor": 0.05, "err_max": 0.01, "err_truncation": 0.004}}, floor=0.05, floor_source="x")
    assert np.isfinite(ok["budget"])
