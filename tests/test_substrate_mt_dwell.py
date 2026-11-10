"""The MT bound pool's dwell on a wall (``Wall.mt_dwell``): a wall that binds (``mt_reactivity`` positive on a side)
states its bound pool's mean dwell per side, and only such a wall does; it round-trips through JSON, the validator
holds each side's dwell to that side's reactivity, and a size scale keeps it (a time) while it multiplies the
reactivity (a rate)."""
import dataclasses

import pytest

from dmipy_replay.spec.substrate import (Domain, Dwell, PoolSpec, Seeding, Sided, SpecError, SubstrateSpec, Surface,
                                         Validity, Wall, validate)


def _spec(mt_reactivity, mt_dwell):
    surf = Surface("cylinder", axis=[0.0, 0.0, 1.0],
                   instances={"centers": [[-2e-6, 0.0, 0.0], [2e-6, 0.0, 0.0]], "radii": [1e-6, 0.8e-6]})
    wall = Wall("objects", surf, 1, 0, mt_reactivity=mt_reactivity, mt_dwell=mt_dwell)
    pools = [PoolSpec(0, "extra", None), PoolSpec(1, "intra", None)]
    return SubstrateSpec("test/mt", Domain([-4e-6] * 3, [4e-6] * 3, ["periodic", "periodic", "open"]), pools, [wall],
                         Seeding([0, 1]), Validity(0.8e-6, ["gradient", "relaxation", "surface", "exchange"]))


def test_a_binding_wall_states_its_dwell_and_round_trips():
    spec = _spec(Sided(2e-4, 2e-4), Dwell(0.56e-3, 0.56e-3)).validate()
    again = SubstrateSpec.from_json(spec.to_json())
    assert again == spec and again.walls[0].mt_dwell == Dwell(0.56e-3, 0.56e-3)
    one_side = _spec(Sided(0.0, 2e-4), Dwell(0.0, 1e-3)).validate()          # the outer face binds alone
    assert SubstrateSpec.from_json(one_side.to_json()) == one_side


def test_a_wall_that_binds_nothing_writes_no_dwell():
    spec = _spec(Sided(), None).validate()
    assert "mt_dwell" not in spec.to_dict()["walls"][0]
    assert SubstrateSpec.from_json(spec.to_json()) == spec


@pytest.mark.parametrize("react, dwell, match", [
    (Sided(2e-4, 2e-4), None, "states no mt_dwell"),
    (Sided(), Dwell(1e-3, 1e-3), "binds nothing has no bound pool"),
    (Sided(2e-4, 2e-4), Dwell(1e-3, 0.0), "outside side binds"),
    (Sided(0.0, 2e-4), Dwell(1e-3, 1e-3), "side that binds nothing"),
    (Sided(2e-4, 2e-4), Dwell(float("inf"), 1e-3), "finite number of seconds"),
])
def test_a_dwell_that_does_not_match_the_reactivity_is_refused(react, dwell, match):
    with pytest.raises(SpecError, match=match):
        _spec(react, dwell).validate()


def test_an_unknown_key_of_the_dwell_is_refused():
    d = _spec(Sided(2e-4, 2e-4), Dwell(1e-3, 1e-3)).to_dict()
    d["walls"][0]["mt_dwell"]["mean"] = 1e-3
    with pytest.raises(SpecError, match="unknown keys"):
        validate(d)


def test_scaled_multiplies_the_reactivity_and_keeps_the_dwell():
    spec = _spec(Sided(2e-4, 1e-4), Dwell(0.56e-3, 0.3e-3)).validate()
    out = spec.scaled(2.0)
    assert out.walls[0].mt_reactivity == Sided(4e-4, 2e-4)
    assert out.walls[0].mt_dwell == Dwell(0.56e-3, 0.3e-3)
