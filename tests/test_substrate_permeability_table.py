"""A wall's permeability per instance and over the walk's time (``Wall.permeability_table``): it round-trips through
JSON, the validator holds it to its shape and to the wall's ``permeability`` (the largest value of each direction),
and a size scale multiplies its values as rates and leaves its breakpoints, which are times."""
import dataclasses

import pytest

from dmipy_replay.spec.substrate import (Directional, Domain, PermeabilityTable, PoolSpec, Seeding, SpecError,
                                         SubstrateSpec, Surface, Validity, Wall)


def _spec(table, permeability=None):
    surf = Surface("cylinder", axis=[0.0, 0.0, 1.0],
                   instances={"centers": [[-2e-6, 0.0, 0.0], [2e-6, 0.0, 0.0], [0.0, 3e-6, 0.0]],
                              "radii": [1e-6, 0.8e-6, 1.2e-6]})
    if permeability is None:
        permeability = Directional(max(max(r) for r in table.in_to_out), max(max(r) for r in table.out_to_in))
    wall = Wall("objects", surf, 1, 0, permeability, permeability_table=table)
    pools = [PoolSpec(0, "extra", None), PoolSpec(1, "intra", None)]
    return SubstrateSpec("test/packed", Domain([-4e-6] * 3, [4e-6] * 3, ["periodic", "periodic", "open"]), pools, [wall],
                         Seeding([0, 1]), Validity(0.8e-6, ["gradient", "relaxation", "surface", "exchange"]))


TABLE = PermeabilityTable([0.02, 0.05], [[1e-5, 2e-5, 0.0], [3e-5, 3e-5, 3e-5], [0.0, 0.0, 0.0]],
                          [[1e-5, 2e-5, 0.0], [3e-5, 3e-5, 3e-5], [0.0, 0.0, 0.0]])


def test_the_table_round_trips_and_validates():
    spec = _spec(TABLE).validate()
    again = SubstrateSpec.from_json(spec.to_json())
    assert again == spec and again.walls[0].permeability_table == TABLE
    assert again.walls[0].permeability == Directional(3e-5, 3e-5)
    constant = _spec(PermeabilityTable([], [[1e-5, 2e-5, 0.0]], [[1e-5, 2e-5, 0.0]])).validate()   # per instance only
    assert SubstrateSpec.from_json(constant.to_json()) == constant


def test_a_wall_without_a_table_writes_none():
    spec = dataclasses.replace(_spec(TABLE), walls=[dataclasses.replace(_spec(TABLE).walls[0], permeability_table=None)])
    assert "permeability_table" not in spec.to_dict()["walls"][0]
    assert SubstrateSpec.from_json(spec.to_json()) == spec


@pytest.mark.parametrize("table, permeability, match", [
    (PermeabilityTable([0.05, 0.02], TABLE.in_to_out, TABLE.out_to_in), None, "strictly increasing"),
    (PermeabilityTable([-0.01, 0.02], TABLE.in_to_out, TABLE.out_to_in), None, "strictly increasing"),
    (PermeabilityTable([0.02], TABLE.in_to_out, TABLE.out_to_in), None, "one row per interval"),
    (PermeabilityTable([0.02, 0.05], [[1e-5, 2e-5], [3e-5] * 2, [0.0] * 2], TABLE.out_to_in), None, "one value per instance"),
    (PermeabilityTable([0.02, 0.05], [[1e-5, -2e-5, 0.0], [3e-5] * 3, [0.0] * 3], TABLE.out_to_in), None, ">= 0"),
    (TABLE, Directional(2e-5, 3e-5), "largest value"),
])
def test_an_ill_formed_table_is_refused(table, permeability, match):
    if permeability is None:
        permeability = Directional(max(max(r) for r in table.in_to_out), max(max(r) for r in table.out_to_in))
    with pytest.raises(SpecError, match=match):
        _spec(table, permeability).validate()


def test_scaled_multiplies_the_rates_and_keeps_the_times():
    spec = _spec(TABLE).validate()
    out = spec.scaled(2.0)
    t = out.walls[0].permeability_table
    assert t.breakpoints == TABLE.breakpoints
    assert t.in_to_out == [[2.0 * v for v in row] for row in TABLE.in_to_out]
    assert t.out_to_in == [[2.0 * v for v in row] for row in TABLE.out_to_in]
    assert out.walls[0].permeability == Directional(6e-5, 6e-5)
