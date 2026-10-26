"""The rules the code is held to, checked by reading it (the reader-side half of dmipy-sim's test_api_surface): one
acquisition object and one RF dialect, scanner numbers read from the cited catalogue only, a replay setting stated as
tissue / orientation / scanner, one basis and the refusal of every pack that predates it."""
import ast
import inspect
import re
from pathlib import Path

import numpy as np
import pytest

import dmipy_replay as dr

SRC = Path(dr.__file__).parent
RF_DICT_BOUNDARY = {"dmipy_replay.acquisition.rf"}
SCANNER_SOURCE = {"dmipy_replay.acquisition.scanner_constants.SCANNER_CONSTANTS"}
SCANNER_VIEWS = {"dmipy_replay.acquisition.scanners.SCANNERS"}


def _modules():
    for py in sorted(SRC.rglob("*.py")):
        yield py, ".".join(py.relative_to(SRC.parent).with_suffix("").parts)


def test_rf_events_are_objects_outside_the_boundary():
    """A pulse is an ``RFEvent``: read as a dict (``['t_s']``) or spelled as one (``{'t_s': ...}``) only inside
    ``acquisition.rf``, the object's own serialised record."""
    read = re.compile(r"""(?:\.get\(|\[)\s*['"](t_s|b1_envelope)['"]""")
    spelled = re.compile(r"""['"]t_s['"]\s*:|\bdict\(\s*t_s\s*=""")
    strays = {m for py, m in _modules() if m not in RF_DICT_BOUNDARY
              and (read.search(py.read_text()) or spelled.search(py.read_text()))}
    assert not strays, f"RF events as dicts outside {sorted(RF_DICT_BOUNDARY)}: {sorted(strays)}"


def test_no_acquisition_side_channel():
    """No callable takes an acquisition AND its RF schedule / echo / refocus time beside it."""
    acq, side = {"waveform", "wf", "seq", "sequence", "acq"}, {"rf_events", "refocus_time", "echo_steps"}
    found = set()
    for py, mod in _modules():
        for node in ast.walk(ast.parse(py.read_text())):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                params = {a.arg for a in node.args.args + node.args.kwonlyargs}
                if params & acq and params & side:
                    found.add(f"{mod}.{node.name}")
    assert not found, f"side channels: {sorted(found)}"


def test_scanner_numbers_live_in_one_place():
    """A module-scope catalogue name is the cited JSON or a view derived from it, and a view carries no number."""
    pat = re.compile(r"^[A-Z_]*(SCANNER|SYSTEMS)[A-Z_]*$")
    found, literal = set(), set()
    for py, mod in _modules():
        tree = ast.parse(py.read_text())
        nested = {id(n) for scope in ast.walk(tree) if isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                  for n in ast.walk(scope) if n is not scope}
        for node in ast.walk(tree):
            if id(node) not in nested and isinstance(node, (ast.Assign, ast.AnnAssign)):
                for t in (node.targets if isinstance(node, ast.Assign) else [node.target]):
                    if isinstance(t, ast.Name) and pat.match(t.id):
                        found.add(f"{mod}.{t.id}")
                        if any(isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(n.value, bool)
                               for n in ast.walk(node.value)):
                            literal.add(f"{mod}.{t.id}")
    assert found == SCANNER_SOURCE | SCANNER_VIEWS, sorted(found)
    assert not literal, sorted(literal)


def test_replay_knobs_are_tissue_orientation_and_scanner():
    """No replay entry point takes a flat physical value; every one takes ``scanner``."""
    flat = {"T2", "T1", "rho2", "rho1", "D", "B0", "b0_dir", "chi_iso", "chi_aniso", "T2_s", "T1_s", "B0_T"}
    for fn in (dr.ReplayPack.replay, dr.ReplayPack.walker_signals, dr.ReplayPack.walker_phases, dr.ReplayPack.replay_bloch):
        params = inspect.signature(fn).parameters
        assert not flat & set(params), f"{fn.__qualname__}: {sorted(flat & set(params))}"
        assert "scanner" in params and params["tissue"].default is None, fn.__qualname__


def test_a_bare_gradient_array_is_refused():
    p = dr.read_rpk(Path(__file__).parent / "data" / "cylinder.rpk")
    seq = dr.pgse([[1, 0, 0]], 0.006, 0.02, bvalues=[1e9])
    with pytest.raises(TypeError, match="ScannerSequence"):
        p.replay(np.asarray(seq.G))


@pytest.mark.parametrize("meta,key", [
    (dict(walk_params=dict(n_t=11, dt_traj=1e-3, block=dict(h=1e-3))), "walk_params.n_t"),
    (dict(walk_params=dict(block=dict(h=1e-3)), compression=dict(basis="continuous")), "compression.basis"),
    (dict(walk_params=dict(block=dict(h=1e-3), cell_size=1e-5)), "walk_params.cell_size"),
    (dict(walk_params=dict(block=dict(h=1e-3)), compression=dict(method="identity")), "compression.method = identity"),
    (dict(walk_params=dict(T_max=0.01)), "no walk_params.block"),
    ({"walk_params": {"block": {}, "qv": {"field_increments": [0.0]}},
      "compression": {"channels": {"susc_field_basis": {"channels": ["iso_local", "iso_P_xx", "iso_P_yy", "iso_P_zz"]}}}},
     "compression.channels.susc_field_basis carrying iso_P_zz"),
])
def test_a_pack_that_predates_the_one_basis_is_refused_by_name(meta, key):
    from dmipy_replay.replay.replay import _refuse_predating
    with pytest.raises(ValueError, match=r"This pack carries `" + re.escape(key) + r"`.*A pack is never converted"):
        _refuse_predating(meta, "old.rpk")


def test_a_knob_the_pack_does_not_carry_is_refused():
    """The cylinder walk has no field channel: a susceptibility is refused, never returned as a field-free signal."""
    p = dr.read_rpk(Path(__file__).parent / "data" / "cylinder.rpk")
    with pytest.raises(ValueError):
        p.replay(dr.gre(0.006), tissue=dr.Tissue(chi_iso=-1e-7), scanner=3.0)


def test_the_field_carries_twelve_components():
    from dmipy_replay.replay import field as f
    assert len(f.CHANNEL_NAMES) == 12 and "iso_P_zz" not in f.CHANNEL_NAMES and f.ISO_CHANNEL_NAMES == f.CHANNEL_NAMES[:6]
    rng = np.random.default_rng(0)
    c, H = rng.normal(size=(5, 12)), [0.3, -0.5, 0.8]
    P, q = f.iso_P6(c), f.q_of_H(H)
    np.testing.assert_allclose(P[:, 2], 3 * c[:, 0] - c[:, 1] - c[:, 2])
    np.testing.assert_allclose(f.field_terms(c, H)[0], c[:, 0] - P @ q, atol=1e-12)
    np.testing.assert_allclose(f.field_terms(c, H)[1], c[:, 6:12] @ q)
    assert f.field_terms(c[:, :6], H)[1] is None
