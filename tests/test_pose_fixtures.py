"""The pose expansion on the committed fixture packs: the expansion read at a pose is the pack's replay at that pose
within the expansion's own misfit, a single-pose distribution composes to the expansion at that pose, the powder
average is the expansion's zeroth coefficient, and the three backends compute one expansion to float32 rounding.
NumPy is the default; JAX and torch run where they are stated."""
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("scipy")                    # the pose extra

import dmipy_replay as dr                       # noqa: E402
from dmipy_replay.replay.so3 import Distribution, rotation_of  # noqa: E402

DATA = Path(__file__).parent / "data"
R = rotation_of([0.3, 0.4, 0.866], roll=0.7)


@pytest.fixture(scope="module")
def case():
    pk = dr.read_rpk(DATA / "cylinder.rpk")
    T = pk.T_window
    seq = dr.pgse([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]], 0.2 * T, 0.5 * T, bvalues=[1e9, 1e9], TE=0.95 * T, slew_rate=np.inf)
    return pk, seq, pk.pose_response(seq)


def test_the_default_is_numpy_and_the_closed_form(case):
    pk, seq, r = case
    assert r.route == "closed" and r.coeffs.shape[0] == seq.n_meas


def test_the_expansion_at_a_pose_is_the_replay_at_it_within_its_misfit(case):
    pk, seq, r = case
    direct = np.asarray(pk.replay(seq, orientation=R, complex_signal=True, tol=None)).ravel()   # every stored band
    assert np.all(np.abs(r.at(R) - direct) <= np.asarray(r.misfit) + 1e-12)


def test_a_single_pose_distribution_composes_to_the_expansion_at_that_pose(case):
    pk, seq, r = case
    via = np.asarray(pk.replay(seq, orientation=Distribution.pose(R, lmax=r.lmax, nmax=r.nmax), complex_signal=True)).ravel()
    np.testing.assert_allclose(via, r.at(R), rtol=0, atol=1e-12)


def test_the_powder_average_is_the_zeroth_coefficient(case):
    pk, seq, r = case
    np.testing.assert_allclose(r.compose(Distribution.uniform(r.lmax, r.nmax)), r.coeffs[:, 0], rtol=0, atol=1e-15)


@pytest.mark.parametrize("backend,kw", [("jax", {}), ("torch", {"device": "cpu"})])
def test_jax_and_torch_compute_the_numpy_expansion_to_float32_rounding(case, backend, kw):
    pytest.importorskip(backend)
    pk, seq, r = case
    other = pk.pose_response(seq, backend=backend, **kw)
    assert (other.lmax, other.nmax) == (r.lmax, r.nmax)
    assert np.abs(other.coeffs - r.coeffs).max() <= 1e-5 * np.abs(r.coeffs).max()


def test_an_unknown_backend_is_refused_by_name(case):
    pk, seq, _r = case
    with pytest.raises(ValueError, match="'numpy', 'jax' or 'torch'"):
        pk.pose_response(seq, backend="cupy")


def _classes(seq):
    """Three encoding classes of one timing class: ``seq``, its rows at 0.8 of their amplitude, and its rows turned."""
    from dmipy_replay.acquisition.waveforms import rotate_waveform
    scaled = dr.ScannerSequence(G=0.8 * np.asarray(seq.G), dt=seq.dt, rf=seq.rf, readout=seq.readout)
    return [seq, scaled, rotate_waveform(seq, R)]


def test_a_series_class_is_its_closed_form_within_the_two_misfits(case):
    from dmipy_replay.replay import so3
    pk, _seq, _r = case
    T = pk.T_window
    seq = dr.pgse([[0, 0, 1], [1, 0, 0]], 0.2 * T, 0.5 * T, gradient_strengths=[0.05, 0.03], TE=0.95 * T, slew_rate=200.0)
    classes = _classes(seq)
    series = pk.pose_series(classes).responses()
    closed = pk.pose_responses(classes)
    for s, c in zip(series, closed):
        assert s.route == "series" and c.route == "closed"
        ref = so3.rebanded(c.coeffs, c.lmax, c.nmax, s.lmax, s.nmax) if (c.lmax, c.nmax) != (s.lmax, s.nmax) else c.coeffs
        assert np.all(np.linalg.norm(s.coeffs - ref, axis=1) <= s.misfit + c.misfit + 1e-12)


def test_the_jax_twins_are_the_replay_to_float32_rounding(case):
    pytest.importorskip("jax")
    from dmipy_replay.replay.jax_routes import replay_batch_jax, replay_signal_jax
    from dmipy_replay.replay.replay import compile_scheme
    pk, seq, _r = case
    W = compile_scheme(np.asarray(seq.G_eff, np.float64), float(seq.dt), pk.K, T=pk.T_window)
    C, w = np.asarray(pk.position_coeffs), np.asarray(pk.spin_weights)
    ref = np.asarray(pk.replay(seq, complex_signal=True, tol=None)).ravel()
    np.testing.assert_allclose(np.asarray(replay_signal_jax(C, w, W)), ref, atol=1e-5)
    batch = np.asarray(replay_batch_jax(C, w, np.stack([W, 0.5 * W])))
    np.testing.assert_allclose(batch[0], np.abs(ref), atol=1e-5)
    np.testing.assert_allclose(batch[1], np.abs(np.asarray(replay_signal_jax(C, w, 0.5 * W))), atol=1e-6)


def test_the_jax_bloch_propagator_is_the_numpy_one_to_float32_rounding(case):
    pytest.importorskip("jax")
    pk, seq, _r = case
    np.testing.assert_allclose(np.asarray(pk.replay_bloch(seq, backend="jax", b1_scale=0.95)),
                               np.asarray(pk.replay_bloch(seq, b1_scale=0.95)), atol=2e-5)
