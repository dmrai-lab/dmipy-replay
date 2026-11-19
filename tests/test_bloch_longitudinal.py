"""The Bloch route's longitudinal model, as its docstring states it: ``M_z`` decays by ``exp(-t / T1)`` and does not
recover toward equilibrium. An ideal 90-180 spin echo reads the same as a propagator that recovers; a stimulated
echo's recall and a repeated excitation train do not (they read the decay-only closed form). A change to the model
changes these tests and the docstring together."""
import numpy as np
import pytest

from dmipy_replay.acquisition.rf import RFEvent
from dmipy_replay.replay.bloch import replay_bloch

H, T1, T2 = 1e-4, 1.0, 0.08


def _rot(flip_deg, axis_deg):
    f, a = np.deg2rad(flip_deg), np.deg2rad(axis_deg)
    u = np.array([np.cos(a), np.sin(a), 0.0])
    K = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
    return np.eye(3) + np.sin(f) * K + (1 - np.cos(f)) * K @ K


def _closed_form(pulses, T, recover):
    """``Mx + i My`` at ``T`` of one spin under hard ``pulses`` ``[(t, flip, axis), ...]``, relaxing between them, with
    or without recovery toward equilibrium."""
    M, t = np.array([0.0, 0.0, 1.0]), 0.0
    def free(M, d):
        e1, e2 = np.exp(-d / T1), np.exp(-d / T2)
        return np.array([M[0] * e2, M[1] * e2, M[2] * e1 + ((1.0 - e1) if recover else 0.0)])
    for ts, f, ax in pulses:
        M = _rot(f, ax) @ free(M, ts - t); t = ts
    M = free(M, T - t)
    return M[0] + 1j * M[1]


def _bloch(pulses, T, **kw):
    n_t = int(round(T / H)) + 1
    rf = [RFEvent(ts, f, "Mz→Mxy", axis_deg=ax) for ts, f, ax in pulses]
    return replay_bloch(np.zeros((4, n_t, 3), np.float32), H, np.zeros((1, n_t, 3)), H, rf, T1=T1, T2=T2, **kw)


def test_a_walker_below_equilibrium_decays_and_does_not_recover():
    M_init = np.zeros((1, 3, 4)); M_init[0, 0] = 0.4; M_init[0, 2] = 0.3
    M, _ = _bloch([], 0.05, M_init=M_init, return_state=True)
    np.testing.assert_allclose(M[0, 2], 0.3 * np.exp(-0.05 / T1), rtol=1e-12)
    np.testing.assert_allclose(M[0, 0], 0.4 * np.exp(-0.05 / T2), rtol=1e-12)


SPIN_ECHO = ([(0.0, 90.0, 90.0), (0.025, 180.0, 0.0)], 0.05)
STIMULATED = ([(0.0, 90.0, 0.0), (0.005, 90.0, 0.0), (0.015, 90.0, 0.0)], 0.02)
TRAIN = ([(k * 0.1, 60.0, 0.0) for k in range(20)], 1.901)


def test_an_ideal_spin_echo_is_exact_under_the_model():
    s = complex(_bloch(*SPIN_ECHO)[0])
    assert abs(s - _closed_form(*SPIN_ECHO, recover=True)) <= 1e-12


@pytest.mark.parametrize("read,omitted", [(STIMULATED, 1.08e-2), (TRAIN, 1.0)])
def test_a_pulse_that_tips_regrown_magnetisation_back_is_outside_the_model(read, omitted):
    s = complex(_bloch(*read)[0])
    assert abs(s - _closed_form(*read, recover=False)) <= 1e-12            # the route is the decay-only model
    ref = _closed_form(*read, recover=True)
    assert abs(s - ref) / abs(ref) == pytest.approx(omitted, rel=1e-2)    # the recovery it omits
