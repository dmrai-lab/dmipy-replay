"""The native ``.seq`` reader against dmipy-sim's pypulseq-based one (the ``.seq`` files and readings of ``data/seq``, recorded with dmipy-sim and frozen): the same
gradient on the same grid, the same readout, the same RF instants; and every pulse's flip as pypulseq computes it
from the file's own samples, which sim's reader does not read (it takes 90 / 180 for an excitation / refocusing)."""
import json
from pathlib import Path

import numpy as np
import pytest

from dmipy_replay.sequences.pulseq import from_pulseq, read_seq, decompress_shape

DATA = Path(__file__).parent / "data" / "seq"
REF = np.load(DATA / "reference.npz")
INDEX = json.loads((DATA / "reference.json").read_text())
NAMES = sorted(INDEX)


@pytest.mark.parametrize("name", NAMES)
def test_gradient_grid_readout(name):
    s = from_pulseq(DATA / f"{name}.seq")
    assert s.dt == pytest.approx(float(REF[f"{name}/dt"]), rel=1e-12)
    np.testing.assert_array_equal(np.asarray(s.readout), REF[f"{name}/readout"])
    G, G_ref = np.asarray(s.G, np.float64), REF[f"{name}/G"]
    assert G.shape == G_ref.shape
    np.testing.assert_allclose(G, G_ref, rtol=0, atol=1e-6 * max(np.abs(G_ref).max(), 1e-12))


@pytest.mark.parametrize("name", NAMES)
def test_rf_instants_and_roles(name):
    s = from_pulseq(DATA / f"{name}.seq")
    np.testing.assert_allclose([e.t_s for e in s.rf], REF[f"{name}/rf_t"], atol=1e-9)
    assert [e.label for e in s.rf] == INDEX[name]["labels"]


@pytest.mark.parametrize("name", [n for n in NAMES if n.startswith("foreign")])
def test_flips_are_the_files(name):
    """A foreign file's pulses carry their played B1: the flip is the nutation the file's samples play, pypulseq's
    sum of ``|signal| dt`` (an event's flip is ``gamma int |B1| dt``; a sinc's net on-resonance rotation is less, and
    the vector-Bloch route plays the lobes with their signs)."""
    s = from_pulseq(DATA / f"{name}.seq")
    assert all(e.envelope is not None for e in s.rf)
    np.testing.assert_allclose([e.flip_deg for e in s.rf], REF[f"{name}/nutation_pypulseq"], rtol=1e-6)


def test_refocusing_phase_is_the_axis():
    s = from_pulseq(DATA / "foreign_sinc_se.seq")
    assert s.rf[1].axis_deg == pytest.approx(90.0, abs=1e-3)              # the file writes the phase to its precision


def test_shape_decompression():
    """Pulseq's run-length code of the derivative: a value written twice is followed by its extra repeat count."""
    x = np.array([0.0, 0.1, 0.2, 0.2, 0.2, 0.2, 0.1, 0.0, 0.0])
    d = np.diff(np.concatenate([[0.0], x]))
    np.testing.assert_allclose(decompress_shape(9, [0.0, 0.1, 0.1, 0.0, 0.0, 0.0, 1.0, -0.1, -0.1, 0.0, 0.0]), x, atol=1e-12)
    np.testing.assert_allclose(decompress_shape(9, d), d)          # as many values as samples: stored as is


def test_version_is_read():
    f = read_seq(DATA / "foreign_sinc_se.seq")
    assert f.version[:2] >= (1, 4)
