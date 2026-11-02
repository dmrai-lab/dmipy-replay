"""The conversion of the harmonic conventions in circulation into the basis the format requires (RPH.md 4.1), checked
against DIPY as the oracle.

The basis is dmipy-analytic's (`dmipy_analytic.harmonics.real_sh`, pinned to DIPY's non-legacy `tournier` there), and
`FOD.from_sh` converts the conventions in circulation into it: the composition's reduction to a distribution over
directions holds only in an orthonormal basis, and the error a non-orthonormal one introduces vanishes exactly where
the gradient is parallel to the field -- the one configuration a cursory check would use.
"""
import numpy as np
import numpy.testing as npt
import pytest

pytest.importorskip("scipy")                    # the pose extra

from dmipy_replay.replay.fod import FOD
from dmipy_analytic.harmonics import real_sh, sh_block

from dmipy_replay.replay.so3 import sphere_quadrature

LMAX = 8


def _dipy(name, lmax, dirs):
    # skip rather than error where dipy is absent: it is a dev-extra, test-only dependency, so a
    # lean install runs the rest of the suite instead of reporting failures it cannot fix
    pytest.importorskip("dipy", reason="dipy (dev extra) pins the normative SH basis")
    from dipy.reconst.shm import real_sh_descoteaux, real_sh_tournier
    th = np.arccos(np.clip(dirs[:, 2], -1, 1)); ph = np.arctan2(dirs[:, 1], dirs[:, 0])
    if name == "mrtrix":
        return real_sh_tournier(lmax, th, ph, legacy=True)[0]
    if name == "tournier":
        return real_sh_tournier(lmax, th, ph, legacy=False)[0]
    return real_sh_descoteaux(lmax, th, ph, legacy=False)[0]


@pytest.mark.parametrize("basis,legacy", [("tournier07", True), ("tournier07", False),
                                          ("descoteaux07", False)])
def test_from_sh_converts_a_named_basis_into_the_required_one(basis, legacy):
    """A density expanded in a named convention, converted, must equal its projection onto the
    required basis -- the conversion checked as a change of basis rather than as a formula."""
    name = {("tournier07", True): "mrtrix", ("tournier07", False): "tournier",
            ("descoteaux07", False): "descoteaux"}[(basis, legacy)]
    dirs, w = sphere_quadrature(48, 96)
    Yo, Yn = real_sh(LMAX, dirs), _dipy(name, LMAX, dirs)
    rng = np.random.default_rng(0)
    c_req = np.zeros(Yo.shape[1]); c_req[0] = 1.0 / np.sqrt(4 * np.pi)   # a unit-integral density
    c_req[1:] = rng.standard_normal(Yo.shape[1] - 1) * 0.01
    f = Yo @ c_req
    # the same density's coefficients in the source convention, by its own least-squares projection
    c_src = np.linalg.lstsq(Yn * np.sqrt(w)[:, None], f * np.sqrt(w), rcond=None)[0]
    got = FOD.from_sh(c_src, basis=basis, legacy=legacy, normalize=False).coeffs
    npt.assert_allclose(got, c_req, atol=1e-8)


def test_an_unconverted_legacy_fod_is_wrong_where_a_cursory_check_would_not_look():
    """Why the conversion is enforced rather than documented. Reading legacy coefficients as if they
    were the required basis leaves the ``m = 0`` terms alone, so the error vanishes for a density
    that is zonal about the axis one happens to test and appears only away from it."""
    lmax = 4
    rng = np.random.default_rng(1)
    c = np.zeros(real_sh(lmax, np.zeros((1, 3)) + np.array([0.0, 0.0, 1.0])).shape[1])
    c[0] = 1.0 / np.sqrt(4 * np.pi)
    c[1:] = rng.standard_normal(c.size - 1) * 0.02
    wrong = FOD.native(c, normalize=True)                            # read as if already required
    right = FOD.from_sh(c, basis="tournier07", legacy=True, normalize=True)
    m0 = [sh_block(l).start + l for l in range(0, lmax + 1, 2)]
    npt.assert_allclose(wrong.coeffs[m0], right.coeffs[m0], rtol=1e-12)   # m = 0 identical
    off = np.setdiff1d(np.arange(c.size), m0)
    assert np.abs(wrong.coeffs[off] - right.coeffs[off]).max() > 1e-3     # everything else is not
