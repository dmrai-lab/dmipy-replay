"""The SO(3) representation of a pose response: the basis, and the distributions composed against it (#157)."""
import dmipy_replay.replay.so3 as _r_so3
import numpy as np
import pytest

pytest.importorskip("scipy")                    # the pose extra

from dmipy_analytic import harmonics

from dmipy_replay.replay import so3
from dmipy_replay.replay.fod import FOD


def test_the_basis_is_orthonormal_under_the_quadrature():
    """Peter-Weyl: ``sqrt(2l+1) D^l_{mn}`` are orthonormal in the normalised Haar measure, and the quadrature
    is exact for the band it is built for. Without this the composition is not an inner product."""
    lmax = 4
    R, w, _d, _r = so3.so3_quadrature(lmax)
    A = so3.so3_design(lmax, R)
    G = (A * w[:, None]).T @ A
    assert A.shape[1] == so3.n_so3_coeffs(lmax)
    np.testing.assert_allclose(G, np.eye(A.shape[1]), atol=1e-9)
    np.testing.assert_allclose(w.sum(), 1.0, atol=1e-12)


def test_a_band_limited_response_is_recovered_exactly():
    """A function in the space is recovered by the quadrature projection with no solve, and reproduced at
    rotations off the grid: the projection is exact, not an interpolation."""
    lmax, nmax = 4, 2
    truth = np.random.default_rng(0).normal(size=so3.n_so3_coeffs(lmax, nmax))
    R, w, A = so3.quadrature_design(lmax, nmax)
    c = so3.project(A, w, A @ truth)
    np.testing.assert_allclose(c, truth, atol=1e-10)
    Q = so3.haar_rotations(20, seed=4)
    np.testing.assert_allclose(so3.evaluate(c, lmax, Q, nmax),
                               so3.so3_design(lmax, Q, nmax) @ truth, atol=1e-9)
    # a whole acquisition projects in one product, real or complex
    Y = A @ np.random.default_rng(1).normal(size=(truth.size, 3)) * (1 + 2j)
    np.testing.assert_allclose(so3.project(A, w, Y).shape, (truth.size, 3))


def test_one_pose_composes_to_the_response_at_that_pose():
    """The delta distribution is the consistency check of the whole scheme: composing it must return the
    response evaluated there, so a phantom slot holding a single pose needs no separate code path."""
    lmax, nmax = 6, 3
    c = np.random.default_rng(1).normal(size=so3.n_so3_coeffs(lmax, nmax))
    R0 = so3.haar_rotations(1, seed=5)[0]
    f = so3.delta_coeffs(R0, lmax, nmax)
    np.testing.assert_allclose(f @ c, float(so3.evaluate(c, lmax, R0[None], nmax)[0]), rtol=1e-9)


def test_a_distribution_over_directions_has_no_azimuthal_coefficients():
    """An ODF says where the axis points and nothing about the substrate's spin about it. Its coefficients must
    therefore vanish for every ``n != 0``, which is what makes the roll integral exact and free -- no sampling
    of rolls anywhere."""
    lmax, nmax = 6, 3
    f = so3.axis_density_coeffs(FOD.watson(8.0, mu=(0.3, 0.5, 0.81), lmax=6).coeffs, lmax, nmax)
    idx = so3.so3_index(lmax, nmax)
    off = np.array([abs(v) for (l, m, n), v in zip(idx, f) if n != 0])
    on = np.array([abs(v) for (l, m, n), v in zip(idx, f) if n == 0])
    assert off.max() < 1e-8 < on.max()
    assert abs(f[0] - 1.0) < 1e-9                                    # a density integrates to one
    # and the contract that matters: composing through SO(3) equals the sphere integral of the same ODF
    # against the same response, for any response the band can hold
    lmax_r = 4
    c = np.random.default_rng(2).normal(size=so3.n_so3_coeffs(lmax_r, nmax))
    fod = FOD.watson(2.0, mu=(0.3, 0.5, 0.81), lmax=6)               # broad enough to be positive in band
    f_r = so3.axis_density_coeffs(fod.coeffs, lmax_r, nmax)
    from dmipy_replay.replay.so3 import sphere_quadrature
    dirs, w = sphere_quadrature(24, 48)
    rho = fod.evaluate(dirs)
    assert rho.min() > 0                                             # no clipping, so the two sides agree exactly
    rolls = np.arange(16) * (2 * np.pi / 16)
    R = np.stack([_r_so3.rotation_of(d, roll=r) for d in dirs for r in rolls])
    E = so3.evaluate(c, lmax_r, R, nmax).reshape(len(dirs), len(rolls))
    direct = float((w * rho * E.mean(axis=1)).sum() / (w * rho).sum())   # roll-averaged, ODF-weighted
    np.testing.assert_allclose(f_r @ c, direct, rtol=1e-6, atol=1e-9)


def test_watson_and_an_isotropic_bingham_are_the_same_distribution():
    """A Bingham with equal dispersions is a Watson: the anisotropic form must contain the isotropic one, or
    the two dispersion parameters do not mean what they claim."""
    lmax, nmax = 6, 2
    frame = _r_so3.rotation_of((0.0, 0.0, 1.0))
    w8 = so3.watson_coeffs(8.0, mu=(0.0, 0.0, 1.0), lmax=lmax, nmax=nmax)
    b8 = so3.bingham_coeffs(frame, (8.0, 8.0), lmax=lmax, nmax=nmax)   # equal concentrations: the same cone
    np.testing.assert_allclose(b8, w8, atol=1e-8)
    # and an anisotropic one is genuinely different: a fan in one plane, not a cone
    fan = so3.bingham_coeffs(frame, (1.0, 40.0), lmax=lmax, nmax=nmax)
    assert np.abs(fan - w8).max() > 1e-2
    assert abs(fan[0] - 1.0) < 1e-9


def test_the_azimuthal_band_is_measured_not_asserted():
    """``energy`` is what chooses ``nmax``: a response with structure only in the pose direction puts all its
    energy at ``n = 0``, and one with azimuthal structure does not."""
    lmax, nmax = 4, 2
    idx = so3.so3_index(lmax, nmax)
    axial = np.array([1.0 if n == 0 else 0.0 for (_l, _m, n) in idx])
    per_l, per_n = so3.energy(axial, lmax, nmax)
    assert per_n[0] > 0 and np.allclose(per_n[1:], 0.0)
    assert per_l.sum() == pytest.approx(per_n.sum())
    skew = np.array([1.0 if n == 1 else 0.0 for (_l, _m, n) in idx])
    _pl, pn = so3.energy(skew, lmax, nmax)
    assert pn[1] > 0 and pn[0] == 0.0


def test_two_real_wigner_blocks_multiply_through_the_coupling_tables():
    """``M^l1[m,n] M^l2[m',n'] = sum_L K_L[(m,m'),M] M^L[M,N] conj(K_L[(n,n'),N])`` with ``K_L`` from the complex
    Clebsch-Gordan coefficients and the real-from-complex unitaries: the product of two expansions on SO(3) is a
    contraction with these tables, the whole basis of composing a field with a gradient (#197 step 3)."""
    for (l1, l2) in [(1, 1), (2, 3), (4, 2), (6, 5), (9, 5)]:
        C = np.concatenate([so3.clebsch_gordan(l1, l2, L) for L in range(abs(l1 - l2), l1 + l2 + 1)], axis=1)
        np.testing.assert_allclose(C.T @ C, np.eye(C.shape[1]), atol=1e-10)          # CG: orthogonal
        np.testing.assert_allclose(C @ C.T, np.eye(C.shape[0]), atol=1e-10)
    Rs = so3.haar_rotations(3, 11)
    for (l1, l2) in [(3, 1), (4, 6), (9, 5)]:
        B1, B2 = harmonics.wigner_blocks(l1, Rs)[l1], harmonics.wigner_blocks(l2, Rs)[l2]
        BL = harmonics.wigner_blocks(l1 + l2, Rs)
        K = so3.coupling(l1, l2)
        for r in range(Rs.shape[0]):
            prod = sum(K[L] @ BL[L][r] @ K[L].conj().T for L in K)
            np.testing.assert_allclose(prod, np.kron(B1[r], B2[r]), atol=1e-12)
