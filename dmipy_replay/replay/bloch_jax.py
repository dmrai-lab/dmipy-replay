"""The vector-Bloch replay of a walk kept at its block ends as a jitted ``lax.scan`` (:func:`replay_bloch_jax`): the
JAX propagator ``ReplayPack.replay_bloch(backend="jax")`` runs, the numpy reference being
:func:`dmipy_replay.replay.bloch.replay_bloch`. JAX is the ``jax`` extra, imported when the propagator runs."""
import functools

import numpy as np

from .._extras import need
from .bloch import _bloch_replay_output, _bloch_replay_terms, _rf_increment, _BLOCH_PHASE_TABLE_BYTES

__all__ = ["replay_bloch_jax"]


def replay_bloch_jax(trajectory, h, G, dt_wf, rf_events, *,
                     T2=None, T1=None, comp_traj=None,
                     T2_per_comp=None, T1_per_comp=None,
                     extra_phase_per_step=None,
                     dlog_boundary_unit=None, surface_relaxivity=None, D=None,
                     echo_steps=None, echo_per_walker=False,
                     weights=None, return_walker_signals=False,
                     b1_scale=None, slice_offsets=None, slice_gradient=0.0,
                     bound_frac=None, T2_bound=None, T1_bound=None,
                     off_resonance_bound=0.0, phase_table_bytes=_BLOCH_PHASE_TABLE_BYTES,
                     t0=None, M_init=None, return_state=False):
    """:func:`~dmipy_replay.replay.bloch.replay_bloch` as a jitted ``lax.scan`` over the pieces of the walk, vectorised over measurements.

    Same arguments, same per-piece operator and same outputs as the numpy reference. Arithmetic is float32 on
    the device, so signals agree with the reference to about 1e-5 of the phase scale. Each measurement needs
    an ``(n_pieces, n_w)`` float32 phase table on the device; measurements are batched so that the tables of one
    batch stay within ``phase_table_bytes``, and a single measurement whose table exceeds it runs alone.
    """
    jax = need("jax")
    jnp = jax.numpy
    tm = _bloch_replay_terms(trajectory, h, G, dt_wf, rf_events, T2=T2, T1=T1,
                             comp_traj=comp_traj, T2_per_comp=T2_per_comp, T1_per_comp=T1_per_comp,
                             extra_phase_per_step=extra_phase_per_step,
                             dlog_boundary_unit=dlog_boundary_unit,
                             surface_relaxivity=surface_relaxivity, D=D, b1_scale=b1_scale,
                             slice_offsets=slice_offsets, slice_gradient=slice_gradient,
                             bound_frac=bound_frac, T2_bound=T2_bound, T1_bound=T1_bound,
                             off_resonance_bound=off_resonance_bound, weights=weights, echo_steps=echo_steps, t0=t0)
    n_w, n_t, _ = trajectory.shape
    n_p = tm.k.size
    f32 = jnp.float32
    traj = jnp.asarray(trajectory, f32)
    zeros_p = np.zeros((n_p, 1))
    arrays = (traj[:, jnp.asarray(tm.k), :], traj[:, jnp.asarray(tm.k + 1), :],           # (n_w, n_p, 3) each
              jnp.asarray(tm.flips, f32), jnp.asarray(tm.axes, f32), jnp.asarray(tm.b1, f32),
              jnp.asarray(tm.dphi_extra if tm.dphi_extra is not None else zeros_p, f32),
              jnp.asarray(tm.surf if tm.surf is not None else zeros_p + 1.0, f32),
              jnp.asarray(tm.E2, f32), jnp.asarray(tm.E1, f32),
              jnp.asarray(tm.weights if tm.weights is not None else np.full(n_w, 1.0 / n_w), f32))
    n_meas = tm.w_lo.shape[0]
    per_batch = max(1, int(phase_table_bytes // (4 * n_p * n_w)))
    W = jnp.asarray(np.stack([tm.w_lo, tm.w_hi], axis=1), f32)                        # (n_meas, 2, n_p, 3)
    if M_init is None:
        M0 = np.zeros((n_meas, 3, n_w), np.float32); M0[:, 2, :] = 1.0
    else:
        M0 = np.asarray(M_init, np.float32)
    M0 = jnp.asarray(M0)
    scan = _bloch_scan_batch()
    parts = [scan(W[a:a + per_batch], M0[a:a + per_batch], *arrays, echo_idx=tm.echo_pieces, per_walker=bool(echo_per_walker))
             for a in range(0, n_meas, per_batch)]
    M_all = np.concatenate([np.asarray(p[0], np.float64) for p in parts])
    rec = np.concatenate([np.asarray(p[1], np.complex128) for p in parts])
    last = np.concatenate([np.asarray(p[2], np.complex128) for p in parts])
    echo_out = None if tm.echo_pieces is None else rec
    out = _bloch_replay_output(last, M_all[-1], echo_out, echo_steps, return_walker_signals)
    return (M_all, out) if return_state else out


@functools.lru_cache(maxsize=None)
def _bloch_scan_batch():
    """The jitted batch of measurements of :func:`replay_bloch_jax` (below), built once per process: module state, so
    the executable is reused by every call of the same shapes."""
    import jax
    import jax.numpy as jnp

    @functools.partial(jax.jit, static_argnames=("echo_idx", "per_walker"))
    def scan_batch(W_b, M0_b, r_lo, r_hi, flips, axes, b1, dphi_extra, surf, E2, E1, wn, *, echo_idx, per_walker):
        """One batch of measurements of :func:`replay_bloch_jax`: ``vmap`` over ``W_b`` ``(n_b, 2, n_p, 3)`` (the
        piece weights on the two bounding block ends) and ``M0_b`` ``(n_b, 3, n_w)`` (the state every walker starts in)
        of a ``lax.scan`` over the pieces. Module-level and jitted on array arguments, so the executable is reused
        by every call of the same shapes."""
        n_w = r_lo.shape[0]
        n_ev = flips.shape[1]

        def readout(M):
            mxy = M[0] + 1j * M[1]
            return mxy if per_walker else jnp.sum(mxy * wn)

        def run_meas(Wm, M0):
            dphi = (jnp.einsum("pd,wpd->pw", Wm[0], r_lo, precision=jax.lax.Precision.HIGHEST)
                    + jnp.einsum("pd,wpd->pw", Wm[1], r_hi, precision=jax.lax.Precision.HIGHEST) + dphi_extra)

            def step(M, x):
                fl, ax, dphi_t, e2, e1, sf = x
                for j in range(n_ev):
                    M = _rf_increment(M, fl[j] * b1, ax[j], xp=jnp)
                c, s = jnp.cos(dphi_t), jnp.sin(dphi_t)
                Mx = (c * M[0] - s * M[1]) * sf * e2
                My = (s * M[0] + c * M[1]) * sf * e2
                M = jnp.stack([Mx, My, M[2] * e1])
                return M, readout(M)
            M_last, out_t = jax.lax.scan(step, M0.astype(r_lo.dtype), (flips, axes, dphi, E2, E1, surf))
            rec = out_t[-1] if echo_idx is None else out_t[jnp.asarray(echo_idx)]
            last = jnp.sum(out_t[-1] * wn) if per_walker else out_t[-1]
            return M_last, rec, last
        return jax.vmap(run_meas)(W_b, M0_b)

    return scan_batch
