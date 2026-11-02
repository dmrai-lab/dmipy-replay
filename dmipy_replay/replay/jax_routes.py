"""The differentiable JAX twins of the replay (:func:`replay_signal_jax`, :func:`replay_batch_jax`): the forward a
gradient-based waveform or B1 optimiser differentiates through, and a fit evaluates over a grid of poses in one device
call. JAX is the ``jax`` extra, imported when a twin runs."""

__all__ = ["replay_signal_jax", "replay_batch_jax"]


def replay_signal_jax(position_coeffs, spin_weights, W, *, surface_logw=None):
    """JAX/autodiff twin of :func:`replay_signal` -- differentiable in the compiled scheme ``W``
    (hence in the waveform ``G`` that produced it) and jittable. ``position_coeffs`` is
    ``(n_walkers, K+2, n_axes)`` -- two endpoints then sine bands -- and ``W`` must come from
    :func:`compile_scheme`, which emits the matching row order.

    ``surface_logw`` is the per-walker surface log-weight from :func:`surface_logweight` (it does
    not depend on ``W``, so it is computed once on the host and passed in). Returns the complex
    signal (take ``abs`` for magnitude). This is the forward a gradient-based waveform/B1
    optimizer differentiates through."""
    import jax.numpy as jnp
    C = jnp.asarray(position_coeffs)
    N_w, K, _ = C.shape
    phi = C.reshape(N_w, K * C.shape[2]) @ jnp.asarray(W)
    w0 = jnp.asarray(spin_weights)
    w_eff = w0 if surface_logw is None else w0 * jnp.exp(jnp.asarray(surface_logw))
    return (w_eff[:, None] * jnp.exp(1j * phi)).sum(0) / w0.sum()


def replay_batch_jax(position_coeffs, spin_weights, W_batch, *, surface_logw=None):
    """:func:`replay_signal_jax` over a batch of compiled schemes ``W_batch`` ``(n_batch, n_c (K+2), n_meas)``
    -- one candidate orientation or knob setting per entry -- as one traced call: ``(n_batch, n_meas)``
    magnitudes. What a fit evaluates over a grid of poses in one device call."""
    import jax
    import jax.numpy as jnp
    fn = lambda W: jnp.abs(replay_signal_jax(position_coeffs, spin_weights, W, surface_logw=surface_logw))
    return jax.vmap(fn)(jnp.asarray(W_batch))
