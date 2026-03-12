# ------------------------------------------------------------
# tests/test_kernel_circle_nt.py
# ------------------------------------------------------------

import jax.numpy as jnp

from core.kernel_circle import pairwise_principal_angle, theta_bias, theta_nobias
from core.model import build_mlp, build_mlp_custom


def test_theta_nobias_matches_neural_tangents_depth1_b0():
    """
    Compare the closed-form nobias kernel on S^1 against the analytic
    infinite-width NTK returned by neural_tangents for a 1-hidden-layer
    ReLU MLP with zero bias variance.

    We use:
        - depth_hidden = 1
        - b_std = 0.0
        - parameterization = "ntk"
    """
    n_points = 256
    gamma = jnp.linspace(0, 2 * jnp.pi, n_points, endpoint=False)
    X = jnp.stack([jnp.cos(gamma), jnp.sin(gamma)], axis=1)

    _, _, kernel_fn = build_mlp(
        width=256,
        b_std=0.0,
        depth_hidden=1,
        parameterization="ntk",
    )

    K_nt = kernel_fn(X, X, "ntk")
    K_nt = jnp.asarray(K_nt)

    assert K_nt.shape == (n_points, n_points)

    delta = pairwise_principal_angle(gamma)
    K_closed = theta_nobias(delta)

    assert K_closed.shape == (n_points, n_points)
    assert jnp.allclose(K_nt, K_closed / 2, atol=1e-4, rtol=1e-4)


def test_theta_bias_matches_neural_tangents_depth1_b0():
    """
    Compare the closed-form full kernel on S^1 against the analytic
    infinite-width NTK returned by neural_tangents for a 1-hidden-layer
    ReLU MLP with non-zero bias variance.

    We use:
        - depth_hidden = 1
        - b_std = 1.0
        - parameterization = "ntk"
    """
    n_points = 256
    gamma = jnp.linspace(0, 2 * jnp.pi, n_points, endpoint=False)
    X = jnp.stack([jnp.cos(gamma), jnp.sin(gamma)], axis=1)

    _, _, kernel_fn = build_mlp_custom(
        width=256,
        b_std=1.0,
        depth_hidden=1,
        parameterization="ntk",
    )

    K_nt = kernel_fn(X, X, "ntk")
    K_nt = jnp.asarray(K_nt)

    assert K_nt.shape == (n_points, n_points)

    delta = pairwise_principal_angle(gamma)
    K_closed = theta_bias(delta)

    print(f"{K_nt=}")
    print(f"{K_closed=}")

    assert K_closed.shape == (n_points, n_points)
    assert jnp.allclose(K_nt, K_closed, atol=1e-4, rtol=1e-4)
