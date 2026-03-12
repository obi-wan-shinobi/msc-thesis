# ------------------------------------------
# core/kernel_circle.py
# ------------------------------------------

from typing import Optional

import jax.numpy as jnp

TWO_PI = 2.0 * jnp.pi


def pairwise_principal_angle(
    gamma1: jnp.ndarray,
    gamma2: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """
    Compute pairwise principal angular distances on circle.

    Given angles gamma1 and gamma2, this returns the matrix of principal angular differences
        delta_{ij} = min(|gamma1_i - gamma2_j|, 2pi - |gamma1_i - gamma2_j|)
    which always lies in [0, pi].

    Args:
        gamma1: Array of shape [N].
        gamma2: Optional array of shape [M]. If None, uses gamma1.

    Returns:
        delta: Array of shape [N, M] with entries in [0,pi].
    """

    gamma1 = jnp.asarray(gamma1)
    gamma2 = gamma1 if gamma2 is None else jnp.asarray(gamma2)

    diff = jnp.abs(gamma1[:, None] - gamma2[None, :])
    delta = jnp.minimum(diff, TWO_PI - diff)

    return delta


def theta_nobias(delta: jnp.ndarray) -> jnp.ndarray:
    """
    Closed-form no-bias NTK kernel on S^1 as a function of the principal angle delta.

    For delta in [0,pi],
        Theta_nobias(delta) = (1/2pi) [ sin(delta) + 2(pi - delta) cos(delta) ].

    Args:
        delta: Array of principal angular differences in [0,pi].

    Returns:
        Array of the same shape as delta.
    """

    delta = jnp.asarray(delta)

    value = jnp.sin(delta) + 2.0 * (jnp.pi - delta) * jnp.cos(delta)

    return value / TWO_PI


def theta_bias(delta: jnp.ndarray) -> jnp.ndarray:
    """
    Closed-form bias NTK kernel on S^1 as a function of the principal angle delta.

    For delta in [0,pi],
        Theta_bias(delta) = (1/2pi) [sin(delta) + 2(pi - delta) cos(delta) + (pi - delta)]

    Equivalently,
        Theta_bias(delta) = Theta_nobias(delta) + (pi - delta) / (2pi)

    Args:
        delta: Array of principal angular differences in [0,pi].
    Returns:
        Array of the same shape as delta
    """
    delta = jnp.asarray(delta)

    return theta_nobias(delta) + (jnp.pi - delta) / TWO_PI
