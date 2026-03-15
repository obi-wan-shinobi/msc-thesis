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


def K0(delta: jnp.ndarray) -> jnp.ndarray:
    """
    Arc-cosine / NNGP covariance kernel block on S^1.

    For delta in [0, pi],
        K0(delta) = (1 / 2pi) * [sin(delta) + (pi - delta) cos(delta)].

    Args:
        delta: Principal angular differences in [0, pi].

    Returns:
        Array with same shape as delta.
    """
    delta = jnp.asarray(delta)
    value = jnp.sin(delta) + (jnp.pi - delta) * jnp.cos(delta)
    return value / TWO_PI


def K1(delta: jnp.ndarray) -> jnp.ndarray:
    """
    Derivative-gate kernel block on S^1.

    For delta in [0, pi],
        K1(delta) = (pi - delta) / (2pi).

    Args:
        delta: Principal angular differences in [0, pi].

    Returns:
        Array with same shape as delta.
    """
    delta = jnp.asarray(delta)
    return (jnp.pi - delta) / TWO_PI


def theta_nobias(delta: jnp.ndarray) -> jnp.ndarray:
    """
    Infinite-width NTK on S^1 without hidden bias contribution.

    Decomposition:
        Theta_nobias(delta) = K0(delta) + cos(delta) * K1(delta)

    Equivalent closed form:
        Theta_nobias(delta)
        = (1 / 2pi) * [sin(delta) + 2 (pi - delta) cos(delta)].
    """
    delta = jnp.asarray(delta)
    return K0(delta) + jnp.cos(delta) * K1(delta)


def theta_bias(delta: jnp.ndarray) -> jnp.ndarray:
    """
    Infinite-width NTK on S^1 with hidden bias contribution.

    Decomposition:
        Theta_bias(delta) = K0(delta) + (1 + cos(delta)) * K1(delta)

    Equivalent closed form:
        Theta_bias(delta)
        = (1 / 2pi) * [sin(delta) + 2 (pi - delta) cos(delta) + (pi - delta)].
    """
    delta = jnp.asarray(delta)
    return K0(delta) + (1.0 + jnp.cos(delta)) * K1(delta)


def kernel_cross_matrix_from_gamma(
    gamma1: jnp.ndarray,
    gamma2: jnp.ndarray,
    kernel: str = "bias",
) -> jnp.ndarray:
    """
    Build a cross-kernel matrix on S^1 from two angle arrays.

    Given gamma1 of shape [N] and gamma2 of shape [M], this returns
    the matrix K of shape [N, M] with entries
        K_{ij} = Theta(gamma1_i, gamma2_j),
    where Theta is either the bias-included or no-bias closed-form kernel.

    Args:
        gamma1: Array of shape [N] of angles in radians.
        gamma2: Array of shape [M] of angles in radians.
        kernel: Either "bias" or "nobias".

    Returns:
        Kernel matrix of shape [N, M].
    """
    delta = pairwise_principal_angle(gamma1, gamma2)

    if kernel == "bias":
        return theta_bias(delta)
    if kernel == "nobias":
        return theta_nobias(delta)

    raise ValueError(f"Unknown kernel='{kernel}'. Expected 'bias' or 'nobias'.")


def kernel_matrix_from_gamma(
    gamma: jnp.ndarray,
    kernel: str = "bias",
) -> jnp.ndarray:
    """
    Build a square kernel matrix on S^1 from one angle array.

    Args:
        gamma: Array of shape [N] of angles in radians.
        kernel: Either "bias" or "nobias".

    Returns:
        Kernel matrix of shape [N, N].
    """
    return kernel_cross_matrix_from_gamma(gamma, gamma, kernel=kernel)
