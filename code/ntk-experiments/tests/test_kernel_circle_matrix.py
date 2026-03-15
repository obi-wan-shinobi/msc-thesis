# ------------------------------------------------------------
# tests/test_kernel_circle_matrix.py
# ------------------------------------------------------------

import jax.numpy as jnp

from core.kernel_circle import kernel_cross_matrix_from_gamma, kernel_matrix_from_gamma


def test_kernel_matrix_shape():
    """Square kernel matrix should be N x N."""
    gamma = jnp.linspace(0, 2 * jnp.pi, 32, endpoint=False)

    K = kernel_matrix_from_gamma(gamma, kernel="bias")

    assert K.shape == (32, 32)


def test_cross_kernel_shape():
    """Cross kernel matrix should be N x M."""
    gamma1 = jnp.linspace(0, 2 * jnp.pi, 20, endpoint=False)
    gamma2 = jnp.linspace(0, 2 * jnp.pi, 15, endpoint=False)

    K = kernel_cross_matrix_from_gamma(gamma1, gamma2, kernel="bias")

    assert K.shape == (20, 15)


def test_kernel_matrix_symmetry():
    """Gram matrix should be symmetric."""
    gamma = jnp.linspace(0, 2 * jnp.pi, 64, endpoint=False)

    K = kernel_matrix_from_gamma(gamma, kernel="bias")

    assert jnp.allclose(K, K.T, atol=1e-6)


def test_cross_vs_square_consistency():
    """
    kernel_matrix_from_gamma(gamma) should equal
    kernel_cross_matrix_from_gamma(gamma, gamma).
    """
    gamma = jnp.linspace(0, 2 * jnp.pi, 50, endpoint=False)

    K1 = kernel_matrix_from_gamma(gamma, kernel="bias")
    K2 = kernel_cross_matrix_from_gamma(gamma, gamma, kernel="bias")

    assert jnp.allclose(K1, K2, atol=1e-6)


def test_diagonal_values_bias():
    """
    For the bias kernel, Θ(0) = 3/2.
    """
    gamma = jnp.linspace(0, 2 * jnp.pi, 40, endpoint=False)

    K = kernel_matrix_from_gamma(gamma, kernel="bias")

    diag = jnp.diag(K)

    assert jnp.allclose(diag, 1.5, atol=1e-6)


def test_diagonal_values_nobias():
    """
    For the no-bias kernel, Θ(0) = 1.
    """
    gamma = jnp.linspace(0, 2 * jnp.pi, 40, endpoint=False)

    K = kernel_matrix_from_gamma(gamma, kernel="nobias")

    diag = jnp.diag(K)

    assert jnp.allclose(diag, 1.0, atol=1e-6)


def test_kernel_matrix_psd():
    gamma = jnp.linspace(0, 2 * jnp.pi, 50, endpoint=False)

    K = kernel_matrix_from_gamma(gamma, kernel="bias")

    eigvals = jnp.linalg.eigvalsh(K)

    assert jnp.min(eigvals) > -1e-6
