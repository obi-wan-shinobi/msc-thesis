# ------------------------------------------------------------
# tests/test_kernel_analysis.py
# ------------------------------------------------------------

import jax.numpy as jnp

from core.kernel_analysis import (
    kernel_eigendecomposition,
    project_residuals_onto_eigenvectors,
    project_residuals_onto_fourier_modes,
)
from core.kernel_circle import kernel_matrix_from_gamma


def test_kernel_eigendecomposition_shapes():
    gamma = jnp.array([0.0, 1.0, 2.0, 3.0])
    theta_xx = kernel_matrix_from_gamma(gamma, kernel="bias")

    evals, evecs = kernel_eigendecomposition(theta_xx)

    assert evals.shape == (4,)
    assert evecs.shape == (4, 4)


def test_kernel_eigendecomposition_reconstructs_matrix():
    gamma = jnp.array([0.0, 1.0, 2.0, 3.0])
    theta_xx = kernel_matrix_from_gamma(gamma, kernel="bias")

    evals, evecs = kernel_eigendecomposition(theta_xx)
    theta_reconstructed = evecs @ jnp.diag(evals) @ evecs.T

    assert jnp.allclose(theta_xx, theta_reconstructed, atol=1e-6)


def test_project_residuals_onto_eigenvectors_shape():
    residuals = jnp.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ]
    )
    evecs = jnp.eye(2)

    coeffs = project_residuals_onto_eigenvectors(residuals, evecs)

    assert coeffs.shape == (2, 2)
    assert jnp.allclose(coeffs, residuals)


def test_project_residuals_onto_fourier_modes_shapes():
    gamma = jnp.linspace(0.0, 2.0 * jnp.pi, 8, endpoint=False)
    residuals = jnp.ones((5, 8))

    out = project_residuals_onto_fourier_modes(residuals, gamma, ks=[0, 1, 2])

    assert out["cos_0"].shape == (5,)
    assert out["cos_1"].shape == (5,)
    assert out["sin_1"].shape == (5,)
    assert out["cos_2"].shape == (5,)
    assert out["sin_2"].shape == (5,)
