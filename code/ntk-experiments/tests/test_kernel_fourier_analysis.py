import jax.numpy as jnp
import numpy as np

from core.fourier_basis_circle import build_real_fourier_basis
from core.kernel_analysis import (
    compute_lemma_error_metrics,
    compute_lemma_objects,
    continuum_fourier_eigenvalues_bias,
    expand_frequency_eigenvalues_to_basis,
    finite_n_diagonal_benchmark,
    matrix_error_metrics,
    per_mode_action_relative_errors,
)


def test_expand_frequency_eigenvalues_to_basis():
    mode_freqs = jnp.array([0, 1, 1, 2, 2, 3, 3], dtype=jnp.int32)
    lambda_by_k = jnp.array([10.0, 20.0, 30.0, 40.0])

    out = expand_frequency_eigenvalues_to_basis(mode_freqs, lambda_by_k)

    np.testing.assert_allclose(
        np.asarray(out),
        np.array([10.0, 20.0, 20.0, 30.0, 30.0, 40.0, 40.0]),
        atol=1e-8,
        rtol=1e-8,
    )


def test_finite_n_diagonal_benchmark():
    mode_freqs = jnp.array([0, 1, 1, 2, 2], dtype=jnp.int32)
    lambda_by_k = jnp.array([1.0, 2.0, 3.0])
    g0 = 5.0
    n = 10

    out = finite_n_diagonal_benchmark(
        mode_freqs=mode_freqs,
        lambda_by_k=lambda_by_k,
        g0=g0,
        n=n,
    )

    expected = np.array(
        [
            0.9 * 1.0 + 0.1 * 5.0,
            0.9 * 2.0 + 0.1 * 5.0,
            0.9 * 2.0 + 0.1 * 5.0,
            0.9 * 3.0 + 0.1 * 5.0,
            0.9 * 3.0 + 0.1 * 5.0,
        ]
    )

    np.testing.assert_allclose(np.asarray(out), expected, atol=1e-6, rtol=1e-6)


def test_matrix_error_metrics():
    M = jnp.array([[1.0, -2.0], [3.0, -4.0]])
    out = matrix_error_metrics(M)

    assert np.isclose(float(out["max"]), 4.0)
    assert np.isclose(float(out["fro"]), np.sqrt(30.0))


def test_compute_lemma_objects_shapes():
    n = 8
    K_max = 2
    d = 2 * K_max + 1

    gamma = jnp.linspace(0.0, 2.0 * jnp.pi, n, endpoint=False)
    Phi = build_real_fourier_basis(gamma, K_max)["Phi"]

    A = jnp.eye(n)
    lambda_n_diag = jnp.linspace(1.0, 2.0, d)

    out = compute_lemma_objects(Phi=Phi, A=A, lambda_n_diag=lambda_n_diag)

    assert out["G"].shape == (d, d)
    assert out["H"].shape == (d, d)
    assert out["E"].shape == (n, d)
    assert out["Lambda_n"].shape == (d, d)


def test_compute_lemma_objects_identity_case():
    """
    If A = I and lambda_n_diag = 1 for all modes, then
        E = A Phi - Phi Lambda = 0.
    """
    n = 64
    K_max = 3
    d = 2 * K_max + 1

    gamma = jnp.linspace(0.0, 2.0 * jnp.pi, n, endpoint=False)
    Phi = build_real_fourier_basis(gamma, K_max)["Phi"]

    A = jnp.eye(n)
    lambda_n_diag = jnp.ones(d)

    out = compute_lemma_objects(Phi=Phi, A=A, lambda_n_diag=lambda_n_diag)

    np.testing.assert_allclose(
        np.asarray(out["E"]), np.zeros((n, d)), atol=1e-8, rtol=1e-8
    )


def test_per_mode_action_relative_errors_zero_case():
    n = 16
    d = 5
    E = jnp.zeros((n, d))
    Phi = jnp.ones((n, d))

    rel = per_mode_action_relative_errors(E, Phi)

    np.testing.assert_allclose(np.asarray(rel), np.zeros(d), atol=1e-8, rtol=1e-8)


def test_compute_lemma_error_metrics_zero_case():
    d = 5
    n = 10
    G = jnp.eye(d)
    H = jnp.diag(jnp.arange(1.0, d + 1.0))
    Lambda_n = H
    E = jnp.zeros((n, d))

    out = compute_lemma_error_metrics(G=G, H=H, E=E, Lambda_n=Lambda_n)

    assert np.isclose(float(out["gram_err_max"]), 0.0)
    assert np.isclose(float(out["gram_err_fro"]), 0.0)
    assert np.isclose(float(out["comp_err_max"]), 0.0)
    assert np.isclose(float(out["comp_err_fro"]), 0.0)
    assert np.isclose(float(out["action_err_max"]), 0.0)
    assert np.isclose(float(out["action_err_fro"]), 0.0)


def test_dense_grid_gram_is_close_to_identity():
    """
    Structural sanity test:
    on a dense evenly spaced grid, the continuum-orthonormal Fourier basis
    should satisfy G = (1/n) Phi^T Phi ≈ I.
    """
    n = 4096
    K_max = 8
    d = 2 * K_max + 1

    gamma = jnp.linspace(0.0, 2.0 * jnp.pi, n, endpoint=False)
    Phi = build_real_fourier_basis(gamma, K_max)["Phi"]

    A = jnp.eye(n)
    lambda_n_diag = jnp.ones(d)

    out = compute_lemma_objects(Phi=Phi, A=A, lambda_n_diag=lambda_n_diag)
    G = out["G"]

    np.testing.assert_allclose(np.asarray(G), np.eye(d), atol=5e-4, rtol=5e-4)


def test_bias_continuum_eigenvalues_are_positive_for_first_modes():
    ks = jnp.arange(0, 10)
    vals = continuum_fourier_eigenvalues_bias(ks)

    assert np.all(np.asarray(vals) > 0.0)
