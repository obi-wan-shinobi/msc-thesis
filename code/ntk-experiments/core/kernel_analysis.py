# ---------------------------------------------
# core/kernel_analysis.py
# ---------------------------------------------

import jax.numpy as jnp


def continuum_fourier_eigenvalues_bias(ks: jnp.ndarray) -> jnp.ndarray:
    """
    Continuum Fourier eigenvalues for the full bias NTK operator on S^1.

    The continuum operator is
        (Tf)(phi) = ∫ Theta(phi - psi) f(psi) dpsi / (2pi)

    so these are eigenvalues of T, not of the raw discrete Gram matrix K.
    On an evenly spaced grid with n points, the raw matrix eigenvalues satisfy
        lambda_k(K) ≈ n * lambda_k(T).
    """
    ks = jnp.asarray(ks)
    kf = ks.astype(jnp.float32)

    out = jnp.zeros_like(kf)

    out = jnp.where(ks == 0, 0.25 + 3.0 / (jnp.pi**2), out)
    out = jnp.where(ks == 1, 0.25 + 1.0 / (jnp.pi**2), out)

    even_mask = (ks >= 2) & (ks % 2 == 0)
    odd_mask = (ks >= 3) & (ks % 2 == 1)

    out = jnp.where(
        even_mask,
        (kf**2 + 3.0) / (jnp.pi**2 * (kf**2 - 1.0) ** 2),
        out,
    )
    out = jnp.where(
        odd_mask,
        1.0 / (jnp.pi**2 * kf**2),
        out,
    )

    return out


def kernel_eigendecomposition(
    theta_xx: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Eigendecomposition of a symmetric kernel matrix, sorted in descending order.
    """
    theta_xx = jnp.asarray(theta_xx)

    evals, evecs = jnp.linalg.eigh(theta_xx)
    idx = jnp.argsort(evals)[::-1]

    evals = evals[idx]
    evecs = evecs[:, idx]

    return evals, evecs


def project_residuals_onto_eigenvectors(
    residuals: jnp.ndarray,
    evecs: jnp.ndarray,
) -> jnp.ndarray:
    """
    Project residual trajectories onto kernel eigenvectors.

    Args:
        residuals: Array of shape [T, n].
        evecs: Eigenvector matrix of shape [n, n], columns are eigenvectors.

    Returns:
        coeffs: Array of shape [T, n], where coeffs[t, k] = <r_t, q_k>.
    """
    residuals = jnp.asarray(residuals)
    evecs = jnp.asarray(evecs)

    return residuals @ evecs


def normalize_vector(v: jnp.ndarray, eps: float = 1e-12) -> jnp.ndarray:
    return v / (jnp.linalg.norm(v) + eps)


def cosine_mode(gamma: jnp.ndarray, k: int) -> jnp.ndarray:
    return jnp.cos(k * gamma)


def sine_mode(gamma: jnp.ndarray, k: int) -> jnp.ndarray:
    return jnp.sin(k * gamma)


def project_residuals_onto_fourier_modes(
    residuals: jnp.ndarray,
    gamma: jnp.ndarray,
    ks: list[int],
) -> dict[str, jnp.ndarray]:
    """
    Project residuals onto sampled Fourier modes on the training grid.

    Args:
        residuals: Array of shape [T, n].
        gamma: Array of shape [n].
        ks: Frequencies to project onto.

    Returns:
        Dictionary mapping mode name -> trajectory of shape [T].
    """
    out = {}

    for k in ks:
        ck = normalize_vector(cosine_mode(gamma, k))
        out[f"cos_{k}"] = residuals @ ck

        if k > 0:
            sk = normalize_vector(sine_mode(gamma, k))
            out[f"sin_{k}"] = residuals @ sk

    return out
