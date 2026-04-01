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


def expand_frequency_eigenvalues_to_basis(
    mode_freqs: jnp.ndarray,
    lambda_by_k: jnp.ndarray,
) -> jnp.ndarray:
    """
    Expand one eigenvalue per Fourier frequency k to one eigenvalue per basis column.

    Example:
        mode_freqs     = [0, 1, 1, 2, 2, 3, 3]
        lambda_by_k    = [l0, l1, l2, l3]
        returns        = [l0, l1, l1, l2, l2, l3, l3]

    Args:
        mode_freqs: Integer array of shape [d], where each entry is the frequency
            attached to one basis column.
        lambda_by_k: Array of shape [K_max + 1], where lambda_by_k[k] is the
            continuum eigenvalue for frequency k.

    Returns:
        lambda_basis: Array of shape [d].
    """
    mode_freqs = jnp.asarray(mode_freqs, dtype=jnp.int32)
    lambda_by_k = jnp.asarray(lambda_by_k)

    if mode_freqs.ndim != 1:
        raise ValueError(f"mode_freqs must be 1D, got shape {mode_freqs.shape}.")
    if lambda_by_k.ndim != 1:
        raise ValueError(f"lambda_by_k must be 1D, got shape {lambda_by_k.shape}.")

    max_freq_needed = int(jnp.max(mode_freqs))
    if lambda_by_k.shape[0] <= max_freq_needed:
        raise ValueError(
            "lambda_by_k is too short for the requested mode frequencies: "
            f"need at least {max_freq_needed + 1} entries, got {lambda_by_k.shape[0]}."
        )

    return lambda_by_k[mode_freqs]


def finite_n_diagonal_benchmark(
    mode_freqs: jnp.ndarray,
    lambda_by_k: jnp.ndarray,
    g0: float,
    n: int,
) -> jnp.ndarray:
    """
    Build the finite-n diagonal benchmark Lambda^(n) on basis columns.

    For each basis column p with associated frequency k_p,
        lambda_p^(n) = (1 - 1/n) * lambda_{k_p} + (1/n) * g(0)

    Args:
        mode_freqs: Integer array of shape [d], one frequency per basis column.
        lambda_by_k: Continuum eigenvalues by frequency, shape [K_max + 1].
        g0: Kernel diagonal value Theta(x, x) = Theta(0).
        n: Number of training samples.

    Returns:
        lambda_n_diag: Array of shape [d].
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}.")

    lambda_basis = expand_frequency_eigenvalues_to_basis(mode_freqs, lambda_by_k)
    return (1.0 - 1.0 / n) * lambda_basis + (g0 / n)


def compute_lemma_objects(
    Phi: jnp.ndarray,
    A: jnp.ndarray,
    lambda_n_diag: jnp.ndarray,
) -> dict:
    """
    Compute the core lemma-validation matrices:
        G = (1/n) Phi^T Phi
        H = (1/n) Phi^T A Phi
        E = A Phi - Phi Lambda^(n)

    where A is already the normalized empirical operator A = K / n.

    Args:
        Phi: Basis matrix of shape [n, d].
        A: Empirical operator matrix of shape [n, n].
        lambda_n_diag: Diagonal entries of Lambda^(n), shape [d].

    Returns:
        dict with:
            G: [d, d]
            H: [d, d]
            E: [n, d]
            Lambda_n: [d, d]
    """
    Phi = jnp.asarray(Phi)
    A = jnp.asarray(A)
    lambda_n_diag = jnp.asarray(lambda_n_diag)

    if Phi.ndim != 2:
        raise ValueError(f"Phi must be 2D, got shape {Phi.shape}.")
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"A must be square, got shape {A.shape}.")
    if Phi.shape[0] != A.shape[0]:
        raise ValueError(
            f"Phi and A have incompatible shapes: Phi {Phi.shape}, A {A.shape}."
        )
    if lambda_n_diag.ndim != 1 or lambda_n_diag.shape[0] != Phi.shape[1]:
        raise ValueError(
            "lambda_n_diag must be a 1D array with one entry per basis column: "
            f"got shape {lambda_n_diag.shape}, expected ({Phi.shape[1]},)."
        )

    n = Phi.shape[0]

    G = (Phi.T @ Phi) / n
    H = (Phi.T @ A @ Phi) / n
    E = A @ Phi - Phi * lambda_n_diag[None, :]
    Lambda_n = jnp.diag(lambda_n_diag)

    return {
        "G": G,
        "H": H,
        "E": E,
        "Lambda_n": Lambda_n,
    }


def matrix_error_metrics(M: jnp.ndarray) -> dict:
    """
    Compute max-entry and Frobenius norms of a matrix.

    Args:
        M: Array of shape [a, b].

    Returns:
        dict with:
            max: ||M||_max = max_ij |M_ij|
            fro: ||M||_F
    """
    M = jnp.asarray(M)
    return {
        "max": jnp.max(jnp.abs(M)),
        "fro": jnp.linalg.norm(M, ord="fro"),
    }


def per_mode_action_relative_errors(
    E: jnp.ndarray,
    Phi: jnp.ndarray,
    eps: float = 1e-12,
) -> jnp.ndarray:
    """
    Compute per-mode relative action errors:
        relerr_p = ||E[:, p]||_2 / ||Phi[:, p]||_2

    where
        E[:, p] = A phi_p - lambda_p^(n) phi_p.

    Args:
        E: Action error matrix of shape [n, d].
        Phi: Basis matrix of shape [n, d].
        eps: Small constant for numerical stability.

    Returns:
        Array of shape [d].
    """
    E = jnp.asarray(E)
    Phi = jnp.asarray(Phi)

    if E.shape != Phi.shape:
        raise ValueError(
            f"E and Phi must have the same shape, got {E.shape} and {Phi.shape}."
        )

    num = jnp.linalg.norm(E, axis=0)
    den = jnp.linalg.norm(Phi, axis=0)
    return num / (den + eps)


def compute_lemma_error_metrics(
    G: jnp.ndarray,
    H: jnp.ndarray,
    E: jnp.ndarray,
    Lambda_n: jnp.ndarray,
) -> dict:
    """
    Compute the scalar metrics used in the three finite-sample lemma checks.

    Args:
        G: Gram matrix, shape [d, d]
        H: Compressed operator, shape [d, d]
        E: Action error matrix, shape [n, d]
        Lambda_n: Finite-n diagonal benchmark matrix, shape [d, d]

    Returns:
        dict with:
            gram_err_max
            gram_err_fro
            comp_err_max
            comp_err_fro
            action_err_max
            action_err_fro
    """
    G = jnp.asarray(G)
    H = jnp.asarray(H)
    E = jnp.asarray(E)
    Lambda_n = jnp.asarray(Lambda_n)

    I = jnp.eye(G.shape[0], dtype=G.dtype)

    gram_metrics = matrix_error_metrics(G - I)
    comp_metrics = matrix_error_metrics(H - Lambda_n)
    action_metrics = matrix_error_metrics(E)

    return {
        "gram_err_max": gram_metrics["max"],
        "gram_err_fro": gram_metrics["fro"],
        "comp_err_max": comp_metrics["max"],
        "comp_err_fro": comp_metrics["fro"],
        "action_err_max": action_metrics["max"],
        "action_err_fro": action_metrics["fro"],
    }


def compressed_operator_from_lemma_objects(
    G: jnp.ndarray,
    H: jnp.ndarray,
    reg: float = 1e-8,
) -> jnp.ndarray:
    """
    Effective compressed operator in sampled Fourier coordinates:

        C = G^{-1} H
    """
    G = jnp.asarray(G)
    H = jnp.asarray(H)

    if G.shape != H.shape:
        raise ValueError(
            f"G and H must have the same shape, got {G.shape} and {H.shape}."
        )

    I = jnp.eye(G.shape[0], dtype=G.dtype)
    return jnp.linalg.solve(G + reg * I, H)
