# -----------------------------------------------
# core/preconditioned_operator.py
# -----------------------------------------------


from __future__ import annotations

from typing import Optional

import jax.numpy as jnp


def build_theory_preconditioner(
    Phi: jnp.ndarray,
    lambda_n_diag: jnp.ndarray,
    tau: float = 0.0,
    reg: float = 1e-8,
) -> dict:
    """
    Build the theory-driven low-mode preconditioner

        P = Phi (Lambda_n + tau I)^{-1} G^{-1} Phi^T / n

    where
        G = (1/n) Phi^T Phi.

    Args:
        Phi: [n, d] sampled Fourier basis matrix.
        lambda_n_diag: [d] finite-n diagonal benchmark entries.
        tau: regularization for the inverse eigenvalues.
        reg: small ridge for stable inversion of G.

    Returns:
        dict with:
            G: [d, d]
            G_inv: [d, d]
            Lambda_n: [d, d]
            M: [d, d] = (Lambda_n + tau I)^{-1}
            P: [n, n]
    """
    Phi = jnp.asarray(Phi)
    lambda_n_diag = jnp.asarray(lambda_n_diag)

    if Phi.ndim != 2:
        raise ValueError(f"Phi must be 2D, got shape {Phi.shape}.")
    if lambda_n_diag.ndim != 1 or lambda_n_diag.shape[0] != Phi.shape[1]:
        raise ValueError(
            f"lambda_n_diag must have shape ({Phi.shape[1]},), got {lambda_n_diag.shape}."
        )
    if tau < 0:
        raise ValueError(f"tau must be nonnegative, got {tau}.")
    if reg < 0:
        raise ValueError(f"reg must be nonnegative, got {reg}.")

    n, d = Phi.shape
    I_d = jnp.eye(d, dtype=Phi.dtype)

    G = (Phi.T @ Phi) / n
    G_inv = jnp.linalg.solve(G + reg * I_d, I_d)

    Lambda_n = jnp.diag(lambda_n_diag)
    M = jnp.diag(1.0 / (lambda_n_diag + tau))

    P = (Phi @ M @ G_inv @ Phi.T) / n

    return {
        "G": G,
        "G_inv": G_inv,
        "Lambda_n": Lambda_n,
        "M": M,
        "P": P,
    }


def build_theory_preconditioned_operators(
    A_train: jnp.ndarray,
    Phi_train: jnp.ndarray,
    lambda_n_diag: jnp.ndarray,
    *,
    Phi_eval: Optional[jnp.ndarray] = None,
    tau: float = 0.0,
    reg: float = 1e-8,
) -> dict:
    """
    Build the theory-driven preconditioned operators.

    Train operator:
        B_train = P_train A_train

    Eval operator:
        B_eval_train = P_eval_train A_train

    where
        P_train      = Phi_train M G^{-1} Phi_train^T / n
        P_eval_train = Phi_eval  M G^{-1} Phi_train^T / n

    Args:
        A_train: [n, n] normalized empirical operator, typically K_train / n.
        Phi_train: [n, d] sampled Fourier basis on train points.
        lambda_n_diag: [d] finite-n diagonal benchmark entries.
        Phi_eval: optional [m, d] sampled Fourier basis on eval points.
        tau: regularization for inverse eigenvalues.
        reg: ridge for stable inversion of G.

    Returns:
        dict with:
            B_train: [n, n]
            B_eval_train: [m, n] or None
            plus outputs of build_theory_preconditioner(...)
    """
    A_train = jnp.asarray(A_train)
    Phi_train = jnp.asarray(Phi_train)

    if A_train.ndim != 2 or A_train.shape[0] != A_train.shape[1]:
        raise ValueError(f"A_train must be square, got shape {A_train.shape}.")
    if Phi_train.ndim != 2 or Phi_train.shape[0] != A_train.shape[0]:
        raise ValueError(
            f"Phi_train must have shape ({A_train.shape[0]}, d), got {Phi_train.shape}."
        )

    prec = build_theory_preconditioner(
        Phi=Phi_train,
        lambda_n_diag=lambda_n_diag,
        tau=tau,
        reg=reg,
    )

    M = prec["M"]
    G_inv = prec["G_inv"]
    P_train = prec["P"]

    n = Phi_train.shape[0]
    B_train = P_train @ A_train

    B_eval_train = None
    if Phi_eval is not None:
        Phi_eval = jnp.asarray(Phi_eval)
        if Phi_eval.ndim != 2 or Phi_eval.shape[1] != Phi_train.shape[1]:
            raise ValueError(
                f"Phi_eval must have shape (m, {Phi_train.shape[1]}), got {Phi_eval.shape}."
            )

        P_eval_train = (Phi_eval @ M @ G_inv @ Phi_train.T) / n
        B_eval_train = P_eval_train @ A_train

    return {
        **prec,
        "B_train": B_train,
        "B_eval_train": B_eval_train,
    }
