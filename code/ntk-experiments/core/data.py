# ------------------------------------------------------------
# core/data.py
# ------------------------------------------------------------
"""
Data generation and target construction utilities
for NTK experiments on the unit circle and Gaussian inputs.
"""

from typing import NamedTuple, Tuple

import jax.numpy as jnp
from jax import random

# ============================================================
# Fourier-based target specification
# ============================================================


class FourierTarget(NamedTuple):
    """Parametrization of a Fourier-mixture target on the unit circle."""

    Ks: jnp.ndarray  # integer frequencies
    amps: jnp.ndarray  # amplitudes
    phases: jnp.ndarray  # phase shifts (radians)


# --- Build a Fourier target ---
def make_fourier_target(Ks, amps, phases=None) -> FourierTarget:
    Ks, amps = jnp.asarray(Ks), jnp.asarray(amps)
    phases = jnp.zeros_like(Ks) if phases is None else jnp.asarray(phases)
    assert (
        Ks.shape == amps.shape == phases.shape
    ), "Ks, amps, phases must have same shape"
    return FourierTarget(Ks=Ks, amps=amps, phases=phases)


# --- Evaluate f*(γ) = Σ a_k sin(kγ + φ_k) ---
def f_star_gamma(gamma: jnp.ndarray, spec: FourierTarget) -> jnp.ndarray:
    A = spec.Ks[:, None] * gamma[None, :] + spec.phases[:, None]
    return jnp.einsum("m,mn->n", spec.amps, jnp.sin(A))


# --- Evaluate f*(x) for 2D circle points ---
def f_star_X(X: jnp.ndarray, spec: FourierTarget) -> jnp.ndarray:
    gamma = jnp.arctan2(X[:, 1], X[:, 0])
    return f_star_gamma(gamma, spec)


# --- Partial Fourier sums ---
def f_star_gamma_partial(gamma, spec, k_count):
    """Partial sum using first k_count Fourier modes."""
    return f_star_gamma(
        gamma,
        FourierTarget(spec.Ks[:k_count], spec.amps[:k_count], spec.phases[:k_count]),
    )


# --- Sampling points on the circle for a given Fourier target ---
def sample_train_on_circle(
    gamma_eval: jnp.ndarray,
    X_eval: jnp.ndarray,
    spec: FourierTarget,
    M_train: int,
    key: random.PRNGKey,
    add_noise_std: float = 0.0,
):
    """
    Randomly sample M_train points from the circle and return (X_train, gamma_train, y_train).
    """
    idx = random.choice(key, gamma_eval.shape[0], shape=(M_train,), replace=False)
    X_train = X_eval[idx]
    gamma_train = gamma_eval[idx]
    y_train = f_star_gamma(gamma_train, spec)
    if add_noise_std > 0:
        y_train = y_train + add_noise_std * random.normal(
            random.PRNGKey(1), y_train.shape
        )
    return X_train, gamma_train, y_train


# ============================================================
# Additional helpers for non-Fourier experiments
# ============================================================


def make_probe_circle(n_points: int) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Unit circle probe manifold used for kernel profiling.
    Returns:
        gamma: [N] angles
        X_circle: [N,2] coordinates
        x0: [1,2] anchor at (1,0)
    """
    gamma = jnp.linspace(-jnp.pi, jnp.pi, n_points, endpoint=False)
    X_circle = jnp.stack([jnp.cos(gamma), jnp.sin(gamma)], axis=1)
    x0 = jnp.array([[1.0, 0.0]])
    return gamma, X_circle, x0


def make_gaussian_regression_task(M_train: int, key: random.PRNGKey):
    """
    Regression task on Gaussian inputs with f*(x)=x1*x2.
    Returns:
        Xtr, ytr
    """
    Xtr = random.normal(key, (M_train, 2))
    ytr = Xtr[:, 0] * Xtr[:, 1]
    return Xtr, ytr


def make_circle_regression_dataset(N_eval: int, M_train: int, seed: int):
    """
    Dataset on the unit circle for f*(x)=x1*x2 regression.
    Returns:
        gamma_eval, X_eval, gamma_train, X_train, y_train
    """
    gamma_eval = jnp.linspace(-jnp.pi, jnp.pi, N_eval, endpoint=False)
    X_eval = jnp.stack([jnp.cos(gamma_eval), jnp.sin(gamma_eval)], axis=1)
    key = random.PRNGKey(seed)
    idx = random.choice(key, N_eval, shape=(M_train,), replace=False)
    X_train, gamma_train = X_eval[idx], gamma_eval[idx]
    y_train = X_train[:, 0] * X_train[:, 1]
    return gamma_eval, X_eval, gamma_train, X_train, y_train
