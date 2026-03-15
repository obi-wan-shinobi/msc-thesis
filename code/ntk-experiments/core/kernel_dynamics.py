# ---------------------------------------------
# core/kernel_dynamics.py
# ---------------------------------------------

from typing import Optional

import jax.numpy as jnp
import jax.random as jr

from core.kernel_circle import K0, kernel_matrix_from_gamma, pairwise_principal_angle


def gp_init_from_gamma(
    key,
    gamma: jnp.ndarray,
    jitter: float = 1e-8,
) -> jnp.ndarray:
    """
    Sample u0 ~ N(0, K0_XX), where K0 is the infinite-width GP covariance.
    """
    delta = pairwise_principal_angle(gamma)

    cov = K0(delta)
    cov = cov + jitter * jnp.eye(cov.shape[0])

    mean = jnp.zeros(gamma.shape[0])
    return jr.multivariate_normal(key, mean=mean, cov=cov)


def zero_init_from_gamma(gamma: jnp.ndarray) -> jnp.ndarray:
    return jnp.zeros(gamma.shape[0])


def kernel_gd_step(
    y_pred: jnp.ndarray,
    y: jnp.ndarray,
    theta_xx: jnp.ndarray,
    eta: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    One kernel gradient descent step for square loss (1/2)||y_pred-y||^2.
    """
    r = y_pred - y
    y_pred_next = y_pred - eta * (theta_xx @ r)
    r_next = y_pred_next - y
    return y_pred_next, r_next


def run_kernel_gd(
    gamma: jnp.ndarray,
    y: jnp.ndarray,
    eta: float,
    steps: int,
    kernel: str = "bias",
    y_pred_0: Optional[jnp.ndarray] = None,
):
    """
    Run kernel GD on train predictions.
    """
    theta_xx = kernel_matrix_from_gamma(gamma, kernel=kernel)

    if y_pred_0 is None:
        y_pred = jnp.zeros_like(y)
    else:
        y_pred = y_pred_0

    y_preds = [y_pred]
    rs = [y_pred - y]
    losses = [0.5 * jnp.sum((y_pred - y) ** 2)]

    for _ in range(steps):
        y_pred, r = kernel_gd_step(y_pred, y, theta_xx, eta)
        y_preds.append(y_pred)
        rs.append(r)
        losses.append(0.5 * jnp.sum(r**2))

    return {
        "theta_xx": theta_xx,
        "y_pred": jnp.stack(y_preds),
        "r": jnp.stack(rs),
        "loss": jnp.array(losses),
    }
