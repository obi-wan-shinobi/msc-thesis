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


def run_operator_gd(
    A_train: jnp.ndarray,
    y_train: jnp.ndarray,
    eta: float,
    steps: int,
    y_pred_train_0: Optional[jnp.ndarray] = None,
    A_eval_train: Optional[jnp.ndarray] = None,
    y_pred_eval_0: Optional[jnp.ndarray] = None,
    save_every: int = 100,
):
    """
    Run gradient descent driven by a precomputed operator A_train.

    Train dynamics:
        r_t = y_pred_train_t - y_train
        y_pred_train_{t+1} = y_pred_train_t - eta * A_train @ r_t

    Optional eval dynamics:
        y_pred_eval_{t+1} = y_pred_eval_t - eta * A_eval_train @ r_t

    where the eval update uses the same train residual r_t.

    Args:
        A_train: [n, n] train operator, typically K_train / n.
        y_train: [n] train targets.
        eta: step size.
        steps: number of GD steps.
        y_pred_train_0: optional initial train prediction, shape [n].
        A_eval_train: optional [m, n] eval-train operator, typically K_eval_train / n.
        y_pred_eval_0: optional initial eval prediction, shape [m].
        save_every: save eval snapshots every save_every steps, plus step 0 and final step.

    Returns:
        dict with:
            y_pred_train: [steps+1, n]
            r_train: [steps+1, n]
            loss: [steps+1]
            eta: scalar
            snapshot_steps: [S]
            y_pred_eval_snapshots: [S, m] or None
            y_pred_eval_final: [m] or None
    """
    A_train = jnp.asarray(A_train)
    y_train = jnp.asarray(y_train)

    if A_train.ndim != 2 or A_train.shape[0] != A_train.shape[1]:
        raise ValueError(f"A_train must be square, got shape {A_train.shape}.")
    if y_train.ndim != 1 or y_train.shape[0] != A_train.shape[0]:
        raise ValueError(
            f"y_train must have shape ({A_train.shape[0]},), got {y_train.shape}."
        )
    if steps < 0:
        raise ValueError(f"steps must be nonnegative, got {steps}.")
    if save_every <= 0:
        raise ValueError(f"save_every must be positive, got {save_every}.")

    n = y_train.shape[0]

    if y_pred_train_0 is None:
        y_pred_train = jnp.zeros_like(y_train)
    else:
        y_pred_train = jnp.asarray(y_pred_train_0)
        if y_pred_train.shape != (n,):
            raise ValueError(
                f"y_pred_train_0 must have shape ({n},), got {y_pred_train.shape}."
            )

    use_eval = A_eval_train is not None
    if use_eval:
        A_eval_train = jnp.asarray(A_eval_train)
        if A_eval_train.ndim != 2 or A_eval_train.shape[1] != n:
            raise ValueError(
                f"A_eval_train must have shape [m, {n}], got {A_eval_train.shape}."
            )

        m = A_eval_train.shape[0]
        if y_pred_eval_0 is None:
            y_pred_eval = jnp.zeros((m,), dtype=y_train.dtype)
        else:
            y_pred_eval = jnp.asarray(y_pred_eval_0)
            if y_pred_eval.shape != (m,):
                raise ValueError(
                    f"y_pred_eval_0 must have shape ({m},), got {y_pred_eval.shape}."
                )
    else:
        y_pred_eval = None

    y_preds = [y_pred_train]
    rs = [y_pred_train - y_train]
    losses = [0.5 * jnp.sum((y_pred_train - y_train) ** 2)]

    snapshot_steps = []
    y_eval_snaps = []

    if use_eval:
        snapshot_steps.append(0)
        y_eval_snaps.append(y_pred_eval)

    for t in range(steps):
        r_t = y_pred_train - y_train

        y_pred_train = y_pred_train - eta * (A_train @ r_t)

        if use_eval:
            y_pred_eval = y_pred_eval - eta * (A_eval_train @ r_t)

        r_next = y_pred_train - y_train

        y_preds.append(y_pred_train)
        rs.append(r_next)
        losses.append(0.5 * jnp.sum(r_next**2))

        step_idx = t + 1
        if use_eval and ((step_idx % save_every == 0) or (step_idx == steps)):
            snapshot_steps.append(step_idx)
            y_eval_snaps.append(y_pred_eval)

    return {
        "y_pred_train": jnp.stack(y_preds),
        "r_train": jnp.stack(rs),
        "loss": jnp.asarray(losses),
        "eta": jnp.asarray(eta),
        "snapshot_steps": (
            jnp.asarray(snapshot_steps, dtype=jnp.int32)
            if use_eval
            else jnp.asarray([], dtype=jnp.int32)
        ),
        "y_pred_eval_snapshots": jnp.stack(y_eval_snaps) if use_eval else None,
        "y_pred_eval_final": y_pred_eval if use_eval else None,
    }
