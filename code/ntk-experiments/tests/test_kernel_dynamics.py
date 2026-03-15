# ------------------------------------------------------------
# tests/test_kernel_dynamics.py
# ------------------------------------------------------------

import jax.numpy as jnp
import numpy as np

from core.kernel_circle import kernel_matrix_from_gamma
from core.kernel_dynamics import kernel_gd_step, run_kernel_gd, zero_init_from_gamma


def test_zero_init_from_gamma():
    gamma = jnp.array([0.1, 1.2, 3.4])
    y_pred_0 = zero_init_from_gamma(gamma)

    assert y_pred_0.shape == (3,)
    assert jnp.allclose(y_pred_0, jnp.zeros(3))


def test_kernel_gd_step_matches_manual_update():
    y_pred = jnp.array([1.0, -1.0])
    y = jnp.array([0.5, 0.5])
    theta_xx = jnp.array(
        [
            [2.0, 1.0],
            [1.0, 3.0],
        ]
    )
    eta = 0.2

    r = y_pred - y
    expected_y_pred_next = y_pred - eta * (theta_xx @ r)
    expected_r_next = expected_y_pred_next - y

    y_pred_next, r_next = kernel_gd_step(y_pred, y, theta_xx, eta)

    assert jnp.allclose(y_pred_next, expected_y_pred_next)
    assert jnp.allclose(r_next, expected_r_next)


def test_kernel_gd_step_zero_residual_is_fixed_point():
    y = jnp.array([1.0, -2.0, 0.5])
    y_pred = y.copy()
    theta_xx = jnp.array(
        [
            [2.0, 0.5, 0.0],
            [0.5, 1.5, 0.1],
            [0.0, 0.1, 1.0],
        ]
    )
    eta = 0.3

    y_pred_next, r_next = kernel_gd_step(y_pred, y, theta_xx, eta)

    assert jnp.allclose(y_pred_next, y_pred)
    assert jnp.allclose(r_next, jnp.zeros_like(y))


def test_run_kernel_gd_output_shapes():
    gamma = jnp.array([0.0, 1.0, 2.0, 3.0])
    y = jnp.array([1.0, -1.0, 0.5, 2.0])

    out = run_kernel_gd(gamma, y, eta=0.1, steps=5, kernel="bias")

    assert out["theta_xx"].shape == (4, 4)
    assert out["y_pred"].shape == (6, 4)  # includes step 0
    assert out["r"].shape == (6, 4)
    assert out["loss"].shape == (6,)


def test_run_kernel_gd_zero_init_has_correct_initial_state():
    gamma = jnp.array([0.2, 1.4, 2.7])
    y = jnp.array([1.0, -2.0, 0.5])

    out = run_kernel_gd(gamma, y, eta=0.1, steps=3, kernel="bias", y_pred_0=None)

    assert jnp.allclose(out["y_pred"][0], jnp.zeros_like(y))
    assert jnp.allclose(out["r"][0], -y)
    assert jnp.allclose(out["loss"][0], 0.5 * jnp.sum(y**2))


def test_run_kernel_gd_first_step_matches_kernel_gd_step():
    gamma = jnp.array([0.3, 1.0, 2.5])
    y = jnp.array([1.0, 0.0, -1.0])
    eta = 0.15

    out = run_kernel_gd(gamma, y, eta=eta, steps=1, kernel="bias", y_pred_0=None)

    theta_xx = kernel_matrix_from_gamma(gamma, kernel="bias")
    y_pred_0 = jnp.zeros_like(y)

    y_pred_1_expected, r1_expected = kernel_gd_step(y_pred_0, y, theta_xx, eta)

    assert jnp.allclose(out["theta_xx"], theta_xx)
    assert jnp.allclose(out["y_pred"][1], y_pred_1_expected)
    assert jnp.allclose(out["r"][1], r1_expected)


def test_run_kernel_gd_residual_follows_linear_dynamics():
    gamma = jnp.array([0.0, 1.0, 2.0])
    y = jnp.array([1.0, -1.0, 2.0])
    eta = 0.05
    steps = 4

    out = run_kernel_gd(gamma, y, eta=eta, steps=steps, kernel="bias", y_pred_0=None)

    theta_xx = out["theta_xx"]
    n = y.shape[0]
    A = jnp.eye(n) - eta * theta_xx

    r_expected = [-y]
    r = -y
    for _ in range(steps):
        r = A @ r
        r_expected.append(r)
    r_expected = jnp.stack(r_expected)

    assert jnp.allclose(out["r"], r_expected)


def test_run_kernel_gd_loss_matches_residual_definition():
    gamma = jnp.array([0.4, 2.0, 5.0])
    y = jnp.array([1.5, -0.5, 0.25])

    out = run_kernel_gd(gamma, y, eta=0.1, steps=5, kernel="bias", y_pred_0=None)

    expected_loss = 0.5 * jnp.sum(out["r"] ** 2, axis=1)

    assert jnp.allclose(out["loss"], expected_loss)


def test_run_kernel_gd_loss_decreases_for_small_step_size():
    gamma = jnp.linspace(0.0, 2.0 * jnp.pi, 8, endpoint=False)
    y = jnp.array([1.0, -1.0, 0.5, 0.0, 2.0, -0.5, 1.5, -2.0])

    theta_xx = kernel_matrix_from_gamma(gamma, kernel="bias")
    lam_max = np.max(np.linalg.eigvalsh(np.asarray(theta_xx)))

    eta = 0.9 * (2.0 / lam_max) * 0.5

    out = run_kernel_gd(
        gamma, y, eta=float(eta), steps=10, kernel="bias", y_pred_0=None
    )

    diffs = np.diff(np.asarray(out["loss"]))
    assert np.all(diffs <= 1e-8), diffs
