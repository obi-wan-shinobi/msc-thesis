import jax.numpy as jnp

from core.kernel_circle import pairwise_principal_angle, theta_bias, theta_nobias


def test_pairwise_principal_angle_shape():
    gamma1 = jnp.array([0.0, jnp.pi / 2])
    gamma2 = jnp.array([0.0, jnp.pi, 3 * jnp.pi / 2])
    delta = pairwise_principal_angle(gamma1, gamma2)
    assert delta.shape == (2, 3)


def test_pairwise_principal_angle_range():
    gamma = jnp.array([0.0, jnp.pi / 2, jnp.pi, 3 * jnp.pi / 2])
    delta = pairwise_principal_angle(gamma)
    assert jnp.all(delta >= 0.0)
    assert jnp.all(delta <= jnp.pi)


def test_pairwise_principal_angle_diagonal_zero():
    gamma = jnp.array([0.0, 1.0, 2.0])
    delta = pairwise_principal_angle(gamma)
    assert jnp.allclose(jnp.diag(delta), 0.0)


def test_pairwise_principal_angle_known_values():
    gamma1 = jnp.array([0.0])
    gamma2 = jnp.array([0.0, jnp.pi, 3 * jnp.pi / 2])
    delta = pairwise_principal_angle(gamma1, gamma2)
    expected = jnp.array([[0.0, jnp.pi, jnp.pi / 2]])
    assert jnp.allclose(delta, expected)


def test_theta_nobias_special_values():
    delta = jnp.array([0.0, jnp.pi])
    out = theta_nobias(delta)
    expected = jnp.array([1.0, 0.0])
    assert jnp.allclose(out, expected, atol=1e-7)


def test_theta_bias_special_values():
    delta = jnp.array([0.0, jnp.pi])
    out = theta_bias(delta)
    expected = jnp.array([1.5, 0.0])
    assert jnp.allclose(out, expected, atol=1e-7)


def test_theta_bias_relation():
    delta = jnp.linspace(0.0, jnp.pi, 100)
    lhs = theta_bias(delta) - theta_nobias(delta)
    rhs = (jnp.pi - delta) / (2.0 * jnp.pi)
    assert jnp.allclose(lhs, rhs, atol=1e-7)
