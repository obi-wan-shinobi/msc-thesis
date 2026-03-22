# ------------------------------------------------------------
# tests/test_fourier_basis_circle.py
# ------------------------------------------------------------

import jax.numpy as jnp
import numpy as np

from core.fourier_basis_circle import build_real_fourier_basis, fourier_mode_metadata


def test_fourier_mode_metadata_K0():
    meta = fourier_mode_metadata(0)

    assert meta["mode_names"] == ["const"]
    assert meta["mode_types"] == ["const"]
    np.testing.assert_array_equal(np.asarray(meta["mode_freqs"]), np.array([0]))


def test_fourier_mode_metadata_K3():
    meta = fourier_mode_metadata(3)

    assert meta["mode_names"] == [
        "const",
        "cos_1",
        "sin_1",
        "cos_2",
        "sin_2",
        "cos_3",
        "sin_3",
    ]
    assert meta["mode_types"] == [
        "const",
        "cos",
        "sin",
        "cos",
        "sin",
        "cos",
        "sin",
    ]
    np.testing.assert_array_equal(
        np.asarray(meta["mode_freqs"]),
        np.array([0, 1, 1, 2, 2, 3, 3]),
    )


def test_build_real_fourier_basis_shapes():
    gamma = jnp.array([0.0, 0.5, 1.0, 1.5])
    out = build_real_fourier_basis(gamma, K_max=4)

    Phi = out["Phi"]
    Phi_unit = out["Phi_unit"]

    n = gamma.shape[0]
    d = 2 * 4 + 1

    assert Phi.shape == (n, d)
    assert Phi_unit.shape == (n, d)
    assert len(out["mode_names"]) == d
    assert len(out["mode_types"]) == d
    assert out["mode_freqs"].shape == (d,)


def test_build_real_fourier_basis_first_columns_match_definition():
    gamma = jnp.array([0.0, jnp.pi / 3, jnp.pi / 2])
    out = build_real_fourier_basis(gamma, K_max=2)
    Phi = out["Phi"]

    sqrt2 = np.sqrt(2.0)

    expected = np.stack(
        [
            np.ones(3),
            sqrt2 * np.cos(np.array(gamma)),
            sqrt2 * np.sin(np.array(gamma)),
            sqrt2 * np.cos(2.0 * np.array(gamma)),
            sqrt2 * np.sin(2.0 * np.array(gamma)),
        ],
        axis=1,
    )

    np.testing.assert_allclose(np.asarray(Phi), expected, atol=1e-7, rtol=1e-7)


def test_phi_unit_columns_have_unit_norm():
    gamma = jnp.linspace(0.0, 2.0 * jnp.pi, 32, endpoint=False)
    out = build_real_fourier_basis(gamma, K_max=5)

    Phi_unit = np.asarray(out["Phi_unit"])
    col_norms = np.linalg.norm(Phi_unit, axis=0)

    np.testing.assert_allclose(col_norms, np.ones_like(col_norms), atol=1e-6, rtol=1e-6)


def test_dense_grid_empirical_gram_is_close_to_identity():
    """
    On an evenly spaced dense grid, (1/n) Phi^T Phi should be very close to I
    for the continuum-orthonormal real Fourier basis.
    """
    n = 4096
    K_max = 12

    gamma = jnp.linspace(0.0, 2.0 * jnp.pi, n, endpoint=False)
    out = build_real_fourier_basis(gamma, K_max=K_max)
    Phi = out["Phi"]

    G = (Phi.T @ Phi) / n
    I = jnp.eye(2 * K_max + 1)

    np.testing.assert_allclose(np.asarray(G), np.asarray(I), atol=5e-4, rtol=5e-4)


def test_Kmax_zero_returns_constant_only():
    gamma = jnp.linspace(0.0, 2.0 * jnp.pi, 10, endpoint=False)
    out = build_real_fourier_basis(gamma, K_max=0)

    Phi = np.asarray(out["Phi"])

    assert Phi.shape == (10, 1)
    np.testing.assert_allclose(Phi[:, 0], np.ones(10), atol=1e-7, rtol=1e-7)
    assert out["mode_names"] == ["const"]
    assert out["mode_types"] == ["const"]
    np.testing.assert_array_equal(np.asarray(out["mode_freqs"]), np.array([0]))
