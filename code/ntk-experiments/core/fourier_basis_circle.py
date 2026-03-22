# ------------------------------------------------------------
# core/fourier_basis_circle.py
# ------------------------------------------------------------

from __future__ import annotations

from typing import List

import jax.numpy as jnp


def _safe_normalize_columns(Phi: jnp.ndarray, eps: float = 1e-12) -> jnp.ndarray:
    """
    Column-wise Euclidean normalization.

    This is for diagnostics / projections only.
    Do NOT use these normalized columns for the lemma objects G, H, or A Phi.
    """
    norms = jnp.linalg.norm(Phi, axis=0, keepdims=True)
    return Phi / (norms + eps)


def fourier_mode_metadata(K_max: int) -> dict:
    """
    Metadata for the real Fourier basis on S^1 up to frequency K_max.

    Basis ordering:
        [1, sqrt(2) cos(gamma), sqrt(2) sin(gamma), ..., sqrt(2) cos(K_max gamma), sqrt(2) sin(K_max gamma)]

    Returns:
        dict with:
            mode_names: list[str]
            mode_freqs: jnp.ndarray shape [d]
            mode_types: list[str]
    """
    if K_max < 0:
        raise ValueError(f"K_max must be >= 0, got {K_max}.")

    mode_names: List[str] = ["const"]
    mode_freqs: List[int] = [0]
    mode_types: List[str] = ["const"]

    for k in range(1, K_max + 1):
        mode_names.append(f"cos_{k}")
        mode_freqs.append(k)
        mode_types.append("cos")

        mode_names.append(f"sin_{k}")
        mode_freqs.append(k)
        mode_types.append("sin")

    return {
        "mode_names": mode_names,
        "mode_freqs": jnp.asarray(mode_freqs, dtype=jnp.int32),
        "mode_types": mode_types,
    }


def build_real_fourier_basis(
    gamma: jnp.ndarray,
    K_max: int,
) -> dict:
    """
    Build the continuum-orthonormal real Fourier basis matrix on S^1 evaluated at sampled angles.

    Columns are:
        phi_0(gamma)   = 1
        phi_{k,c}(g)   = sqrt(2) cos(k g)
        phi_{k,s}(g)   = sqrt(2) sin(k g),   k=1,...,K_max

    These functions are orthonormal in L^2(S^1, dgamma / 2pi).

    Args:
        gamma: shape [n], angles in radians.
        K_max: maximum Fourier frequency.

    Returns:
        dict with:
            Phi: shape [n, d]
            Phi_unit: shape [n, d], sample-l2 normalized columns (diagnostics only)
            mode_names: list[str]
            mode_freqs: shape [d]
            mode_types: list[str]
    """
    gamma = jnp.asarray(gamma)

    if gamma.ndim != 1:
        raise ValueError(f"gamma must be a 1D array, got shape {gamma.shape}.")
    if K_max < 0:
        raise ValueError(f"K_max must be >= 0, got {K_max}.")

    cols = [jnp.ones_like(gamma)]

    sqrt2 = jnp.sqrt(2.0)
    for k in range(1, K_max + 1):
        cols.append(sqrt2 * jnp.cos(k * gamma))
        cols.append(sqrt2 * jnp.sin(k * gamma))

    Phi = jnp.stack(cols, axis=1)

    meta = fourier_mode_metadata(K_max)

    return {
        "Phi": Phi,
        "Phi_unit": _safe_normalize_columns(Phi),
        "mode_names": meta["mode_names"],
        "mode_freqs": meta["mode_freqs"],
        "mode_types": meta["mode_types"],
    }
