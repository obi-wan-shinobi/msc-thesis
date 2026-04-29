# ------------------------------------------------------------
# experiments/ntk_frozen_block_concentration_on_S1.py
# ------------------------------------------------------------
"""
Monte Carlo concentration study for the frozen finite-width NTK block on S^1.

Goal
----
For a fixed low-frequency Fourier subspace H_K (dim d=2K+1), estimate how
the projected empirical frozen operator

    B_m^(K) := P_K T_0 P_K

concentrates around the continuum diagonal reference

    Lambda_K = diag(lambda_0, lambda_1, lambda_1, ..., lambda_K, lambda_K)

as width m increases.

Practical discretization
------------------------
We work on an evenly spaced grid gamma_i on S^1 and use the sampled Fourier
block directly:

    B_m^(K) = (1/n) Phi^T A Phi,

where A = K_emp / n and Phi evaluates the real Fourier basis functions
[1, sqrt(2) cos(k gamma), sqrt(2) sin(k gamma)]_{k=1..K}.

The script saves both:
  - empirical finite-width concentration metrics,
  - deterministic grid-to-continuum mismatch (B_infty_grid - Lambda_K)

so random-width effects can be separated from discretization effects.

Primary error object throughout this script:

    B_m^(K) - Lambda_K

(referred to in code variable names as bm_minus_lambda).
"""

from __future__ import annotations

import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
import yaml

from core.analysis import empirical_ntk_matrix
from core.fourier_basis_circle import build_real_fourier_basis
from core.kernel_analysis import continuum_fourier_eigenvalues_bias
from core.kernel_circle import kernel_matrix_from_gamma
from core.model import build_mlp_custom
from utils.artifacts import make_run_dir, save_json, save_npz, write_config_copy

TWO_PI = 2.0 * jnp.pi


def _resolve_seed_list(seed_cfg, base_seed: int) -> list[int]:
    if isinstance(seed_cfg, int):
        return [base_seed + s for s in range(seed_cfg)]
    return [int(s) for s in seed_cfg]


def _build_circle_grid(n_points: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    gamma = jnp.linspace(0.0, TWO_PI, num=n_points, endpoint=False)
    X = jnp.stack([jnp.cos(gamma), jnp.sin(gamma)], axis=1)
    return gamma, X


def _matrix_norms(M: np.ndarray) -> tuple[float, float, float]:
    M = np.asarray(M, dtype=np.float64)
    max_norm = float(np.max(np.abs(M)))
    fro_norm = float(np.linalg.norm(M, ord="fro"))
    op_norm = float(np.linalg.norm(M, ord=2))
    return max_norm, fro_norm, op_norm


def _expand_lambda_diag(mode_freqs: np.ndarray, lambda_by_k: np.ndarray) -> np.ndarray:
    mode_freqs = np.asarray(mode_freqs, dtype=np.int32)
    lambda_by_k = np.asarray(lambda_by_k, dtype=np.float64)

    if mode_freqs.ndim != 1:
        raise ValueError(f"mode_freqs must be 1D, got shape {mode_freqs.shape}.")
    if lambda_by_k.ndim != 1:
        raise ValueError(f"lambda_by_k must be 1D, got shape {lambda_by_k.shape}.")

    max_freq = int(np.max(mode_freqs))
    if max_freq >= lambda_by_k.shape[0]:
        raise ValueError(
            f"lambda_by_k is too short for mode_freqs: need >= {max_freq + 1}, "
            f"got {lambda_by_k.shape[0]}."
        )

    return lambda_by_k[mode_freqs]


def _theory_tail_bound(
    m: int,
    eps_grid: np.ndarray,
    d: int,
    subexp_L: float,
    c_value: float,
) -> np.ndarray:
    """
    Bound family used in notebook overlays:

        P(||B_m - Lambda||_max >= eps)
        <= 2 d^2 exp( - c m min(eps^2/L^2, eps/L) )

    where L and c are treated as tunable constants for empirical comparison.
    """
    eps_grid = np.asarray(eps_grid, dtype=np.float64)
    term = np.minimum((eps_grid**2) / (subexp_L**2), eps_grid / subexp_L)
    rhs = 2.0 * (d**2) * np.exp(-c_value * float(m) * term)
    return np.minimum(rhs, 1.0)


def _one_neuron_y_stats(params) -> dict:
    """
    Compute diagnostics for Y_r = 2||w_r||^2 + 4 a_r^2 from initialization params.

    For build_mlp_custom(depth_hidden=1), params has structure:
        [(W1, b1), (), (Wout, None)]
    with W1 shape [2, m], Wout shape [m, 1].
    """
    try:
        W1 = np.asarray(params[0][0], dtype=np.float64)
        Wout = np.asarray(params[-1][0], dtype=np.float64)
    except Exception as exc:
        raise ValueError(
            "Could not parse parameter tree for one-neuron diagnostics."
        ) from exc

    if W1.ndim != 2:
        raise ValueError(f"Expected W1 to be rank-2, got shape {W1.shape}.")
    if Wout.ndim != 2 or Wout.shape[1] != 1:
        raise ValueError(f"Expected Wout shape [m,1], got {Wout.shape}.")
    if W1.shape[1] != Wout.shape[0]:
        raise ValueError(
            f"Hidden width mismatch: W1 has {W1.shape[1]} columns, "
            f"Wout has {Wout.shape[0]} rows."
        )

    a = Wout[:, 0]
    w_norm_sq = np.sum(W1**2, axis=0)
    Y = 2.0 * w_norm_sq + 4.0 * (a**2)

    return {
        "y_mean": float(np.mean(Y)),
        "y_std": float(np.std(Y)),
        "y_max": float(np.max(Y)),
        "mgf_1_over_16": float(np.mean(np.exp(Y / 16.0))),
    }


def run(config_path: str):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    exp_cfg = cfg["experiment"]
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    basis_cfg = cfg["fourier_basis"]
    sweep_cfg = cfg["sweep"]
    analysis_cfg = cfg.get("analysis", {})
    artifacts_cfg = cfg.get("artifacts", {})

    base_seed = int(exp_cfg.get("seed", 0))

    n_points = int(data_cfg["n_points"])
    if n_points <= 0:
        raise ValueError(f"data.n_points must be positive, got {n_points}.")

    K_max = int(basis_cfg["max_frequency"])
    if K_max < 0:
        raise ValueError(
            f"fourier_basis.max_frequency must be nonnegative, got {K_max}."
        )

    d = 2 * K_max + 1
    if d >= n_points:
        raise ValueError(
            f"Need d=2*K_max+1 < n_points for a stable projected block. "
            f"Got d={d}, n_points={n_points}."
        )

    depth_hidden = int(model_cfg.get("depth_hidden", 1))
    if depth_hidden != 1:
        raise ValueError(
            "This theorem experiment is written for a one-hidden-layer model "
            f"(depth_hidden=1), got {depth_hidden}."
        )

    parameterization = str(model_cfg.get("parameterization", "ntk"))
    b_std = float(model_cfg.get("b_std", 1.0))

    widths = [int(w) for w in sweep_cfg["widths"]]
    if len(widths) == 0:
        raise ValueError("sweep.widths must not be empty.")

    seed_list = _resolve_seed_list(sweep_cfg["seeds"], base_seed)
    if len(seed_list) == 0:
        raise ValueError("sweep.seeds resolves to an empty seed list.")

    eps_grid = np.asarray(analysis_cfg.get("eps_grid", [0.01, 0.02, 0.05, 0.1]))
    eps_grid = np.unique(eps_grid.astype(np.float64))
    if np.any(eps_grid <= 0.0):
        raise ValueError("analysis.eps_grid must contain strictly positive values.")

    c_grid = np.asarray(analysis_cfg.get("c_grid", [0.05, 0.1, 0.2, 0.5, 1.0]))
    c_grid = np.unique(c_grid.astype(np.float64))
    if np.any(c_grid <= 0.0):
        raise ValueError("analysis.c_grid must contain strictly positive values.")

    subexp_L = float(analysis_cfg.get("subexp_L", 32.0))
    if subexp_L <= 0.0:
        raise ValueError(f"analysis.subexp_L must be positive, got {subexp_L}.")

    compute_one_neuron_stats = bool(analysis_cfg.get("compute_one_neuron_stats", True))
    save_full_blocks = bool(artifacts_cfg.get("save_full_blocks", True))

    save_dir = make_run_dir(exp_cfg.get("save_dir", "results"), exp_cfg["name"])
    write_config_copy(save_dir, cfg)

    print("=== Frozen NTK block concentration on S^1 ===")
    print(f"Saving results to: {save_dir}")
    print(f"n_points={n_points}, K_max={K_max}, d={d}")
    print(f"widths={widths}, seeds={seed_list}")
    print(f"eps_grid={eps_grid.tolist()}")
    print(f"c_grid={c_grid.tolist()}, subexp_L={subexp_L}")
    print(f"save_full_blocks={save_full_blocks}")
    print(f"compute_one_neuron_stats={compute_one_neuron_stats}\n")
    t0 = time.time()

    # ------------------------------------------------------------
    # Geometry + Fourier basis
    # ------------------------------------------------------------
    gamma, X = _build_circle_grid(n_points)
    basis = build_real_fourier_basis(gamma, K_max=K_max)

    Phi = np.asarray(basis["Phi"], dtype=np.float64)
    mode_names = np.asarray(basis["mode_names"])
    mode_freqs = np.asarray(basis["mode_freqs"], dtype=np.int32)
    mode_types = np.asarray(basis["mode_types"])

    # ------------------------------------------------------------
    # Continuum reference block Lambda_K
    # ------------------------------------------------------------
    lambda_by_k = np.asarray(
        continuum_fourier_eigenvalues_bias(jnp.arange(K_max + 1)),
        dtype=np.float64,
    )
    lambda_diag = _expand_lambda_diag(mode_freqs, lambda_by_k)
    Lambda_K = np.diag(lambda_diag)

    # Deterministic grid projection of analytic infinite-width kernel
    # used to separate discretization from finite-width randomness.
    A_infty_grid = np.asarray(
        kernel_matrix_from_gamma(gamma, kernel="bias"),
        dtype=np.float64,
    ) / float(n_points)
    B_infty_grid = (Phi.T @ A_infty_grid @ Phi) / float(n_points)
    Bm_minus_Lambda_grid = B_infty_grid - Lambda_K
    (
        bm_minus_lambda_grid_max,
        bm_minus_lambda_grid_fro,
        bm_minus_lambda_grid_op,
    ) = _matrix_norms(Bm_minus_Lambda_grid)

    save_npz(
        save_dir / "probe_geometry.npz",
        gamma=np.asarray(gamma),
        X=np.asarray(X),
    )

    save_npz(
        save_dir / "basis_metadata.npz",
        mode_names=mode_names,
        mode_freqs=mode_freqs,
        mode_types=mode_types,
        Phi=Phi.astype(np.float32),
    )

    save_npz(
        save_dir / "theory_reference.npz",
        lambda_by_k=lambda_by_k.astype(np.float32),
        lambda_diag=lambda_diag.astype(np.float32),
        Lambda_K=Lambda_K.astype(np.float32),
        A_infty_grid=A_infty_grid.astype(np.float32),
        B_infty_grid=B_infty_grid.astype(np.float32),
        Bm_minus_Lambda_grid=Bm_minus_Lambda_grid.astype(np.float32),
        bm_minus_lambda_grid_max=np.asarray(
            [bm_minus_lambda_grid_max], dtype=np.float32
        ),
        bm_minus_lambda_grid_fro=np.asarray(
            [bm_minus_lambda_grid_fro], dtype=np.float32
        ),
        bm_minus_lambda_grid_op=np.asarray([bm_minus_lambda_grid_op], dtype=np.float32),
    )

    # ------------------------------------------------------------
    # Width sweep
    # ------------------------------------------------------------
    runs_manifest = {}
    runs_dir = save_dir / "runs"
    runs_dir.mkdir(exist_ok=True)

    n_seed = len(seed_list)
    n_eps = len(eps_grid)
    n_c = len(c_grid)
    n_width = len(widths)

    err_max_mean_by_width = np.full((n_width,), np.nan, dtype=np.float64)
    err_max_std_by_width = np.full((n_width,), np.nan, dtype=np.float64)
    err_fro_mean_by_width = np.full((n_width,), np.nan, dtype=np.float64)
    err_fro_std_by_width = np.full((n_width,), np.nan, dtype=np.float64)
    err_op_mean_by_width = np.full((n_width,), np.nan, dtype=np.float64)
    err_op_std_by_width = np.full((n_width,), np.nan, dtype=np.float64)

    err_max_q10_by_width = np.full((n_width,), np.nan, dtype=np.float64)
    err_max_q50_by_width = np.full((n_width,), np.nan, dtype=np.float64)
    err_max_q90_by_width = np.full((n_width,), np.nan, dtype=np.float64)

    exceed_prob_max_by_width = np.full((n_width, n_eps), np.nan, dtype=np.float64)
    theory_bound_max_by_width = np.full((n_width, n_c, n_eps), np.nan, dtype=np.float64)

    mean_block_by_width = np.full((n_width, d, d), np.nan, dtype=np.float64)
    mean_bm_minus_lambda_by_width = np.full((n_width, d, d), np.nan, dtype=np.float64)
    mean_bm_minus_binfty_grid_by_width = np.full(
        (n_width, d, d),
        np.nan,
        dtype=np.float64,
    )
    mean_bm_minus_lambda_max_by_width = np.full((n_width,), np.nan, dtype=np.float64)
    mean_bm_minus_lambda_fro_by_width = np.full((n_width,), np.nan, dtype=np.float64)
    mean_bm_minus_lambda_op_by_width = np.full((n_width,), np.nan, dtype=np.float64)

    mgf_1_over_16_mean_by_width = np.full((n_width,), np.nan, dtype=np.float64)
    mgf_1_over_16_std_by_width = np.full((n_width,), np.nan, dtype=np.float64)

    for w_idx, width in enumerate(widths):
        print(f"Width = {width}")

        init_fn, apply_fn, _ = build_mlp_custom(
            width=width,
            depth_hidden=depth_hidden,
            b_std=b_std,
            parameterization=parameterization,
        )

        err_max = np.full((n_seed,), np.nan, dtype=np.float64)
        err_fro = np.full((n_seed,), np.nan, dtype=np.float64)
        err_op = np.full((n_seed,), np.nan, dtype=np.float64)

        err_rand_max = np.full((n_seed,), np.nan, dtype=np.float64)
        err_rand_fro = np.full((n_seed,), np.nan, dtype=np.float64)
        err_rand_op = np.full((n_seed,), np.nan, dtype=np.float64)

        diag_abs_max = np.full((n_seed,), np.nan, dtype=np.float64)
        offdiag_abs_max = np.full((n_seed,), np.nan, dtype=np.float64)

        mgf_1_over_16 = None
        y_mean = None
        y_std = None
        y_max = None
        if compute_one_neuron_stats:
            mgf_1_over_16 = np.full((n_seed,), np.nan, dtype=np.float64)
            y_mean = np.full((n_seed,), np.nan, dtype=np.float64)
            y_std = np.full((n_seed,), np.nan, dtype=np.float64)
            y_max = np.full((n_seed,), np.nan, dtype=np.float64)

        B_blocks = None
        Bm_minus_Lambda_blocks = None
        Bm_minus_Binfty_grid_blocks = None
        if save_full_blocks:
            B_blocks = np.full((n_seed, d, d), np.nan, dtype=np.float32)
            Bm_minus_Lambda_blocks = np.full((n_seed, d, d), np.nan, dtype=np.float32)
            Bm_minus_Binfty_grid_blocks = np.full(
                (n_seed, d, d),
                np.nan,
                dtype=np.float32,
            )

        sum_B = np.zeros((d, d), dtype=np.float64)
        sum_bm_minus_lambda = np.zeros((d, d), dtype=np.float64)
        sum_bm_minus_binfty_grid = np.zeros((d, d), dtype=np.float64)

        for s_idx, seed in enumerate(seed_list):
            print(f"  Seed {seed}")

            _, params = init_fn(jax.random.PRNGKey(seed), X.shape)

            A_emp = np.asarray(
                empirical_ntk_matrix(apply_fn, params, X).squeeze(),
                dtype=np.float64,
            ) / float(n_points)

            B_emp = (Phi.T @ A_emp @ Phi) / float(n_points)

            bm_minus_lambda = B_emp - Lambda_K
            bm_minus_binfty_grid = B_emp - B_infty_grid

            e_max, e_fro, e_op = _matrix_norms(bm_minus_lambda)
            r_max, r_fro, r_op = _matrix_norms(bm_minus_binfty_grid)

            err_max[s_idx] = e_max
            err_fro[s_idx] = e_fro
            err_op[s_idx] = e_op

            err_rand_max[s_idx] = r_max
            err_rand_fro[s_idx] = r_fro
            err_rand_op[s_idx] = r_op

            diag_abs_max[s_idx] = float(np.max(np.abs(np.diag(bm_minus_lambda))))
            offdiag_abs_max[s_idx] = float(
                np.max(np.abs(bm_minus_lambda - np.diag(np.diag(bm_minus_lambda))))
            )

            if compute_one_neuron_stats:
                stats = _one_neuron_y_stats(params)
                y_mean[s_idx] = stats["y_mean"]
                y_std[s_idx] = stats["y_std"]
                y_max[s_idx] = stats["y_max"]
                mgf_1_over_16[s_idx] = stats["mgf_1_over_16"]

            if save_full_blocks:
                B_blocks[s_idx, :, :] = B_emp.astype(np.float32)
                Bm_minus_Lambda_blocks[s_idx, :, :] = bm_minus_lambda.astype(np.float32)
                Bm_minus_Binfty_grid_blocks[s_idx, :, :] = bm_minus_binfty_grid.astype(
                    np.float32
                )

            sum_B += B_emp
            sum_bm_minus_lambda += bm_minus_lambda
            sum_bm_minus_binfty_grid += bm_minus_binfty_grid

        mean_B = sum_B / float(n_seed)
        mean_bm_minus_lambda = sum_bm_minus_lambda / float(n_seed)
        mean_bm_minus_binfty_grid = sum_bm_minus_binfty_grid / float(n_seed)

        mean_block_by_width[w_idx, :, :] = mean_B
        mean_bm_minus_lambda_by_width[w_idx, :, :] = mean_bm_minus_lambda
        mean_bm_minus_binfty_grid_by_width[w_idx, :, :] = mean_bm_minus_binfty_grid

        md_max, md_fro, md_op = _matrix_norms(mean_bm_minus_lambda)
        mean_bm_minus_lambda_max_by_width[w_idx] = md_max
        mean_bm_minus_lambda_fro_by_width[w_idx] = md_fro
        mean_bm_minus_lambda_op_by_width[w_idx] = md_op

        err_max_mean_by_width[w_idx] = float(np.mean(err_max))
        err_max_std_by_width[w_idx] = float(np.std(err_max))
        err_fro_mean_by_width[w_idx] = float(np.mean(err_fro))
        err_fro_std_by_width[w_idx] = float(np.std(err_fro))
        err_op_mean_by_width[w_idx] = float(np.mean(err_op))
        err_op_std_by_width[w_idx] = float(np.std(err_op))

        err_max_q10_by_width[w_idx] = float(np.quantile(err_max, 0.10))
        err_max_q50_by_width[w_idx] = float(np.quantile(err_max, 0.50))
        err_max_q90_by_width[w_idx] = float(np.quantile(err_max, 0.90))

        exceed_prob_max_by_width[w_idx, :] = np.mean(
            err_max[:, None] >= eps_grid[None, :],
            axis=0,
        )

        for c_idx, c_value in enumerate(c_grid):
            theory_bound_max_by_width[w_idx, c_idx, :] = _theory_tail_bound(
                m=width,
                eps_grid=eps_grid,
                d=d,
                subexp_L=subexp_L,
                c_value=float(c_value),
            )

        if compute_one_neuron_stats:
            mgf_1_over_16_mean_by_width[w_idx] = float(np.mean(mgf_1_over_16))
            mgf_1_over_16_std_by_width[w_idx] = float(np.std(mgf_1_over_16))

        width_payload = {
            "width": np.asarray([width], dtype=np.int32),
            "seeds": np.asarray(seed_list, dtype=np.int32),
            "err_max": err_max.astype(np.float32),
            "err_fro": err_fro.astype(np.float32),
            "err_op": err_op.astype(np.float32),
            "err_rand_max": err_rand_max.astype(np.float32),
            "err_rand_fro": err_rand_fro.astype(np.float32),
            "err_rand_op": err_rand_op.astype(np.float32),
            "diag_abs_max": diag_abs_max.astype(np.float32),
            "offdiag_abs_max": offdiag_abs_max.astype(np.float32),
            "mean_block": mean_B.astype(np.float32),
            "mean_bm_minus_lambda": mean_bm_minus_lambda.astype(np.float32),
            "mean_bm_minus_binfty_grid": mean_bm_minus_binfty_grid.astype(np.float32),
            "eps_grid": eps_grid.astype(np.float32),
            "exceed_prob_max": exceed_prob_max_by_width[w_idx, :].astype(np.float32),
            "c_grid": c_grid.astype(np.float32),
            "theory_bound_max": theory_bound_max_by_width[w_idx, :, :].astype(
                np.float32
            ),
        }

        if compute_one_neuron_stats:
            width_payload.update(
                {
                    "y_mean": y_mean.astype(np.float32),
                    "y_std": y_std.astype(np.float32),
                    "y_max": y_max.astype(np.float32),
                    "mgf_1_over_16": mgf_1_over_16.astype(np.float32),
                }
            )

        if save_full_blocks:
            width_payload.update(
                {
                    "B_blocks": B_blocks,
                    "Bm_minus_Lambda_blocks": Bm_minus_Lambda_blocks,
                    "Bm_minus_Binfty_grid_blocks": Bm_minus_Binfty_grid_blocks,
                }
            )

        width_path = runs_dir / f"width_{width}.npz"
        save_npz(width_path, **width_payload)
        runs_manifest[str(width)] = f"runs/width_{width}.npz"

    save_npz(
        save_dir / "summary.npz",
        widths=np.asarray(widths, dtype=np.int32),
        eps_grid=eps_grid.astype(np.float32),
        c_grid=c_grid.astype(np.float32),
        err_max_mean=err_max_mean_by_width.astype(np.float32),
        err_max_std=err_max_std_by_width.astype(np.float32),
        err_fro_mean=err_fro_mean_by_width.astype(np.float32),
        err_fro_std=err_fro_std_by_width.astype(np.float32),
        err_op_mean=err_op_mean_by_width.astype(np.float32),
        err_op_std=err_op_std_by_width.astype(np.float32),
        err_max_q10=err_max_q10_by_width.astype(np.float32),
        err_max_q50=err_max_q50_by_width.astype(np.float32),
        err_max_q90=err_max_q90_by_width.astype(np.float32),
        exceed_prob_max_by_width=exceed_prob_max_by_width.astype(np.float32),
        theory_bound_max_by_width=theory_bound_max_by_width.astype(np.float32),
        mean_block_by_width=mean_block_by_width.astype(np.float32),
        mean_bm_minus_lambda_by_width=mean_bm_minus_lambda_by_width.astype(np.float32),
        mean_bm_minus_binfty_grid_by_width=mean_bm_minus_binfty_grid_by_width.astype(
            np.float32
        ),
        mean_bm_minus_lambda_max_by_width=mean_bm_minus_lambda_max_by_width.astype(
            np.float32
        ),
        mean_bm_minus_lambda_fro_by_width=mean_bm_minus_lambda_fro_by_width.astype(
            np.float32
        ),
        mean_bm_minus_lambda_op_by_width=mean_bm_minus_lambda_op_by_width.astype(
            np.float32
        ),
        mgf_1_over_16_mean_by_width=mgf_1_over_16_mean_by_width.astype(np.float32),
        mgf_1_over_16_std_by_width=mgf_1_over_16_std_by_width.astype(np.float32),
    )

    manifest = {
        "probe_geometry": "probe_geometry.npz",
        "basis_metadata": "basis_metadata.npz",
        "theory_reference": "theory_reference.npz",
        "summary": "summary.npz",
        "runs": runs_manifest,
        "meta": {
            "n_points": n_points,
            "K_max": K_max,
            "d": d,
            "widths": widths,
            "seeds": seed_list,
            "depth_hidden": depth_hidden,
            "parameterization": parameterization,
            "b_std": b_std,
            "kernel": "bias",
            "eps_grid": eps_grid.tolist(),
            "c_grid": c_grid.tolist(),
            "subexp_L": subexp_L,
            "save_full_blocks": save_full_blocks,
            "compute_one_neuron_stats": compute_one_neuron_stats,
            "theory_bound_form": (
                "2 d^2 exp(-c m min(eps^2/L^2, eps/L)) with L=subexp_L and c in c_grid"
            ),
            "target_object": "B_m^(K) = P_K T_0 P_K",
            "block_coordinates": "B_m^(K) = (1/n) Phi^T A Phi",
            "reference_object": "Lambda_K = diag(lambda_0,lambda_1,lambda_1,...,lambda_K,lambda_K)",
        },
        "diagnostics": {
            "bm_minus_lambda_grid_max": bm_minus_lambda_grid_max,
            "bm_minus_lambda_grid_fro": bm_minus_lambda_grid_fro,
            "bm_minus_lambda_grid_op": bm_minus_lambda_grid_op,
        },
        "runtime_sec": round(time.time() - t0, 2),
    }

    save_json(save_dir / "manifest.json", manifest)
    print(f"\nDone. Frozen block concentration artifacts saved to {save_dir}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(
            "Usage: python3 -m experiments.ntk_frozen_block_concentration_on_S1 "
            "configs/ntk_frozen_block_concentration_on_S1.yaml"
        )
    else:
        run(sys.argv[1])
