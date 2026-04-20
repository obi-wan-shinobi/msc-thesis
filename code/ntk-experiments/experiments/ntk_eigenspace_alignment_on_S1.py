# ------------------------------------------------------------
# experiments/ntk_eigenspace_alignment_on_S1.py
# ------------------------------------------------------------
"""
Experiment — Frozen NTK eigenspace alignment with Fourier subspaces on S^1

For each (width, seed), at initialization only:
    1) compute empirical frozen NTK K on an evenly spaced circle grid,
    2) form normalized operator A = K / n,
    3) eigendecompose A,
    4) compare empirical eigenspaces to Fourier frequency subspaces.

Saved artifacts are compact and focused on alignment metrics.
"""

import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
import yaml

from core.analysis import empirical_ntk_matrix
from core.fourier_basis_circle import build_real_fourier_basis
from core.kernel_analysis import continuum_fourier_eigenvalues_bias
from core.model import build_mlp_custom
from utils.artifacts import make_run_dir, save_json, save_npz, write_config_copy

TWO_PI = 2.0 * jnp.pi


def _resolve_seed_list(seed_cfg, base_seed: int) -> list[int]:
    if isinstance(seed_cfg, int):
        return [base_seed + s for s in range(seed_cfg)]
    return [int(s) for s in seed_cfg]


def _build_circle_grid(n_points: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    theta = jnp.linspace(0.0, TWO_PI, num=n_points, endpoint=False)
    X = jnp.stack([jnp.cos(theta), jnp.sin(theta)], axis=1)
    return theta, X


def _fourier_subspace_for_frequency(
    Phi_unit: jnp.ndarray,
    mode_freqs: np.ndarray,
    k: int,
) -> jnp.ndarray:
    mask = mode_freqs == k
    U = Phi_unit[:, mask]

    expected_dim = 1 if k == 0 else 2
    if U.shape[1] != expected_dim:
        raise ValueError(
            f"Frequency k={k} expected subspace dim {expected_dim}, got {U.shape[1]}."
        )

    return U


def _principal_angles(U: jnp.ndarray, V: jnp.ndarray) -> jnp.ndarray:
    # U, V: orthonormal basis columns of two subspaces with the same dimension.
    svals = jnp.linalg.svd(U.T @ V, compute_uv=False)
    svals = jnp.clip(svals, -1.0, 1.0)
    return jnp.arccos(svals)


def _projector_distances_from_angles(
    angles: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    sin_t = jnp.sin(angles)
    fro = jnp.sqrt(2.0 * jnp.sum(sin_t**2))
    op = jnp.max(sin_t)
    return fro, op


def _pad_two(values: np.ndarray) -> np.ndarray:
    out = np.full((2,), np.nan, dtype=np.float32)
    n = min(len(values), 2)
    out[:n] = np.asarray(values[:n], dtype=np.float32)
    return out


def _nanmean_std_axis0(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Nan-aware mean/std over seed axis without RuntimeWarnings on all-NaN slices.
    """
    arr = np.asarray(arr)
    out_shape = arr.shape[1:]

    mean = np.full(out_shape, np.nan, dtype=np.float32)
    std = np.full(out_shape, np.nan, dtype=np.float32)

    for idx in np.ndindex(out_shape):
        vals = arr[(slice(None),) + idx]
        vals = vals[~np.isnan(vals)]
        if vals.size > 0:
            mean[idx] = np.mean(vals)
            std[idx] = np.std(vals)

    return mean, std


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

    widths = [int(w) for w in sweep_cfg["widths"]]
    seed_list = _resolve_seed_list(sweep_cfg["seeds"], base_seed)

    depth_hidden = int(model_cfg.get("depth_hidden", 1))
    b_std = float(model_cfg.get("b_std", 1.0))
    parameterization = str(model_cfg.get("parameterization", "ntk"))

    K_max = int(basis_cfg["max_frequency"])
    projector_fro_tol = float(analysis_cfg.get("projector_fro_tol", 0.25))
    save_aligned_eigvecs = bool(artifacts_cfg.get("save_aligned_eigvecs", True))

    if n_points <= 0:
        raise ValueError(f"data.n_points must be positive, got {n_points}.")
    if K_max < 0:
        raise ValueError(
            f"fourier_basis.max_frequency must be nonnegative, got {K_max}."
        )

    frequencies = np.arange(K_max + 1, dtype=np.int32)

    save_dir = make_run_dir(exp_cfg.get("save_dir", "results"), exp_cfg["name"])
    write_config_copy(save_dir, cfg)

    print("=== NTK eigenspace alignment on S^1 ===")
    print(f"Saving results to: {save_dir}")
    print(f"n_points={n_points}, widths={widths}, seeds={seed_list}\n")
    print(f"save_aligned_eigvecs={save_aligned_eigvecs}\n")
    t0 = time.time()

    theta, X = _build_circle_grid(n_points)
    save_npz(
        save_dir / "probe_geometry.npz",
        theta=np.asarray(theta),
        X=np.asarray(X),
    )

    basis = build_real_fourier_basis(theta, K_max=K_max)
    Phi_unit = jnp.asarray(basis["Phi_unit"])
    mode_names = np.asarray(basis["mode_names"])
    mode_freqs = np.asarray(basis["mode_freqs"], dtype=np.int32)
    mode_types = np.asarray(basis["mode_types"])

    save_npz(
        save_dir / "basis_metadata.npz",
        mode_names=mode_names,
        mode_freqs=mode_freqs,
        mode_types=mode_types,
        frequencies=frequencies,
    )

    lambda_by_k = np.asarray(
        continuum_fourier_eigenvalues_bias(jnp.arange(K_max + 1)),
        dtype=np.float32,
    )

    fourier_frames = []
    for k in frequencies:
        U_k_raw = _fourier_subspace_for_frequency(Phi_unit, mode_freqs, int(k))
        U_k, _ = jnp.linalg.qr(U_k_raw)
        fourier_frames.append(U_k)

    runs_manifest = {}
    runs_dir = save_dir / "runs"

    projector_fro_mean_by_width = []
    projector_fro_std_by_width = []
    projector_op_mean_by_width = []
    projector_op_std_by_width = []
    eig_abs_error_mean_by_width = []
    eig_abs_error_std_by_width = []
    pair_splitting_mean_by_width = []
    pair_splitting_std_by_width = []
    principal_angles_mean_by_width = []
    principal_angles_std_by_width = []

    n_freq = len(frequencies)

    for width in widths:
        print(f"Width = {width}")

        n_seed = len(seed_list)
        projector_fro = np.full((n_seed, n_freq), np.nan, dtype=np.float32)
        projector_op = np.full((n_seed, n_freq), np.nan, dtype=np.float32)
        principal_angles = np.full((n_seed, n_freq, 2), np.nan, dtype=np.float32)
        eig_abs_error = np.full((n_seed, n_freq), np.nan, dtype=np.float32)
        pair_splitting = np.full((n_seed, n_freq), np.nan, dtype=np.float32)
        lambda_hat_mean = np.full((n_seed, n_freq), np.nan, dtype=np.float32)
        selected_eigvals = np.full((n_seed, n_freq, 2), np.nan, dtype=np.float32)
        selected_eig_indices = np.full((n_seed, n_freq, 2), -1, dtype=np.int32)
        fourier_coeffs_raw = np.full((n_seed, n_freq, 2, 2), np.nan, dtype=np.float32)
        fourier_coeffs_aligned = np.full(
            (n_seed, n_freq, 2, 2), np.nan, dtype=np.float32
        )
        aligned_eigvecs = None
        if save_aligned_eigvecs:
            aligned_eigvecs = np.full(
                (n_seed, n_freq, 2, n_points),
                np.nan,
                dtype=np.float32,
            )

        init_fn, apply_fn, _ = build_mlp_custom(
            width=width,
            b_std=b_std,
            depth_hidden=depth_hidden,
            parameterization=parameterization,
        )

        for s_idx, seed in enumerate(seed_list):
            print(f"  Seed {seed}")
            _, params = init_fn(jax.random.PRNGKey(seed), X.shape)

            K = empirical_ntk_matrix(apply_fn, params, X)
            A = jnp.asarray(K).squeeze() / n_points

            evals, evecs = jnp.linalg.eigh(A)
            perm = jnp.argsort(evals)[::-1]
            evals = evals[perm]
            evecs = evecs[:, perm]

            for f_idx, k in enumerate(frequencies):
                U_k = fourier_frames[f_idx]
                d_k = U_k.shape[1]

                overlaps = jnp.sum((U_k.T @ evecs) ** 2, axis=0)
                top_idx = jnp.argsort(overlaps)[::-1][:d_k]
                V_k = evecs[:, top_idx]
                M_raw = V_k.T @ U_k

                angles = _principal_angles(U_k, V_k)
                proj_fro, proj_op = _projector_distances_from_angles(angles)

                eigvals_sel = evals[top_idx]
                lambda_hat = jnp.mean(eigvals_sel)
                lambda_k = lambda_by_k[f_idx]
                eig_err = jnp.abs(lambda_hat - lambda_k)

                split = (
                    jnp.abs(eigvals_sel[0] - eigvals_sel[1])
                    if d_k == 2
                    else jnp.asarray(jnp.nan)
                )

                projector_fro[s_idx, f_idx] = float(np.asarray(proj_fro))
                projector_op[s_idx, f_idx] = float(np.asarray(proj_op))
                principal_angles[s_idx, f_idx, :] = _pad_two(np.asarray(angles))
                eig_abs_error[s_idx, f_idx] = float(np.asarray(eig_err))
                pair_splitting[s_idx, f_idx] = float(np.asarray(split))
                lambda_hat_mean[s_idx, f_idx] = float(np.asarray(lambda_hat))

                top_idx_np = np.asarray(top_idx, dtype=np.int32)
                eigvals_sel_np = np.asarray(eigvals_sel, dtype=np.float32)
                selected_eig_indices[s_idx, f_idx, : len(top_idx_np)] = top_idx_np
                selected_eigvals[s_idx, f_idx, : len(eigvals_sel_np)] = eigvals_sel_np
                fourier_coeffs_raw[s_idx, f_idx, :d_k, :d_k] = np.asarray(
                    M_raw,
                    dtype=np.float32,
                )

                U_svd, _, Vt_svd = jnp.linalg.svd(M_raw, full_matrices=False)
                R = U_svd @ Vt_svd
                V_aligned = V_k @ R

                V_aligned_np = np.asarray(V_aligned, dtype=np.float32)
                U_k_np = np.asarray(U_k, dtype=np.float32)
                for comp in range(d_k):
                    if np.dot(V_aligned_np[:, comp], U_k_np[:, comp]) < 0:
                        V_aligned_np[:, comp] *= -1.0

                if save_aligned_eigvecs:
                    aligned_eigvecs[s_idx, f_idx, :d_k, :] = V_aligned_np.T

                M_aligned = np.asarray(V_aligned_np.T @ U_k_np, dtype=np.float32)
                fourier_coeffs_aligned[s_idx, f_idx, :d_k, :d_k] = M_aligned

        width_path = runs_dir / f"width_{width}.npz"
        width_payload = {
            "width": np.asarray([width], dtype=np.int32),
            "seeds": np.asarray(seed_list, dtype=np.int32),
            "frequencies": frequencies,
            "lambda_theory": lambda_by_k,
            "projector_fro": projector_fro,
            "projector_op": projector_op,
            "principal_angles": principal_angles,
            "eig_abs_error": eig_abs_error,
            "pair_splitting": pair_splitting,
            "lambda_hat_mean": lambda_hat_mean,
            "selected_eigvals": selected_eigvals,
            "selected_eig_indices": selected_eig_indices,
            "fourier_coeffs_raw": fourier_coeffs_raw,
            "fourier_coeffs_aligned": fourier_coeffs_aligned,
        }
        if save_aligned_eigvecs:
            width_payload["aligned_eigvecs"] = aligned_eigvecs

        save_npz(width_path, **width_payload)
        runs_manifest[str(width)] = f"runs/width_{width}.npz"

        projector_fro_mean_by_width.append(np.mean(projector_fro, axis=0))
        projector_fro_std_by_width.append(np.std(projector_fro, axis=0))
        projector_op_mean_by_width.append(np.mean(projector_op, axis=0))
        projector_op_std_by_width.append(np.std(projector_op, axis=0))
        eig_abs_error_mean_by_width.append(np.mean(eig_abs_error, axis=0))
        eig_abs_error_std_by_width.append(np.std(eig_abs_error, axis=0))
        pair_mean, pair_std = _nanmean_std_axis0(pair_splitting)
        angle_mean, angle_std = _nanmean_std_axis0(principal_angles)
        pair_splitting_mean_by_width.append(pair_mean)
        pair_splitting_std_by_width.append(pair_std)
        principal_angles_mean_by_width.append(angle_mean)
        principal_angles_std_by_width.append(angle_std)

    projector_fro_mean = np.stack(projector_fro_mean_by_width, axis=0)
    projector_fro_std = np.stack(projector_fro_std_by_width, axis=0)
    projector_op_mean = np.stack(projector_op_mean_by_width, axis=0)
    projector_op_std = np.stack(projector_op_std_by_width, axis=0)
    eig_abs_error_mean = np.stack(eig_abs_error_mean_by_width, axis=0)
    eig_abs_error_std = np.stack(eig_abs_error_std_by_width, axis=0)
    pair_splitting_mean = np.stack(pair_splitting_mean_by_width, axis=0)
    pair_splitting_std = np.stack(pair_splitting_std_by_width, axis=0)
    principal_angles_mean = np.stack(principal_angles_mean_by_width, axis=0)
    principal_angles_std = np.stack(principal_angles_std_by_width, axis=0)

    min_width_for_small_error = np.full((n_freq,), -1, dtype=np.int32)
    for f_idx in range(n_freq):
        good = np.where(projector_fro_mean[:, f_idx] <= projector_fro_tol)[0]
        if len(good) > 0:
            min_width_for_small_error[f_idx] = widths[int(good[0])]

    failing_freqs_at_max_width = frequencies[
        projector_fro_mean[-1, :] > projector_fro_tol
    ]

    save_npz(
        save_dir / "summary.npz",
        widths=np.asarray(widths, dtype=np.int32),
        frequencies=frequencies,
        lambda_theory=lambda_by_k,
        projector_fro_mean=projector_fro_mean,
        projector_fro_std=projector_fro_std,
        projector_op_mean=projector_op_mean,
        projector_op_std=projector_op_std,
        principal_angles_mean=principal_angles_mean,
        principal_angles_std=principal_angles_std,
        eig_abs_error_mean=eig_abs_error_mean,
        eig_abs_error_std=eig_abs_error_std,
        pair_splitting_mean=pair_splitting_mean,
        pair_splitting_std=pair_splitting_std,
        min_width_for_small_error=min_width_for_small_error,
        failing_freqs_at_max_width=failing_freqs_at_max_width,
        projector_fro_tol=np.asarray([projector_fro_tol], dtype=np.float32),
    )

    diagnostics = {
        str(int(k)): (None if int(w) < 0 else int(w))
        for k, w in zip(frequencies.tolist(), min_width_for_small_error.tolist())
    }

    manifest = {
        "probe_geometry": "probe_geometry.npz",
        "basis_metadata": "basis_metadata.npz",
        "summary": "summary.npz",
        "runs": runs_manifest,
        "meta": {
            "n_points": n_points,
            "widths": widths,
            "seeds": seed_list,
            "depth_hidden": depth_hidden,
            "b_std": b_std,
            "parameterization": parameterization,
            "max_frequency": K_max,
            "projector_fro_tol": projector_fro_tol,
            "save_aligned_eigvecs": save_aligned_eigvecs,
        },
        "diagnostics": {
            "min_width_for_small_projector_error_by_frequency": diagnostics,
            "failing_frequencies_at_max_width": failing_freqs_at_max_width.tolist(),
        },
        "runtime_sec": round(time.time() - t0, 2),
    }

    save_json(save_dir / "manifest.json", manifest)
    print(f"\nDone. Eigenspace alignment artifacts saved to {save_dir}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(
            "Usage: python3 -m experiments.ntk_eigenspace_alignment_on_S1 "
            "configs/ntk_eigenspace_alignment_on_S1.yaml"
        )
    else:
        run(sys.argv[1])
