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
from core.data import make_fourier_target
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


def _matrix_norms(M: np.ndarray) -> tuple[float, float, float]:
    M = np.asarray(M, dtype=np.float64)
    return (
        float(np.max(np.abs(M))),
        float(np.linalg.norm(M, ord="fro")),
        float(np.linalg.norm(M, ord=2)),
    )


def _local_eigenvalue_gaps(lambda_by_k: np.ndarray) -> np.ndarray:
    vals = np.asarray(lambda_by_k, dtype=np.float64)
    gaps = np.full(vals.shape, np.nan, dtype=np.float64)

    if len(vals) == 1:
        gaps[0] = np.inf
        return gaps

    for k in range(len(vals)):
        candidates = []
        if k > 0:
            candidates.append(abs(vals[k] - vals[k - 1]))
        if k < len(vals) - 1:
            candidates.append(abs(vals[k] - vals[k + 1]))
        gaps[k] = min(candidates)

    return gaps


def _resolve_target_specs(analysis_cfg: dict, K_max: int) -> list[dict]:
    default_supports = [
        {"name": "low", "frequencies": [0, 1, 2, 3]},
        {"name": "mid", "frequencies": [0, 1, 2, 3, 4, 5, 6]},
        {"name": "high", "frequencies": [0, 1, 2, 3, 4, 5, 6, 7, 8]},
        {
            "name": "very_high",
            "frequencies": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        },
    ]

    target_items = analysis_cfg.get("targets", None)
    if target_items is None:
        target_items = []
        for item in analysis_cfg.get("target_supports", default_supports):
            freqs = [int(k) for k in item["frequencies"]]
            target_items.append(
                {
                    "name": str(item["name"]),
                    "Ks": freqs,
                    "amps": [1.0 for _ in freqs],
                    "phases": [0.0 for _ in freqs],
                }
            )

    out = []
    seen_names = set()
    for item in target_items:
        name = str(item["name"])
        if name in seen_names:
            raise ValueError(f"Duplicate target name: {name}")
        seen_names.add(name)

        Ks = [int(k) for k in item["Ks"]]
        amps = [float(a) for a in item["amps"]]
        phases = [float(p) for p in item.get("phases", [0.0 for _ in Ks])]

        if not (len(Ks) == len(amps) == len(phases)):
            raise ValueError(
                f"Target {name} must satisfy len(Ks)==len(amps)==len(phases)."
            )
        if len(Ks) == 0:
            raise ValueError(f"Target {name} must contain at least one frequency.")
        if any((k < 0 or k > K_max) for k in Ks):
            raise ValueError(
                f"Target {name} has frequency outside [0, {K_max}]: {Ks}."
            )

        unique_freqs = []
        for k in Ks:
            if k not in unique_freqs:
                unique_freqs.append(k)

        spec = make_fourier_target(Ks, amps, phases)
        out.append(
            {
                "name": name,
                "frequencies": unique_freqs,
                "Ks": np.asarray(Ks, dtype=np.int32),
                "amps": np.asarray(amps, dtype=np.float32),
                "phases": np.asarray(phases, dtype=np.float32),
                "spec": spec,
            }
        )

    return out


def _target_coefficients_from_spec(
    Ks: np.ndarray,
    amps: np.ndarray,
    phases: np.ndarray,
    mode_freqs: np.ndarray,
    mode_types: np.ndarray,
) -> np.ndarray:
    """
    Build basis coefficients c in the real Fourier basis (const, sqrt(2)cos,
    sqrt(2)sin) from target terms interpreted as

        f(gamma) = sum_j amps_j * cos(Ks_j * gamma - phases_j).

    With this convention, phase = +pi/2 maps cosine to +sine.
    """
    c = np.zeros(mode_freqs.shape[0], dtype=np.float64)
    sqrt2 = np.sqrt(2.0)

    const_idx = np.where((mode_freqs == 0) & (mode_types == "const"))[0]
    if len(const_idx) != 1:
        raise ValueError("Could not identify unique constant basis mode.")
    const_idx = int(const_idx[0])

    for k_raw, a_raw, p_raw in zip(Ks, amps, phases):
        k = int(k_raw)
        a = float(a_raw)
        p = float(p_raw)

        if k == 0:
            c[const_idx] += a * np.cos(p)
            continue

        cos_idx = np.where((mode_freqs == k) & (mode_types == "cos"))[0]
        sin_idx = np.where((mode_freqs == k) & (mode_types == "sin"))[0]
        if len(cos_idx) != 1 or len(sin_idx) != 1:
            raise ValueError(f"Could not identify unique cos/sin basis for k={k}.")

        c[int(cos_idx[0])] += a * np.cos(p) / sqrt2
        c[int(sin_idx[0])] += a * np.sin(p) / sqrt2

    norm = np.linalg.norm(c)
    if norm <= 0.0:
        raise ValueError("Target coefficient vector has zero norm.")
    return c / norm


def _fourier_action_leakage(
    B: np.ndarray,
    mode_freqs: np.ndarray,
    frequencies: np.ndarray,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Measure whether B maps each Fourier frequency plane mostly into itself.

    For frequency k, columns in the k-plane are the inputs. The diagonal
    strength is the absolute mass of the k-by-k block, while off-block mass is
    absolute output mass outside that frequency plane.
    """
    B = np.asarray(B, dtype=np.float64)
    diag_strength = np.full((len(frequencies),), np.nan, dtype=np.float64)
    off_block_mass = np.full((len(frequencies),), np.nan, dtype=np.float64)
    relative_leakage = np.full((len(frequencies),), np.nan, dtype=np.float64)

    all_idx = np.arange(B.shape[0])
    for f_idx, k in enumerate(frequencies):
        idx = np.where(mode_freqs == int(k))[0]
        outside = np.setdiff1d(all_idx, idx, assume_unique=True)

        block = B[np.ix_(idx, idx)]
        leak = B[np.ix_(outside, idx)]

        diag_strength[f_idx] = np.sum(np.abs(block))
        off_block_mass[f_idx] = np.sum(np.abs(leak))
        relative_leakage[f_idx] = off_block_mass[f_idx] / max(
            diag_strength[f_idx],
            eps,
        )

    return diag_strength, off_block_mass, relative_leakage


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
    reliability_tau = float(analysis_cfg.get("reliability_tau", 1.0))
    leakage_eps = float(analysis_cfg.get("leakage_eps", 1e-12))
    target_specs = _resolve_target_specs(analysis_cfg, K_max)
    save_aligned_eigvecs = bool(artifacts_cfg.get("save_aligned_eigvecs", True))
    save_blocks = bool(artifacts_cfg.get("save_blocks", True))

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
    print(f"save_blocks={save_blocks}")
    print(
        "target specs:",
        {s["name"]: s["frequencies"] for s in target_specs},
        "\n",
    )
    t0 = time.time()

    theta, X = _build_circle_grid(n_points)
    save_npz(
        save_dir / "probe_geometry.npz",
        theta=np.asarray(theta),
        X=np.asarray(X),
    )

    basis = build_real_fourier_basis(theta, K_max=K_max)
    Phi = np.asarray(basis["Phi"], dtype=np.float64)
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
        dtype=np.float64,
    )
    lambda_basis = lambda_by_k[mode_freqs]
    Lambda_K = np.diag(lambda_basis)
    local_gap = _local_eigenvalue_gaps(lambda_by_k)

    support_names = np.asarray([s["name"] for s in target_specs])
    support_frequencies = np.full(
        (len(target_specs), K_max + 1),
        -1,
        dtype=np.int32,
    )
    support_sizes = np.full((len(target_specs),), -1, dtype=np.int32)
    support_lambda_min = np.full((len(target_specs),), np.nan, dtype=np.float32)
    target_coefficients = np.full(
        (len(target_specs), len(mode_freqs)),
        np.nan,
        dtype=np.float32,
    )
    target_denominators = np.full((len(target_specs),), np.nan, dtype=np.float32)
    target_Ks = np.full((len(target_specs), K_max + 1), -1, dtype=np.int32)
    target_amps = np.full((len(target_specs), K_max + 1), np.nan, dtype=np.float32)
    target_phases = np.full((len(target_specs), K_max + 1), np.nan, dtype=np.float32)

    for s_idx, support_spec in enumerate(target_specs):
        freqs = support_spec["frequencies"]
        Ks = np.asarray(support_spec["Ks"], dtype=np.int32)
        amps = np.asarray(support_spec["amps"], dtype=np.float32)
        phases = np.asarray(support_spec["phases"], dtype=np.float32)

        support_sizes[s_idx] = len(freqs)
        support_frequencies[s_idx, : len(freqs)] = np.asarray(freqs, dtype=np.int32)
        support_lambda_min[s_idx] = float(np.min(lambda_by_k[freqs]))
        target_Ks[s_idx, : len(Ks)] = Ks
        target_amps[s_idx, : len(Ks)] = amps
        target_phases[s_idx, : len(Ks)] = phases

        c = _target_coefficients_from_spec(Ks, amps, phases, mode_freqs, mode_types)
        target_coefficients[s_idx, :] = c.astype(np.float32)
        target_denominators[s_idx] = float(np.linalg.norm(Lambda_K @ c))

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
    relative_center_error_mean_by_width = []
    relative_center_error_std_by_width = []
    pair_splitting_mean_by_width = []
    pair_splitting_std_by_width = []
    pair_splitting_over_lambda_mean_by_width = []
    pair_splitting_over_lambda_std_by_width = []
    pair_splitting_over_gap_mean_by_width = []
    pair_splitting_over_gap_std_by_width = []
    principal_angles_mean_by_width = []
    principal_angles_std_by_width = []
    block_op_mean_by_width = []
    block_op_std_by_width = []
    block_fro_mean_by_width = []
    block_fro_std_by_width = []
    block_max_mean_by_width = []
    block_max_std_by_width = []
    relative_leakage_mean_by_width = []
    relative_leakage_std_by_width = []
    target_prediction_error_mean_by_width = []
    target_prediction_error_std_by_width = []
    reliability_ratio_mean_by_width = []
    reliability_ratio_std_by_width = []
    support_max_projector_fro_mean_by_width = []
    support_max_projector_fro_std_by_width = []
    support_max_relative_center_error_mean_by_width = []
    support_max_relative_center_error_std_by_width = []
    support_max_relative_leakage_mean_by_width = []
    support_max_relative_leakage_std_by_width = []

    n_freq = len(frequencies)
    n_support = len(target_specs)

    for width in widths:
        print(f"Width = {width}")

        n_seed = len(seed_list)
        projector_fro = np.full((n_seed, n_freq), np.nan, dtype=np.float32)
        projector_op = np.full((n_seed, n_freq), np.nan, dtype=np.float32)
        principal_angles = np.full((n_seed, n_freq, 2), np.nan, dtype=np.float32)
        eig_abs_error = np.full((n_seed, n_freq), np.nan, dtype=np.float32)
        relative_center_error = np.full((n_seed, n_freq), np.nan, dtype=np.float32)
        pair_splitting = np.full((n_seed, n_freq), np.nan, dtype=np.float32)
        pair_splitting_over_lambda = np.full((n_seed, n_freq), np.nan, dtype=np.float32)
        pair_splitting_over_gap = np.full((n_seed, n_freq), np.nan, dtype=np.float32)
        lambda_hat_mean = np.full((n_seed, n_freq), np.nan, dtype=np.float32)
        selected_eigvals = np.full((n_seed, n_freq, 2), np.nan, dtype=np.float32)
        selected_eig_indices = np.full((n_seed, n_freq, 2), -1, dtype=np.int32)
        fourier_coeffs_raw = np.full((n_seed, n_freq, 2, 2), np.nan, dtype=np.float32)
        fourier_coeffs_aligned = np.full(
            (n_seed, n_freq, 2, 2), np.nan, dtype=np.float32
        )

        block_err_max = np.full((n_seed,), np.nan, dtype=np.float32)
        block_err_fro = np.full((n_seed,), np.nan, dtype=np.float32)
        block_err_op = np.full((n_seed,), np.nan, dtype=np.float32)
        instability_proxy = np.full((n_seed, n_freq), np.nan, dtype=np.float32)
        diag_strength = np.full((n_seed, n_freq), np.nan, dtype=np.float32)
        off_block_mass = np.full((n_seed, n_freq), np.nan, dtype=np.float32)
        relative_leakage = np.full((n_seed, n_freq), np.nan, dtype=np.float32)

        target_prediction_error = np.full(
            (n_seed, n_support),
            np.nan,
            dtype=np.float32,
        )
        reliability_ratio = np.full((n_seed, n_support), np.nan, dtype=np.float32)
        support_max_projector_fro = np.full(
            (n_seed, n_support),
            np.nan,
            dtype=np.float32,
        )
        support_max_relative_center_error = np.full(
            (n_seed, n_support),
            np.nan,
            dtype=np.float32,
        )
        support_max_relative_leakage = np.full(
            (n_seed, n_support),
            np.nan,
            dtype=np.float32,
        )

        B_blocks = None
        Bm_minus_Lambda_blocks = None
        if save_blocks:
            d_block = len(mode_freqs)
            B_blocks = np.full((n_seed, d_block, d_block), np.nan, dtype=np.float32)
            Bm_minus_Lambda_blocks = np.full(
                (n_seed, d_block, d_block),
                np.nan,
                dtype=np.float32,
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
            A_np = np.asarray(A, dtype=np.float64)

            B_m = (Phi.T @ A_np @ Phi) / float(n_points)
            Bm_minus_Lambda = B_m - Lambda_K
            max_norm, fro_norm, op_norm = _matrix_norms(Bm_minus_Lambda)
            block_err_max[s_idx] = max_norm
            block_err_fro[s_idx] = fro_norm
            block_err_op[s_idx] = op_norm
            instability_proxy[s_idx, :] = (
                op_norm / np.maximum(local_gap, leakage_eps)
            ).astype(np.float32)

            ds, obm, rel_leak = _fourier_action_leakage(
                B_m,
                mode_freqs,
                frequencies,
                eps=leakage_eps,
            )
            diag_strength[s_idx, :] = ds.astype(np.float32)
            off_block_mass[s_idx, :] = obm.astype(np.float32)
            relative_leakage[s_idx, :] = rel_leak.astype(np.float32)

            if save_blocks:
                B_blocks[s_idx, :, :] = B_m.astype(np.float32)
                Bm_minus_Lambda_blocks[s_idx, :, :] = Bm_minus_Lambda.astype(np.float32)

            for support_idx, support_spec in enumerate(target_specs):
                freqs = support_spec["frequencies"]
                c = np.asarray(target_coefficients[support_idx], dtype=np.float64)
                denom = float(target_denominators[support_idx])
                target_prediction_error[s_idx, support_idx] = float(
                    np.linalg.norm(Bm_minus_Lambda @ c) / max(denom, leakage_eps)
                )
                reliability_ratio[s_idx, support_idx] = float(
                    op_norm / max(float(support_lambda_min[support_idx]), leakage_eps)
                )

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
                relative_center_error[s_idx, f_idx] = float(
                    np.asarray(eig_err) / max(float(lambda_k), leakage_eps)
                )
                pair_splitting[s_idx, f_idx] = float(np.asarray(split))
                if d_k == 2:
                    split_float = float(np.asarray(split))
                    pair_splitting_over_lambda[s_idx, f_idx] = split_float / max(
                        float(lambda_k),
                        leakage_eps,
                    )
                    pair_splitting_over_gap[s_idx, f_idx] = split_float / max(
                        float(local_gap[f_idx]),
                        leakage_eps,
                    )
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

            for support_idx, support_spec in enumerate(target_specs):
                support_freqs = np.asarray(support_spec["frequencies"], dtype=np.int32)
                freq_mask = np.isin(frequencies, support_freqs)
                support_max_projector_fro[s_idx, support_idx] = float(
                    np.nanmax(projector_fro[s_idx, freq_mask])
                )
                support_max_relative_center_error[s_idx, support_idx] = float(
                    np.nanmax(relative_center_error[s_idx, freq_mask])
                )
                support_max_relative_leakage[s_idx, support_idx] = float(
                    np.nanmax(relative_leakage[s_idx, freq_mask])
                )

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
            "relative_center_error": relative_center_error,
            "pair_splitting": pair_splitting,
            "pair_splitting_over_lambda": pair_splitting_over_lambda,
            "pair_splitting_over_gap": pair_splitting_over_gap,
            "lambda_hat_mean": lambda_hat_mean,
            "selected_eigvals": selected_eigvals,
            "selected_eig_indices": selected_eig_indices,
            "fourier_coeffs_raw": fourier_coeffs_raw,
            "fourier_coeffs_aligned": fourier_coeffs_aligned,
            "block_err_max": block_err_max,
            "block_err_fro": block_err_fro,
            "block_err_op": block_err_op,
            "instability_proxy": instability_proxy,
            "diag_strength": diag_strength,
            "off_block_mass": off_block_mass,
            "relative_leakage": relative_leakage,
            "target_support_names": support_names,
            "target_support_frequencies": support_frequencies,
            "target_support_sizes": support_sizes,
            "target_support_lambda_min": support_lambda_min,
            "target_Ks": target_Ks,
            "target_amps": target_amps,
            "target_phases": target_phases,
            "target_coefficients": target_coefficients,
            "target_denominators": target_denominators,
            "target_prediction_error": target_prediction_error,
            "reliability_ratio": reliability_ratio,
            "support_max_projector_fro": support_max_projector_fro,
            "support_max_relative_center_error": support_max_relative_center_error,
            "support_max_relative_leakage": support_max_relative_leakage,
        }
        if save_aligned_eigvecs:
            width_payload["aligned_eigvecs"] = aligned_eigvecs
        if save_blocks:
            width_payload["B_blocks"] = B_blocks
            width_payload["Bm_minus_Lambda_blocks"] = Bm_minus_Lambda_blocks

        save_npz(width_path, **width_payload)
        runs_manifest[str(width)] = f"runs/width_{width}.npz"

        projector_fro_mean_by_width.append(np.mean(projector_fro, axis=0))
        projector_fro_std_by_width.append(np.std(projector_fro, axis=0))
        projector_op_mean_by_width.append(np.mean(projector_op, axis=0))
        projector_op_std_by_width.append(np.std(projector_op, axis=0))
        eig_abs_error_mean_by_width.append(np.mean(eig_abs_error, axis=0))
        eig_abs_error_std_by_width.append(np.std(eig_abs_error, axis=0))
        relative_center_error_mean_by_width.append(
            np.nanmean(relative_center_error, axis=0)
        )
        relative_center_error_std_by_width.append(
            np.nanstd(relative_center_error, axis=0)
        )
        pair_mean, pair_std = _nanmean_std_axis0(pair_splitting)
        pair_over_lambda_mean, pair_over_lambda_std = _nanmean_std_axis0(
            pair_splitting_over_lambda
        )
        pair_over_gap_mean, pair_over_gap_std = _nanmean_std_axis0(
            pair_splitting_over_gap
        )
        angle_mean, angle_std = _nanmean_std_axis0(principal_angles)
        pair_splitting_mean_by_width.append(pair_mean)
        pair_splitting_std_by_width.append(pair_std)
        pair_splitting_over_lambda_mean_by_width.append(pair_over_lambda_mean)
        pair_splitting_over_lambda_std_by_width.append(pair_over_lambda_std)
        pair_splitting_over_gap_mean_by_width.append(pair_over_gap_mean)
        pair_splitting_over_gap_std_by_width.append(pair_over_gap_std)
        principal_angles_mean_by_width.append(angle_mean)
        principal_angles_std_by_width.append(angle_std)
        block_max_mean_by_width.append(np.mean(block_err_max))
        block_max_std_by_width.append(np.std(block_err_max))
        block_fro_mean_by_width.append(np.mean(block_err_fro))
        block_fro_std_by_width.append(np.std(block_err_fro))
        block_op_mean_by_width.append(np.mean(block_err_op))
        block_op_std_by_width.append(np.std(block_err_op))
        relative_leakage_mean_by_width.append(np.nanmean(relative_leakage, axis=0))
        relative_leakage_std_by_width.append(np.nanstd(relative_leakage, axis=0))
        target_prediction_error_mean_by_width.append(
            np.nanmean(target_prediction_error, axis=0)
        )
        target_prediction_error_std_by_width.append(
            np.nanstd(target_prediction_error, axis=0)
        )
        reliability_ratio_mean_by_width.append(np.nanmean(reliability_ratio, axis=0))
        reliability_ratio_std_by_width.append(np.nanstd(reliability_ratio, axis=0))
        support_max_projector_fro_mean_by_width.append(
            np.nanmean(support_max_projector_fro, axis=0)
        )
        support_max_projector_fro_std_by_width.append(
            np.nanstd(support_max_projector_fro, axis=0)
        )
        support_max_relative_center_error_mean_by_width.append(
            np.nanmean(support_max_relative_center_error, axis=0)
        )
        support_max_relative_center_error_std_by_width.append(
            np.nanstd(support_max_relative_center_error, axis=0)
        )
        support_max_relative_leakage_mean_by_width.append(
            np.nanmean(support_max_relative_leakage, axis=0)
        )
        support_max_relative_leakage_std_by_width.append(
            np.nanstd(support_max_relative_leakage, axis=0)
        )

    projector_fro_mean = np.stack(projector_fro_mean_by_width, axis=0)
    projector_fro_std = np.stack(projector_fro_std_by_width, axis=0)
    projector_op_mean = np.stack(projector_op_mean_by_width, axis=0)
    projector_op_std = np.stack(projector_op_std_by_width, axis=0)
    eig_abs_error_mean = np.stack(eig_abs_error_mean_by_width, axis=0)
    eig_abs_error_std = np.stack(eig_abs_error_std_by_width, axis=0)
    relative_center_error_mean = np.stack(relative_center_error_mean_by_width, axis=0)
    relative_center_error_std = np.stack(relative_center_error_std_by_width, axis=0)
    pair_splitting_mean = np.stack(pair_splitting_mean_by_width, axis=0)
    pair_splitting_std = np.stack(pair_splitting_std_by_width, axis=0)
    pair_splitting_over_lambda_mean = np.stack(
        pair_splitting_over_lambda_mean_by_width,
        axis=0,
    )
    pair_splitting_over_lambda_std = np.stack(
        pair_splitting_over_lambda_std_by_width,
        axis=0,
    )
    pair_splitting_over_gap_mean = np.stack(
        pair_splitting_over_gap_mean_by_width,
        axis=0,
    )
    pair_splitting_over_gap_std = np.stack(
        pair_splitting_over_gap_std_by_width,
        axis=0,
    )
    principal_angles_mean = np.stack(principal_angles_mean_by_width, axis=0)
    principal_angles_std = np.stack(principal_angles_std_by_width, axis=0)
    block_max_mean = np.asarray(block_max_mean_by_width, dtype=np.float32)
    block_max_std = np.asarray(block_max_std_by_width, dtype=np.float32)
    block_fro_mean = np.asarray(block_fro_mean_by_width, dtype=np.float32)
    block_fro_std = np.asarray(block_fro_std_by_width, dtype=np.float32)
    block_op_mean = np.asarray(block_op_mean_by_width, dtype=np.float32)
    block_op_std = np.asarray(block_op_std_by_width, dtype=np.float32)
    relative_leakage_mean = np.stack(relative_leakage_mean_by_width, axis=0)
    relative_leakage_std = np.stack(relative_leakage_std_by_width, axis=0)
    target_prediction_error_mean = np.stack(
        target_prediction_error_mean_by_width,
        axis=0,
    )
    target_prediction_error_std = np.stack(
        target_prediction_error_std_by_width,
        axis=0,
    )
    reliability_ratio_mean = np.stack(reliability_ratio_mean_by_width, axis=0)
    reliability_ratio_std = np.stack(reliability_ratio_std_by_width, axis=0)
    support_max_projector_fro_mean = np.stack(
        support_max_projector_fro_mean_by_width,
        axis=0,
    )
    support_max_projector_fro_std = np.stack(
        support_max_projector_fro_std_by_width,
        axis=0,
    )
    support_max_relative_center_error_mean = np.stack(
        support_max_relative_center_error_mean_by_width,
        axis=0,
    )
    support_max_relative_center_error_std = np.stack(
        support_max_relative_center_error_std_by_width,
        axis=0,
    )
    support_max_relative_leakage_mean = np.stack(
        support_max_relative_leakage_mean_by_width,
        axis=0,
    )
    support_max_relative_leakage_std = np.stack(
        support_max_relative_leakage_std_by_width,
        axis=0,
    )

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
        relative_center_error_mean=relative_center_error_mean,
        relative_center_error_std=relative_center_error_std,
        pair_splitting_mean=pair_splitting_mean,
        pair_splitting_std=pair_splitting_std,
        pair_splitting_over_lambda_mean=pair_splitting_over_lambda_mean,
        pair_splitting_over_lambda_std=pair_splitting_over_lambda_std,
        pair_splitting_over_gap_mean=pair_splitting_over_gap_mean,
        pair_splitting_over_gap_std=pair_splitting_over_gap_std,
        local_gap=local_gap.astype(np.float32),
        lambda_basis=lambda_basis.astype(np.float32),
        block_max_mean=block_max_mean,
        block_max_std=block_max_std,
        block_fro_mean=block_fro_mean,
        block_fro_std=block_fro_std,
        block_op_mean=block_op_mean,
        block_op_std=block_op_std,
        relative_leakage_mean=relative_leakage_mean,
        relative_leakage_std=relative_leakage_std,
        target_support_names=support_names,
        target_support_frequencies=support_frequencies,
        target_support_sizes=support_sizes,
        target_support_lambda_min=support_lambda_min,
        target_Ks=target_Ks,
        target_amps=target_amps,
        target_phases=target_phases,
        target_coefficients=target_coefficients,
        target_denominators=target_denominators,
        target_prediction_error_mean=target_prediction_error_mean,
        target_prediction_error_std=target_prediction_error_std,
        reliability_ratio_mean=reliability_ratio_mean,
        reliability_ratio_std=reliability_ratio_std,
        support_max_projector_fro_mean=support_max_projector_fro_mean,
        support_max_projector_fro_std=support_max_projector_fro_std,
        support_max_relative_center_error_mean=support_max_relative_center_error_mean,
        support_max_relative_center_error_std=support_max_relative_center_error_std,
        support_max_relative_leakage_mean=support_max_relative_leakage_mean,
        support_max_relative_leakage_std=support_max_relative_leakage_std,
        min_width_for_small_error=min_width_for_small_error,
        failing_freqs_at_max_width=failing_freqs_at_max_width,
        projector_fro_tol=np.asarray([projector_fro_tol], dtype=np.float32),
        reliability_tau=np.asarray([reliability_tau], dtype=np.float32),
    )

    diagnostics = {
        str(int(k)): (None if int(w) < 0 else int(w))
        for k, w in zip(frequencies.tolist(), min_width_for_small_error.tolist())
    }

    support_reliability_widths = {}
    support_target_error_widths = {}
    for s_idx, name in enumerate(support_names.tolist()):
        good_rel = np.where(reliability_ratio_mean[:, s_idx] <= reliability_tau)[0]
        good_target = np.where(target_prediction_error_mean[:, s_idx] <= reliability_tau)[0]
        support_reliability_widths[str(name)] = (
            None if len(good_rel) == 0 else int(widths[int(good_rel[0])])
        )
        support_target_error_widths[str(name)] = (
            None if len(good_target) == 0 else int(widths[int(good_target[0])])
        )

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
            "save_blocks": save_blocks,
            "reliability_tau": reliability_tau,
            "target_supports": [
                {
                    "name": s["name"],
                    "frequencies": s["frequencies"],
                    "Ks": [int(k) for k in s["Ks"]],
                    "amps": [float(a) for a in s["amps"]],
                    "phases": [float(p) for p in s["phases"]],
                }
                for s in target_specs
            ],
            "target_phase_convention": "f(theta)=sum_j amp_j * cos(K_j*theta - phase_j)",
            "phase_pi_over_2_means": "+sin component",
        },
        "diagnostics": {
            "min_width_for_small_projector_error_by_frequency": diagnostics,
            "failing_frequencies_at_max_width": failing_freqs_at_max_width.tolist(),
            "min_width_for_reliability_ratio_by_support": support_reliability_widths,
            "min_width_for_target_prediction_error_by_support": support_target_error_widths,
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
