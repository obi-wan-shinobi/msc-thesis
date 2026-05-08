# ------------------------------------------------------------
# experiments/ntk_frozen_vs_evolving_mode_dynamics.py
# ------------------------------------------------------------
"""
Compare frozen-NTK vs evolving-network mode dynamics on S^1.

For each (width, seed):
    - Frozen branch: run operator GD with the NTK frozen at initialization.
    - Evolving branch: train the network and re-evaluate empirical NTK snapshots.

At each snapshot we log:
    - train residual projections onto tracked Fourier modes,
    - raw/aligned Fourier-plane coordinates for selected frequencies,
    - projector distances and principal angles for those frequency subspaces.
"""

import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
import yaml
from jax import jit, value_and_grad

from core.analysis import empirical_ntk_matrix
from core.data import FourierTarget, f_star_gamma
from core.fourier_basis_circle import build_real_fourier_basis
from core.kernel_dynamics import run_operator_gd
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


def _snapshot_steps(steps: int, eval_every: int) -> np.ndarray:
    if steps < 0:
        raise ValueError(f"train.steps must be nonnegative, got {steps}.")
    if eval_every <= 0:
        raise ValueError(f"train.eval_every must be positive, got {eval_every}.")

    snaps = list(range(0, steps + 1, eval_every))
    if snaps[-1] != steps:
        snaps.append(steps)
    return np.asarray(snaps, dtype=np.int32)


def _normalize_frequency_list(name: str, values) -> list[int]:
    out = []
    for v in values:
        k = int(v)
        if k < 0:
            raise ValueError(f"{name} must be nonnegative frequencies, got {k}.")
        if k not in out:
            out.append(k)
    if len(out) == 0:
        raise ValueError(f"{name} must not be empty.")
    return out


def _build_mode_bank(
    gamma_train: jnp.ndarray,
    tracked_freqs: list[int],
    plane_freqs: list[int],
) -> dict:
    max_k = max(max(tracked_freqs), max(plane_freqs))
    basis = build_real_fourier_basis(gamma_train, K_max=max_k)

    Phi_unit = np.asarray(basis["Phi_unit"], dtype=np.float32)
    mode_names = np.asarray(basis["mode_names"])
    mode_freqs = np.asarray(basis["mode_freqs"], dtype=np.int32)
    mode_types = np.asarray(basis["mode_types"])

    tracked_idx = []
    tracked_mode_names = []
    tracked_mode_freqs = []
    tracked_mode_types = []

    for k in tracked_freqs:
        if k == 0:
            idx = np.where((mode_freqs == 0) & (mode_types == "const"))[0]
            if len(idx) != 1:
                raise ValueError("Could not identify unique constant mode column.")
            j = int(idx[0])
            tracked_idx.append(j)
            tracked_mode_names.append(str(mode_names[j]))
            tracked_mode_freqs.append(int(mode_freqs[j]))
            tracked_mode_types.append(str(mode_types[j]))
            continue

        for t in ("cos", "sin"):
            idx = np.where((mode_freqs == k) & (mode_types == t))[0]
            if len(idx) != 1:
                raise ValueError(
                    f"Could not identify unique mode for freq={k}, type={t}."
                )
            j = int(idx[0])
            tracked_idx.append(j)
            tracked_mode_names.append(str(mode_names[j]))
            tracked_mode_freqs.append(int(mode_freqs[j]))
            tracked_mode_types.append(str(mode_types[j]))

    Phi_tracked = Phi_unit[:, tracked_idx]

    plane_frames = {}
    for k in plane_freqs:
        idx = np.where(mode_freqs == k)[0]
        expected_dim = 1 if k == 0 else 2
        if len(idx) != expected_dim:
            raise ValueError(
                f"Expected {expected_dim} basis columns for plane frequency {k}, got {len(idx)}."
            )

        U = Phi_unit[:, idx]
        U, _ = np.linalg.qr(U)
        plane_frames[int(k)] = U.astype(np.float32)

    return {
        "Phi_tracked": Phi_tracked,
        "Phi_unit": Phi_unit,
        "mode_freqs": mode_freqs,
        "mode_types": mode_types,
        "tracked_mode_names": np.asarray(tracked_mode_names),
        "tracked_mode_freqs": np.asarray(tracked_mode_freqs, dtype=np.int32),
        "tracked_mode_types": np.asarray(tracked_mode_types),
        "plane_frames": plane_frames,
    }


def _fourier_plane_metrics(
    A_train: np.ndarray,
    plane_frames: dict[int, np.ndarray],
    plane_freqs: list[int],
) -> dict:
    A_train = np.asarray(A_train, dtype=np.float64)

    evals, evecs = np.linalg.eigh(A_train)
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]

    n_plane = len(plane_freqs)

    projector_fro = np.full((n_plane,), np.nan, dtype=np.float32)
    projector_op = np.full((n_plane,), np.nan, dtype=np.float32)
    principal_angles = np.full((n_plane, 2), np.nan, dtype=np.float32)

    coeff_raw = np.full((n_plane, 2, 2), np.nan, dtype=np.float32)
    coeff_aligned = np.full((n_plane, 2, 2), np.nan, dtype=np.float32)

    eig_indices = np.full((n_plane, 2), -1, dtype=np.int32)
    eigvals = np.full((n_plane, 2), np.nan, dtype=np.float32)

    for f_idx, k in enumerate(plane_freqs):
        U = plane_frames[int(k)]
        d = U.shape[1]

        overlaps = np.sum((U.T @ evecs) ** 2, axis=0)
        top_idx = np.argsort(overlaps)[::-1][:d]
        V = evecs[:, top_idx]

        M_raw = V.T @ U
        coeff_raw[f_idx, :d, :d] = np.asarray(M_raw, dtype=np.float32)

        svals = np.linalg.svd(U.T @ V, compute_uv=False)
        svals = np.clip(svals, -1.0, 1.0)
        angles = np.arccos(svals)
        sin_t = np.sin(angles)

        projector_fro[f_idx] = np.sqrt(2.0 * np.sum(sin_t**2))
        projector_op[f_idx] = np.max(sin_t)
        principal_angles[f_idx, :d] = np.asarray(angles, dtype=np.float32)

        U_svd, _, Vt_svd = np.linalg.svd(M_raw, full_matrices=False)
        R = U_svd @ Vt_svd
        V_aligned = V @ R

        for j in range(d):
            if np.dot(V_aligned[:, j], U[:, j]) < 0:
                V_aligned[:, j] *= -1.0

        M_aligned = V_aligned.T @ U
        coeff_aligned[f_idx, :d, :d] = np.asarray(M_aligned, dtype=np.float32)

        eig_indices[f_idx, :d] = np.asarray(top_idx, dtype=np.int32)
        eigvals[f_idx, :d] = np.asarray(evals[top_idx], dtype=np.float32)

    return {
        "projector_fro": projector_fro,
        "projector_op": projector_op,
        "principal_angles": principal_angles,
        "coeff_raw": coeff_raw,
        "coeff_aligned": coeff_aligned,
        "eig_indices": eig_indices,
        "eigvals": eigvals,
    }


def _build_prefix_frames(
    Phi_unit: np.ndarray,
    mode_freqs: np.ndarray,
    prefix_freqs: list[int],
) -> tuple[dict[int, np.ndarray], np.ndarray]:
    prefix_frames = {}
    prefix_dims = []

    for k in prefix_freqs:
        idx = np.where(mode_freqs <= int(k))[0]
        expected_dim = 1 + 2 * int(k)
        if len(idx) != expected_dim:
            raise ValueError(
                f"Expected cumulative dimension {expected_dim} for prefix k={k}, got {len(idx)}."
            )

        U = Phi_unit[:, idx]
        U, _ = np.linalg.qr(U)
        prefix_frames[int(k)] = U.astype(np.float32)
        prefix_dims.append(expected_dim)

    return prefix_frames, np.asarray(prefix_dims, dtype=np.int32)


def _fourier_prefix_metrics(
    A_train: np.ndarray,
    prefix_frames: dict[int, np.ndarray],
    prefix_freqs: list[int],
    d_prefix_max: int,
) -> dict:
    A_train = np.asarray(A_train, dtype=np.float64)

    evals, evecs = np.linalg.eigh(A_train)
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]

    n_prefix = len(prefix_freqs)

    projector_fro = np.full((n_prefix,), np.nan, dtype=np.float32)
    projector_op = np.full((n_prefix,), np.nan, dtype=np.float32)
    principal_angles = np.full((n_prefix, d_prefix_max), np.nan, dtype=np.float32)
    affinity_rms = np.full((n_prefix,), np.nan, dtype=np.float32)

    eig_indices = np.full((n_prefix, d_prefix_max), -1, dtype=np.int32)
    eigvals = np.full((n_prefix, d_prefix_max), np.nan, dtype=np.float32)
    eigvals_sum = np.full((n_prefix,), np.nan, dtype=np.float32)
    eigvals_mean = np.full((n_prefix,), np.nan, dtype=np.float32)
    eigvals_min = np.full((n_prefix,), np.nan, dtype=np.float32)
    eigvals_max = np.full((n_prefix,), np.nan, dtype=np.float32)

    mu = np.full((n_prefix,), np.nan, dtype=np.float32)
    C_blocks = np.full((n_prefix, d_prefix_max, d_prefix_max), np.nan, dtype=np.float32)

    for p_idx, k in enumerate(prefix_freqs):
        U = prefix_frames[int(k)]
        d = U.shape[1]

        overlaps = np.sum((U.T @ evecs) ** 2, axis=0)
        top_idx = np.argsort(overlaps)[::-1][:d]
        V = evecs[:, top_idx]

        svals = np.linalg.svd(U.T @ V, compute_uv=False)
        svals = np.clip(svals, -1.0, 1.0)
        angles = np.arccos(svals)
        sin_t = np.sin(angles)

        projector_fro[p_idx] = np.sqrt(2.0 * np.sum(sin_t**2))
        projector_op[p_idx] = np.max(sin_t)
        principal_angles[p_idx, :d] = np.asarray(angles, dtype=np.float32)
        affinity_rms[p_idx] = np.sqrt(np.mean(svals**2))

        eig_indices[p_idx, :d] = np.asarray(top_idx, dtype=np.int32)
        eigvals_sel = np.asarray(evals[top_idx], dtype=np.float32)
        eigvals[p_idx, :d] = eigvals_sel
        eigvals_sum[p_idx] = np.sum(eigvals_sel)
        eigvals_mean[p_idx] = np.mean(eigvals_sel)
        eigvals_min[p_idx] = np.min(eigvals_sel)
        eigvals_max[p_idx] = np.max(eigvals_sel)

        Ck = U.T @ A_train @ U
        C_blocks[p_idx, :d, :d] = np.asarray(Ck, dtype=np.float32)
        mu[p_idx] = float(np.trace(Ck) / max(d, 1))

    return {
        "projector_fro": projector_fro,
        "projector_op": projector_op,
        "principal_angles": principal_angles,
        "affinity_rms": affinity_rms,
        "eig_indices": eig_indices,
        "eigvals": eigvals,
        "eigvals_sum": eigvals_sum,
        "eigvals_mean": eigvals_mean,
        "eigvals_min": eigvals_min,
        "eigvals_max": eigvals_max,
        "mu": mu,
        "C_blocks": C_blocks,
    }


def _prefix_block_drift_norms(
    C_t: np.ndarray,
    C_0: np.ndarray,
    prefix_dims: np.ndarray,
) -> dict:
    n_prefix = len(prefix_dims)
    drift_fro = np.full((n_prefix,), np.nan, dtype=np.float32)
    drift_op = np.full((n_prefix,), np.nan, dtype=np.float32)
    drift_max = np.full((n_prefix,), np.nan, dtype=np.float32)

    for p_idx in range(n_prefix):
        d = int(prefix_dims[p_idx])
        delta = np.asarray(C_t[p_idx, :d, :d] - C_0[p_idx, :d, :d], dtype=np.float64)
        drift_fro[p_idx] = np.linalg.norm(delta, ord="fro")
        drift_op[p_idx] = np.linalg.norm(delta, ord=2)
        drift_max[p_idx] = np.max(np.abs(delta))

    return {
        "drift_fro": drift_fro,
        "drift_op": drift_op,
        "drift_max": drift_max,
    }


def _train_least_squares_chunk(
    params,
    apply_fn,
    X_train: jnp.ndarray,
    y_train: jnp.ndarray,
    steps: int,
    lr: float,
):
    """
    Run `steps` full-batch SGD updates on
        L(params) = 0.5 * ||f(params, X_train) - y_train||^2.

    Returns:
        params_new, history_ls (list of scalar least-squares losses per step)
    """

    opt = optax.sgd(lr)
    opt_state = opt.init(params)

    @jit
    def step_fn(p, s):
        loss, grads = value_and_grad(
            lambda q: 0.5 * jnp.sum((apply_fn(q, X_train).squeeze() - y_train) ** 2)
        )(p)
        updates, s = opt.update(grads, s, p)
        p = optax.apply_updates(p, updates)
        return p, s, loss

    _ = step_fn(params, opt_state)

    history_ls = []
    for _ in range(steps):
        params, opt_state, loss = step_fn(params, opt_state)
        history_ls.append(float(loss))

    return params, history_ls


def run(config_path: str):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    exp_cfg = cfg["experiment"]
    data_cfg = cfg["data"]
    target_cfg = cfg["target"]
    model_cfg = cfg["model"]
    train_cfg = cfg["train"]
    sweep_cfg = cfg["sweep"]
    frozen_cfg = cfg.get("frozen", {})
    analysis_cfg = cfg.get("analysis", {})
    artifacts_cfg = cfg.get("artifacts", {})

    base_seed = int(exp_cfg.get("seed", 0))

    n_train = int(data_cfg["n_train"])
    n_eval = int(data_cfg["n_eval"])
    noise_std = float(data_cfg.get("noise_std", 0.0))
    train_grid = str(data_cfg.get("train_grid", "equally_spaced"))

    if train_grid != "equally_spaced":
        raise ValueError(
            f"data.train_grid='{train_grid}' is not supported in this experiment. "
            "Use 'equally_spaced'."
        )

    Ks = np.asarray(target_cfg["Ks"], dtype=float)
    amps = np.asarray(target_cfg["amps"], dtype=float)
    phases = np.asarray(target_cfg["phases"], dtype=float)
    if not (Ks.shape == amps.shape == phases.shape):
        raise ValueError(
            "target.Ks, target.amps, target.phases must have the same length."
        )

    spec = FourierTarget(
        Ks=jnp.asarray(Ks),
        amps=jnp.asarray(amps),
        phases=jnp.asarray(phases),
    )

    depth_hidden = int(model_cfg.get("depth_hidden", 1))
    b_std = float(model_cfg.get("b_std", 1.0))
    parameterization = str(model_cfg.get("parameterization", "ntk"))

    steps = int(train_cfg["steps"])
    lr = float(train_cfg["lr"])
    log_every = train_cfg.get("log_every", None)
    log_every = int(log_every) if log_every is not None else None
    eval_every = int(train_cfg["eval_every"])

    widths = [int(w) for w in sweep_cfg["widths"]]
    seed_list = _resolve_seed_list(sweep_cfg["seeds"], base_seed)

    frozen_eta_cfg = frozen_cfg.get("eta", None)
    frozen_eta_scale = float(frozen_cfg.get("eta_scale", 0.05))

    target_max_k = int(np.max(Ks))
    full_target_freqs = list(range(target_max_k + 1))

    use_full_target_freqs = bool(analysis_cfg.get("use_full_target_freqs", True))
    if use_full_target_freqs:
        tracked_freqs = list(full_target_freqs)
        plane_freqs = list(full_target_freqs)
    else:
        tracked_freqs = _normalize_frequency_list(
            "analysis.tracked_freqs",
            analysis_cfg.get("tracked_freqs", [0, 2, 5, 6]),
        )
        plane_freqs = _normalize_frequency_list(
            "analysis.plane_freqs",
            analysis_cfg.get("plane_freqs", [2, 5, 6]),
        )

    prefix_freqs = list(full_target_freqs)

    if any(k not in tracked_freqs for k in plane_freqs):
        raise ValueError(
            "analysis.plane_freqs must be a subset of analysis.tracked_freqs."
        )

    save_eval_predictions = bool(artifacts_cfg.get("save_eval_predictions", False))

    snapshot_steps = _snapshot_steps(steps, eval_every)
    n_snap = len(snapshot_steps)

    gamma_train, X_train = _build_circle_grid(n_train)
    gamma_eval, X_eval = _build_circle_grid(n_eval)

    y_train_clean = f_star_gamma(gamma_train, spec)
    if noise_std > 0:
        noise_key = jax.random.PRNGKey(base_seed + 777)
        y_train = y_train_clean + noise_std * jax.random.normal(
            noise_key, y_train_clean.shape
        )
    else:
        y_train = y_train_clean
    y_eval_true = f_star_gamma(gamma_eval, spec)

    mode_bank = _build_mode_bank(gamma_train, tracked_freqs, plane_freqs)
    Phi_tracked = mode_bank["Phi_tracked"]
    plane_frames = mode_bank["plane_frames"]
    prefix_frames, prefix_dims = _build_prefix_frames(
        mode_bank["Phi_unit"],
        mode_bank["mode_freqs"],
        prefix_freqs,
    )

    save_dir = make_run_dir(exp_cfg.get("save_dir", "results"), exp_cfg["name"])
    write_config_copy(save_dir, cfg)

    print("=== Frozen vs Evolving NTK Mode Dynamics ===")
    print(f"Saving results to: {save_dir}")
    print(f"widths={widths}, seeds={seed_list}")
    print(f"tracked_freqs={tracked_freqs}, plane_freqs={plane_freqs}")
    print(f"prefix_freqs={prefix_freqs}")
    print(f"steps={steps}, eval_every={eval_every}, snapshots={n_snap}\n")
    t0 = time.time()

    save_npz(
        save_dir / "probe_geometry.npz",
        gamma_train=np.asarray(gamma_train),
        X_train=np.asarray(X_train),
        gamma_eval=np.asarray(gamma_eval),
        X_eval=np.asarray(X_eval),
    )

    save_npz(
        save_dir / "target_task.npz",
        y_train=np.asarray(y_train),
        y_train_clean=np.asarray(y_train_clean),
        y_eval_true=np.asarray(y_eval_true),
        Ks=Ks,
        amps=amps,
        phases=phases,
        noise_std=np.asarray([noise_std], dtype=np.float32),
    )

    save_npz(
        save_dir / "mode_bank.npz",
        tracked_freqs=np.asarray(tracked_freqs, dtype=np.int32),
        plane_freqs=np.asarray(plane_freqs, dtype=np.int32),
        prefix_freqs=np.asarray(prefix_freqs, dtype=np.int32),
        prefix_dims=np.asarray(prefix_dims, dtype=np.int32),
        tracked_mode_names=mode_bank["tracked_mode_names"],
        tracked_mode_freqs=mode_bank["tracked_mode_freqs"],
        tracked_mode_types=mode_bank["tracked_mode_types"],
    )

    runs_manifest = {}
    runs_dir = save_dir / "runs"
    runs_dir.mkdir(exist_ok=True)

    y_train_np = np.asarray(y_train, dtype=np.float32)

    n_mode = Phi_tracked.shape[1]
    n_plane = len(plane_freqs)
    n_prefix = len(prefix_freqs)
    d_prefix_max = int(np.max(prefix_dims))
    n_seed = len(seed_list)

    for width in widths:
        print(f"Width = {width}")

        init_fn, apply_fn, _ = build_mlp_custom(
            width=width,
            depth_hidden=depth_hidden,
            b_std=b_std,
            parameterization=parameterization,
        )

        train_loss_frozen = np.full((n_seed, n_snap), np.nan, dtype=np.float32)
        train_loss_evolving = np.full((n_seed, n_snap), np.nan, dtype=np.float32)
        train_mse_frozen = np.full((n_seed, n_snap), np.nan, dtype=np.float32)
        train_mse_evolving = np.full((n_seed, n_snap), np.nan, dtype=np.float32)

        mode_proj_frozen = np.full((n_seed, n_snap, n_mode), np.nan, dtype=np.float32)
        mode_proj_evolving = np.full((n_seed, n_snap, n_mode), np.nan, dtype=np.float32)

        plane_projector_fro_frozen = np.full(
            (n_seed, n_snap, n_plane), np.nan, dtype=np.float32
        )
        plane_projector_op_frozen = np.full(
            (n_seed, n_snap, n_plane), np.nan, dtype=np.float32
        )
        plane_angles_frozen = np.full(
            (n_seed, n_snap, n_plane, 2), np.nan, dtype=np.float32
        )
        plane_coeff_raw_frozen = np.full(
            (n_seed, n_snap, n_plane, 2, 2), np.nan, dtype=np.float32
        )
        plane_coeff_aligned_frozen = np.full(
            (n_seed, n_snap, n_plane, 2, 2), np.nan, dtype=np.float32
        )
        plane_eig_indices_frozen = np.full(
            (n_seed, n_snap, n_plane, 2), -1, dtype=np.int32
        )
        plane_eigvals_frozen = np.full(
            (n_seed, n_snap, n_plane, 2), np.nan, dtype=np.float32
        )

        plane_projector_fro_evolving = np.full(
            (n_seed, n_snap, n_plane), np.nan, dtype=np.float32
        )
        plane_projector_op_evolving = np.full(
            (n_seed, n_snap, n_plane), np.nan, dtype=np.float32
        )
        plane_angles_evolving = np.full(
            (n_seed, n_snap, n_plane, 2), np.nan, dtype=np.float32
        )
        plane_coeff_raw_evolving = np.full(
            (n_seed, n_snap, n_plane, 2, 2), np.nan, dtype=np.float32
        )
        plane_coeff_aligned_evolving = np.full(
            (n_seed, n_snap, n_plane, 2, 2), np.nan, dtype=np.float32
        )
        plane_eig_indices_evolving = np.full(
            (n_seed, n_snap, n_plane, 2), -1, dtype=np.int32
        )
        plane_eigvals_evolving = np.full(
            (n_seed, n_snap, n_plane, 2), np.nan, dtype=np.float32
        )

        prefix_projector_fro_frozen = np.full(
            (n_seed, n_snap, n_prefix), np.nan, dtype=np.float32
        )
        prefix_projector_op_frozen = np.full(
            (n_seed, n_snap, n_prefix), np.nan, dtype=np.float32
        )
        prefix_angles_frozen = np.full(
            (n_seed, n_snap, n_prefix, d_prefix_max), np.nan, dtype=np.float32
        )
        prefix_affinity_frozen = np.full(
            (n_seed, n_snap, n_prefix), np.nan, dtype=np.float32
        )
        prefix_eig_indices_frozen = np.full(
            (n_seed, n_snap, n_prefix, d_prefix_max), -1, dtype=np.int32
        )
        prefix_eigvals_frozen = np.full(
            (n_seed, n_snap, n_prefix, d_prefix_max), np.nan, dtype=np.float32
        )
        prefix_eigvals_sum_frozen = np.full(
            (n_seed, n_snap, n_prefix), np.nan, dtype=np.float32
        )
        prefix_eigvals_mean_frozen = np.full(
            (n_seed, n_snap, n_prefix), np.nan, dtype=np.float32
        )
        prefix_eigvals_min_frozen = np.full(
            (n_seed, n_snap, n_prefix), np.nan, dtype=np.float32
        )
        prefix_eigvals_max_frozen = np.full(
            (n_seed, n_snap, n_prefix), np.nan, dtype=np.float32
        )
        prefix_mu_frozen = np.full(
            (n_seed, n_snap, n_prefix), np.nan, dtype=np.float32
        )

        prefix_projector_fro_evolving = np.full(
            (n_seed, n_snap, n_prefix), np.nan, dtype=np.float32
        )
        prefix_projector_op_evolving = np.full(
            (n_seed, n_snap, n_prefix), np.nan, dtype=np.float32
        )
        prefix_angles_evolving = np.full(
            (n_seed, n_snap, n_prefix, d_prefix_max), np.nan, dtype=np.float32
        )
        prefix_affinity_evolving = np.full(
            (n_seed, n_snap, n_prefix), np.nan, dtype=np.float32
        )
        prefix_eig_indices_evolving = np.full(
            (n_seed, n_snap, n_prefix, d_prefix_max), -1, dtype=np.int32
        )
        prefix_eigvals_evolving = np.full(
            (n_seed, n_snap, n_prefix, d_prefix_max), np.nan, dtype=np.float32
        )
        prefix_eigvals_sum_evolving = np.full(
            (n_seed, n_snap, n_prefix), np.nan, dtype=np.float32
        )
        prefix_eigvals_mean_evolving = np.full(
            (n_seed, n_snap, n_prefix), np.nan, dtype=np.float32
        )
        prefix_eigvals_min_evolving = np.full(
            (n_seed, n_snap, n_prefix), np.nan, dtype=np.float32
        )
        prefix_eigvals_max_evolving = np.full(
            (n_seed, n_snap, n_prefix), np.nan, dtype=np.float32
        )
        prefix_mu_evolving = np.full(
            (n_seed, n_snap, n_prefix), np.nan, dtype=np.float32
        )

        prefix_C_0 = np.full(
            (n_seed, n_prefix, d_prefix_max, d_prefix_max), np.nan, dtype=np.float32
        )
        prefix_C_t_frozen = np.full(
            (n_seed, n_snap, n_prefix, d_prefix_max, d_prefix_max),
            np.nan,
            dtype=np.float32,
        )
        prefix_C_t_evolving = np.full(
            (n_seed, n_snap, n_prefix, d_prefix_max, d_prefix_max),
            np.nan,
            dtype=np.float32,
        )

        prefix_C_drift_fro_frozen = np.full(
            (n_seed, n_snap, n_prefix), np.nan, dtype=np.float32
        )
        prefix_C_drift_op_frozen = np.full(
            (n_seed, n_snap, n_prefix), np.nan, dtype=np.float32
        )
        prefix_C_drift_max_frozen = np.full(
            (n_seed, n_snap, n_prefix), np.nan, dtype=np.float32
        )

        prefix_C_drift_fro_evolving = np.full(
            (n_seed, n_snap, n_prefix), np.nan, dtype=np.float32
        )
        prefix_C_drift_op_evolving = np.full(
            (n_seed, n_snap, n_prefix), np.nan, dtype=np.float32
        )
        prefix_C_drift_max_evolving = np.full(
            (n_seed, n_snap, n_prefix), np.nan, dtype=np.float32
        )

        eta_frozen_by_seed = np.full((n_seed,), np.nan, dtype=np.float32)
        lambda_max_init_by_seed = np.full((n_seed,), np.nan, dtype=np.float32)

        pred_eval_frozen = None
        pred_eval_evolving = None
        if save_eval_predictions:
            pred_eval_frozen = np.full(
                (n_seed, n_snap, n_eval),
                np.nan,
                dtype=np.float32,
            )
            pred_eval_evolving = np.full(
                (n_seed, n_snap, n_eval),
                np.nan,
                dtype=np.float32,
            )

        for s_idx, seed in enumerate(seed_list):
            print(f"  Seed {seed}")

            _, params0 = init_fn(jax.random.PRNGKey(seed), X_train.shape)

            y_pred_train_0 = np.asarray(
                apply_fn(params0, X_train).squeeze(), dtype=np.float32
            )
            y_pred_eval_0 = np.asarray(
                apply_fn(params0, X_eval).squeeze(), dtype=np.float32
            )

            K0_train = np.asarray(
                empirical_ntk_matrix(apply_fn, params0, X_train).squeeze(),
                dtype=np.float32,
            ) / float(n_train)
            K0_eval_train = np.asarray(
                empirical_ntk_matrix(apply_fn, params0, X_eval, X_train).squeeze(),
                dtype=np.float32,
            ) / float(n_train)

            lambda_max_init = float(np.max(np.linalg.eigvalsh(K0_train)))
            lambda_max_init_by_seed[s_idx] = lambda_max_init

            if frozen_eta_cfg is not None:
                eta_frozen = float(frozen_eta_cfg)
            else:
                eta_frozen = float(frozen_eta_scale / (lambda_max_init + 1e-12))
            eta_frozen_by_seed[s_idx] = eta_frozen

            out_frozen = run_operator_gd(
                A_train=jnp.asarray(K0_train),
                y_train=y_train,
                eta=eta_frozen,
                steps=steps,
                y_pred_train_0=jnp.asarray(y_pred_train_0),
                A_eval_train=jnp.asarray(K0_eval_train),
                y_pred_eval_0=jnp.asarray(y_pred_eval_0),
                save_every=eval_every,
            )

            snapshot_steps_frozen = np.asarray(
                out_frozen["snapshot_steps"], dtype=np.int32
            )
            if not np.array_equal(snapshot_steps_frozen, snapshot_steps):
                raise ValueError(
                    "Frozen branch snapshot steps do not match expected snapshot grid: "
                    f"expected {snapshot_steps.tolist()}, got {snapshot_steps_frozen.tolist()}."
                )

            r_train_frozen = np.asarray(out_frozen["r_train"], dtype=np.float32)[
                snapshot_steps
            ]
            loss_frozen = np.asarray(out_frozen["loss"], dtype=np.float32)[
                snapshot_steps
            ]
            train_loss_frozen[s_idx, :] = loss_frozen
            train_mse_frozen[s_idx, :] = (2.0 * loss_frozen) / float(n_train)
            mode_proj_frozen[s_idx, :, :] = r_train_frozen @ Phi_tracked

            if save_eval_predictions:
                pred_eval_frozen[s_idx, :, :] = np.asarray(
                    out_frozen["y_pred_eval_snapshots"],
                    dtype=np.float32,
                )

            plane_frozen = _fourier_plane_metrics(K0_train, plane_frames, plane_freqs)
            prefix_frozen = _fourier_prefix_metrics(
                K0_train,
                prefix_frames,
                prefix_freqs,
                d_prefix_max,
            )
            prefix_C0_seed = np.asarray(prefix_frozen["C_blocks"], dtype=np.float32)
            prefix_C_0[s_idx, :, :, :] = prefix_C0_seed

            for t_idx in range(n_snap):
                plane_projector_fro_frozen[s_idx, t_idx, :] = plane_frozen[
                    "projector_fro"
                ]
                plane_projector_op_frozen[s_idx, t_idx, :] = plane_frozen[
                    "projector_op"
                ]
                plane_angles_frozen[s_idx, t_idx, :, :] = plane_frozen[
                    "principal_angles"
                ]
                plane_coeff_raw_frozen[s_idx, t_idx, :, :, :] = plane_frozen[
                    "coeff_raw"
                ]
                plane_coeff_aligned_frozen[s_idx, t_idx, :, :, :] = plane_frozen[
                    "coeff_aligned"
                ]
                plane_eig_indices_frozen[s_idx, t_idx, :, :] = plane_frozen[
                    "eig_indices"
                ]
                plane_eigvals_frozen[s_idx, t_idx, :, :] = plane_frozen["eigvals"]

                prefix_projector_fro_frozen[s_idx, t_idx, :] = prefix_frozen[
                    "projector_fro"
                ]
                prefix_projector_op_frozen[s_idx, t_idx, :] = prefix_frozen[
                    "projector_op"
                ]
                prefix_angles_frozen[s_idx, t_idx, :, :] = prefix_frozen[
                    "principal_angles"
                ]
                prefix_affinity_frozen[s_idx, t_idx, :] = prefix_frozen[
                    "affinity_rms"
                ]
                prefix_eig_indices_frozen[s_idx, t_idx, :, :] = prefix_frozen[
                    "eig_indices"
                ]
                prefix_eigvals_frozen[s_idx, t_idx, :, :] = prefix_frozen["eigvals"]
                prefix_eigvals_sum_frozen[s_idx, t_idx, :] = prefix_frozen[
                    "eigvals_sum"
                ]
                prefix_eigvals_mean_frozen[s_idx, t_idx, :] = prefix_frozen[
                    "eigvals_mean"
                ]
                prefix_eigvals_min_frozen[s_idx, t_idx, :] = prefix_frozen[
                    "eigvals_min"
                ]
                prefix_eigvals_max_frozen[s_idx, t_idx, :] = prefix_frozen[
                    "eigvals_max"
                ]
                prefix_mu_frozen[s_idx, t_idx, :] = prefix_frozen["mu"]
                prefix_C_t_frozen[s_idx, t_idx, :, :, :] = prefix_C0_seed

                drift_frozen = _prefix_block_drift_norms(
                    prefix_C0_seed,
                    prefix_C0_seed,
                    prefix_dims,
                )
                prefix_C_drift_fro_frozen[s_idx, t_idx, :] = drift_frozen["drift_fro"]
                prefix_C_drift_op_frozen[s_idx, t_idx, :] = drift_frozen["drift_op"]
                prefix_C_drift_max_frozen[s_idx, t_idx, :] = drift_frozen["drift_max"]

            params = params0
            current_step = 0
            snap_idx = 0

            while True:
                y_pred_train = np.asarray(
                    apply_fn(params, X_train).squeeze(), dtype=np.float32
                )
                r_train = y_pred_train - y_train_np

                sq_norm = np.sum(r_train**2)
                train_loss_evolving[s_idx, snap_idx] = 0.5 * sq_norm
                train_mse_evolving[s_idx, snap_idx] = sq_norm / float(n_train)
                mode_proj_evolving[s_idx, snap_idx, :] = r_train @ Phi_tracked

                if save_eval_predictions:
                    pred_eval_evolving[s_idx, snap_idx, :] = np.asarray(
                        apply_fn(params, X_eval).squeeze(),
                        dtype=np.float32,
                    )

                K_train_t = np.asarray(
                    empirical_ntk_matrix(apply_fn, params, X_train).squeeze(),
                    dtype=np.float32,
                ) / float(n_train)
                plane_evolving = _fourier_plane_metrics(
                    K_train_t, plane_frames, plane_freqs
                )
                prefix_evolving = _fourier_prefix_metrics(
                    K_train_t,
                    prefix_frames,
                    prefix_freqs,
                    d_prefix_max,
                )

                plane_projector_fro_evolving[s_idx, snap_idx, :] = plane_evolving[
                    "projector_fro"
                ]
                plane_projector_op_evolving[s_idx, snap_idx, :] = plane_evolving[
                    "projector_op"
                ]
                plane_angles_evolving[s_idx, snap_idx, :, :] = plane_evolving[
                    "principal_angles"
                ]
                plane_coeff_raw_evolving[s_idx, snap_idx, :, :, :] = plane_evolving[
                    "coeff_raw"
                ]
                plane_coeff_aligned_evolving[s_idx, snap_idx, :, :, :] = plane_evolving[
                    "coeff_aligned"
                ]
                plane_eig_indices_evolving[s_idx, snap_idx, :, :] = plane_evolving[
                    "eig_indices"
                ]
                plane_eigvals_evolving[s_idx, snap_idx, :, :] = plane_evolving[
                    "eigvals"
                ]

                prefix_projector_fro_evolving[s_idx, snap_idx, :] = prefix_evolving[
                    "projector_fro"
                ]
                prefix_projector_op_evolving[s_idx, snap_idx, :] = prefix_evolving[
                    "projector_op"
                ]
                prefix_angles_evolving[s_idx, snap_idx, :, :] = prefix_evolving[
                    "principal_angles"
                ]
                prefix_affinity_evolving[s_idx, snap_idx, :] = prefix_evolving[
                    "affinity_rms"
                ]
                prefix_eig_indices_evolving[s_idx, snap_idx, :, :] = prefix_evolving[
                    "eig_indices"
                ]
                prefix_eigvals_evolving[s_idx, snap_idx, :, :] = prefix_evolving[
                    "eigvals"
                ]
                prefix_eigvals_sum_evolving[s_idx, snap_idx, :] = prefix_evolving[
                    "eigvals_sum"
                ]
                prefix_eigvals_mean_evolving[s_idx, snap_idx, :] = prefix_evolving[
                    "eigvals_mean"
                ]
                prefix_eigvals_min_evolving[s_idx, snap_idx, :] = prefix_evolving[
                    "eigvals_min"
                ]
                prefix_eigvals_max_evolving[s_idx, snap_idx, :] = prefix_evolving[
                    "eigvals_max"
                ]
                prefix_mu_evolving[s_idx, snap_idx, :] = prefix_evolving["mu"]

                prefix_C_t_evolving[s_idx, snap_idx, :, :, :] = prefix_evolving[
                    "C_blocks"
                ]

                drift_evolving = _prefix_block_drift_norms(
                    prefix_evolving["C_blocks"],
                    prefix_C0_seed,
                    prefix_dims,
                )
                prefix_C_drift_fro_evolving[s_idx, snap_idx, :] = drift_evolving[
                    "drift_fro"
                ]
                prefix_C_drift_op_evolving[s_idx, snap_idx, :] = drift_evolving[
                    "drift_op"
                ]
                prefix_C_drift_max_evolving[s_idx, snap_idx, :] = drift_evolving[
                    "drift_max"
                ]

                if current_step >= steps:
                    break

                chunk = min(eval_every, steps - current_step)
                params, history_ls = _train_least_squares_chunk(
                    params,
                    apply_fn,
                    X_train,
                    y_train,
                    steps=chunk,
                    lr=lr,
                )

                current_step += chunk
                snap_idx += 1

                if log_every and (
                    (current_step % log_every == 0) or (current_step == steps)
                ):
                    last_loss = history_ls[-1]
                    print(
                        f"    [evolving] step {current_step:7d} | "
                        f"train_ls={last_loss:.6e}"
                    )

            if snap_idx != n_snap - 1:
                raise RuntimeError(
                    f"Unexpected evolving snapshot count for seed {seed}: "
                    f"expected {n_snap}, got {snap_idx + 1}."
                )

        payload = {
            "width": np.asarray([width], dtype=np.int32),
            "seeds": np.asarray(seed_list, dtype=np.int32),
            "snapshot_steps": snapshot_steps,
            "tracked_freqs": np.asarray(tracked_freqs, dtype=np.int32),
            "plane_freqs": np.asarray(plane_freqs, dtype=np.int32),
            "prefix_freqs": np.asarray(prefix_freqs, dtype=np.int32),
            "prefix_dims": np.asarray(prefix_dims, dtype=np.int32),
            "tracked_mode_names": mode_bank["tracked_mode_names"],
            "tracked_mode_freqs": mode_bank["tracked_mode_freqs"],
            "tracked_mode_types": mode_bank["tracked_mode_types"],
            "eta_frozen": eta_frozen_by_seed,
            "lambda_max_init": lambda_max_init_by_seed,
            "train_loss_frozen": train_loss_frozen,
            "train_loss_evolving": train_loss_evolving,
            "train_mse_frozen": train_mse_frozen,
            "train_mse_evolving": train_mse_evolving,
            "mode_proj_frozen": mode_proj_frozen,
            "mode_proj_evolving": mode_proj_evolving,
            "plane_projector_fro_frozen": plane_projector_fro_frozen,
            "plane_projector_op_frozen": plane_projector_op_frozen,
            "plane_angles_frozen": plane_angles_frozen,
            "plane_coeff_raw_frozen": plane_coeff_raw_frozen,
            "plane_coeff_aligned_frozen": plane_coeff_aligned_frozen,
            "plane_eig_indices_frozen": plane_eig_indices_frozen,
            "plane_eigvals_frozen": plane_eigvals_frozen,
            "plane_projector_fro_evolving": plane_projector_fro_evolving,
            "plane_projector_op_evolving": plane_projector_op_evolving,
            "plane_angles_evolving": plane_angles_evolving,
            "plane_coeff_raw_evolving": plane_coeff_raw_evolving,
            "plane_coeff_aligned_evolving": plane_coeff_aligned_evolving,
            "plane_eig_indices_evolving": plane_eig_indices_evolving,
            "plane_eigvals_evolving": plane_eigvals_evolving,
            "prefix_projector_fro_frozen": prefix_projector_fro_frozen,
            "prefix_projector_op_frozen": prefix_projector_op_frozen,
            "prefix_angles_frozen": prefix_angles_frozen,
            "prefix_affinity_frozen": prefix_affinity_frozen,
            "prefix_eig_indices_frozen": prefix_eig_indices_frozen,
            "prefix_eigvals_frozen": prefix_eigvals_frozen,
            "prefix_eigvals_sum_frozen": prefix_eigvals_sum_frozen,
            "prefix_eigvals_mean_frozen": prefix_eigvals_mean_frozen,
            "prefix_eigvals_min_frozen": prefix_eigvals_min_frozen,
            "prefix_eigvals_max_frozen": prefix_eigvals_max_frozen,
            "prefix_mu_frozen": prefix_mu_frozen,
            "prefix_projector_fro_evolving": prefix_projector_fro_evolving,
            "prefix_projector_op_evolving": prefix_projector_op_evolving,
            "prefix_angles_evolving": prefix_angles_evolving,
            "prefix_affinity_evolving": prefix_affinity_evolving,
            "prefix_eig_indices_evolving": prefix_eig_indices_evolving,
            "prefix_eigvals_evolving": prefix_eigvals_evolving,
            "prefix_eigvals_sum_evolving": prefix_eigvals_sum_evolving,
            "prefix_eigvals_mean_evolving": prefix_eigvals_mean_evolving,
            "prefix_eigvals_min_evolving": prefix_eigvals_min_evolving,
            "prefix_eigvals_max_evolving": prefix_eigvals_max_evolving,
            "prefix_mu_evolving": prefix_mu_evolving,
            "prefix_C_0": prefix_C_0,
            "prefix_C_t_frozen": prefix_C_t_frozen,
            "prefix_C_t_evolving": prefix_C_t_evolving,
            "prefix_C_drift_fro_frozen": prefix_C_drift_fro_frozen,
            "prefix_C_drift_op_frozen": prefix_C_drift_op_frozen,
            "prefix_C_drift_max_frozen": prefix_C_drift_max_frozen,
            "prefix_C_drift_fro_evolving": prefix_C_drift_fro_evolving,
            "prefix_C_drift_op_evolving": prefix_C_drift_op_evolving,
            "prefix_C_drift_max_evolving": prefix_C_drift_max_evolving,
        }

        if save_eval_predictions:
            payload["pred_eval_frozen"] = pred_eval_frozen
            payload["pred_eval_evolving"] = pred_eval_evolving

        width_path = runs_dir / f"width_{width}.npz"
        save_npz(width_path, **payload)
        runs_manifest[str(width)] = f"runs/width_{width}.npz"

    summary = {
        "probe_geometry": "probe_geometry.npz",
        "target_task": "target_task.npz",
        "mode_bank": "mode_bank.npz",
        "runs": runs_manifest,
        "meta": {
            "n_train": n_train,
            "n_eval": n_eval,
            "train_grid": train_grid,
            "noise_std": noise_std,
            "widths": widths,
            "seeds": seed_list,
            "steps": steps,
            "lr": lr,
            "eval_every": eval_every,
            "snapshot_steps": snapshot_steps.tolist(),
            "depth_hidden": depth_hidden,
            "b_std": b_std,
            "parameterization": parameterization,
            "frozen_eta": frozen_eta_cfg,
            "frozen_eta_scale": frozen_eta_scale,
            "target_max_k": target_max_k,
            "use_full_target_freqs": use_full_target_freqs,
            "full_target_freqs": full_target_freqs,
            "tracked_freqs": tracked_freqs,
            "plane_freqs": plane_freqs,
            "prefix_freqs": prefix_freqs,
            "prefix_dims": prefix_dims.tolist(),
            "target_Ks": target_cfg["Ks"],
            "target_amps": target_cfg["amps"],
            "target_phases": target_cfg["phases"],
            "loss_objective": "0.5 * ||f(X_train)-y_train||^2",
            "mse_relation": "train_mse = (2 / n_train) * train_loss",
            "save_eval_predictions": save_eval_predictions,
        },
        "runtime_sec": round(time.time() - t0, 2),
    }
    save_json(save_dir / "manifest.json", summary)

    print(f"\nDone. Frozen-vs-evolving mode dynamics saved to {save_dir}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(
            "Usage: python3 -m experiments.ntk_frozen_vs_evolving_mode_dynamics "
            "configs/ntk_frozen_vs_evolving_mode_dynamics_w256.yaml"
        )
    else:
        run(sys.argv[1])
