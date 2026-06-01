# experiments/kernel_circle_preconditioning.py
import sys
import time
from pathlib import Path

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import yaml

from core.data import FourierTarget, f_star_gamma
from core.fourier_basis_circle import build_real_fourier_basis
from core.kernel_analysis import (
    compressed_operator_from_lemma_objects,
    compute_lemma_objects,
    continuum_fourier_eigenvalues_bias,
    finite_n_diagonal_benchmark,
    matrix_error_metrics,
)
from core.kernel_circle import (
    TWO_PI,
    kernel_cross_matrix_from_gamma,
    kernel_matrix_from_gamma,
)
from core.kernel_dynamics import run_operator_gd
from core.preconditioned_operator import build_theory_preconditioned_operators
from utils.artifacts import ensure_dir, make_run_dir, save_json, save_npz, write_config_copy


def _sample_uniform_gamma(key, n: int) -> jnp.ndarray:
    return jr.uniform(key, shape=(n,), minval=0.0, maxval=TWO_PI)


def _circle_points_from_gamma(gamma: jnp.ndarray) -> jnp.ndarray:
    return jnp.stack([jnp.cos(gamma), jnp.sin(gamma)], axis=1)


def _resolve_seed_list(seed_cfg, base_seed: int) -> list[int]:
    if isinstance(seed_cfg, int):
        return [base_seed + s for s in range(seed_cfg)]
    return [int(s) for s in seed_cfg]


def _kernel_diag_value(kernel_name: str) -> float:
    g0 = kernel_matrix_from_gamma(jnp.array([0.0]), kernel=kernel_name)[0, 0]
    return float(np.asarray(g0))


def _normalized_abs_curves(mode_proj: jnp.ndarray, eps: float = 1e-12) -> jnp.ndarray:
    """
    Normalize each mode trajectory by its initial magnitude:
        curve[t, p] = |proj[t, p]| / max(|proj[0, p]|, eps)
    """
    denom = jnp.maximum(jnp.abs(mode_proj[0:1, :]), eps)
    return jnp.abs(mode_proj) / denom


def _build_empirical_preconditioned_operator(
    A: jnp.ndarray,
    Phi: jnp.ndarray,
    G_inv: jnp.ndarray,
    C: jnp.ndarray,
    tau: float = 0.0,
    reg: float = 1e-8,
) -> dict:
    """
    Empirical low-block preconditioner.

    We define
        M_emp = (C + (tau + reg) I)^{-1}
        P_emp = Phi M_emp G^{-1} Phi^T / n
        B_emp = P_emp A

    where
        C = G^{-1} H
        G = (1/n) Phi^T Phi
        H = (1/n) Phi^T A Phi
    """
    A = jnp.asarray(A)
    Phi = jnp.asarray(Phi)
    G_inv = jnp.asarray(G_inv)
    C = jnp.asarray(C)

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"A must be square, got shape {A.shape}.")
    if Phi.ndim != 2 or Phi.shape[0] != A.shape[0]:
        raise ValueError(f"Phi must have shape ({A.shape[0]}, d), got {Phi.shape}.")
    if G_inv.shape != C.shape:
        raise ValueError(
            f"G_inv and C must have same shape, got {G_inv.shape} and {C.shape}."
        )

    n, d = Phi.shape
    I_d = jnp.eye(d, dtype=Phi.dtype)

    M_emp = jnp.linalg.solve(C + (tau + reg) * I_d, I_d)
    P_emp = (Phi @ M_emp @ G_inv @ Phi.T) / n
    B_emp = P_emp @ A

    return {
        "M_emp": M_emp,
        "P_emp": P_emp,
        "B_emp": B_emp,
    }


def _final_probe_prediction_from_residuals(
    B_probe: jnp.ndarray,
    r_train: jnp.ndarray,
    eta: float,
) -> jnp.ndarray:
    """
    Reconstruct the final probe prediction from the train residual trajectory.

    If the train dynamics are
        y_{t+1} = y_t - eta * B_train r_t,
    then the probe prediction evolves as
        y_probe_{t+1} = y_probe_t - eta * B_probe r_t.

    We assume r_train[t] stores r_t for t = 0, ..., T, so the updates use r_train[:-1].
    """
    B_probe = jnp.asarray(B_probe)
    r_train = jnp.asarray(r_train)

    if r_train.ndim != 2:
        raise ValueError(f"r_train must be 2D, got shape {r_train.shape}.")
    if B_probe.ndim != 2 or B_probe.shape[1] != r_train.shape[1]:
        raise ValueError(
            f"B_probe must have shape (n_probe, n_train) with n_train={r_train.shape[1]}, "
            f"got {B_probe.shape}."
        )

    # shape: [T, n_probe]
    probe_updates = r_train[:-1] @ B_probe.T
    y_pred_probe_final = -eta * jnp.sum(probe_updates, axis=0)
    return y_pred_probe_final


def _latest_or_none(base_dir: Path, exp_name: str) -> Path | None:
    exp_dir = base_dir / exp_name
    latest = exp_dir / "latest"
    if latest.exists():
        return latest.resolve()
    if not exp_dir.exists():
        return None
    runs = [p for p in exp_dir.iterdir() if p.is_dir()]
    return sorted(runs)[-1] if runs else None


def _set_latest_symlink(base_dir: Path, exp_name: str, run_dir: Path) -> None:
    exp_dir = ensure_dir(base_dir / exp_name)
    latest = exp_dir / "latest"
    try:
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        latest.symlink_to(run_dir.name)
    except Exception:
        pass


def run(config_path: str):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    exp_cfg = cfg["experiment"]
    data_cfg = cfg["data"]
    basis_cfg = cfg["basis"]
    target_cfg = cfg["target"]
    kernel_cfg = cfg["kernel"]
    train_cfg = cfg["train"]
    sweep_cfg = cfg["sweep"]
    prec_cfg = cfg.get("preconditioner", {})
    analysis_cfg = cfg.get("analysis", {})
    artifacts_cfg = cfg.get("artifacts", {})

    base_seed = int(exp_cfg.get("seed", 0))
    n_trains = [int(n) for n in data_cfg["n_trains"]]
    noise_std = float(data_cfg.get("noise_std", 0.0))
    n_probe = int(data_cfg.get("n_probe", 2048))

    K_max = int(basis_cfg["K_max"])

    Ks = jnp.asarray(target_cfg["Ks"], dtype=float)
    amps = jnp.asarray(target_cfg["amps"], dtype=float)
    phases = jnp.asarray(target_cfg["phases"], dtype=float)

    kernel_name = kernel_cfg["name"]
    if kernel_name != "bias":
        raise NotImplementedError(
            "This experiment currently supports only kernel='bias', because "
            "continuum_fourier_eigenvalues_bias is implemented only for the bias kernel."
        )

    steps = int(train_cfg["steps"])
    eta = float(train_cfg["eta"])

    tau = float(prec_cfg.get("tau", 0.0))
    reg = float(prec_cfg.get("reg", 1e-8))

    save_profile = str(artifacts_cfg.get("save_profile", "compact")).lower()
    if save_profile not in {"compact", "full"}:
        raise ValueError(
            f"artifacts.save_profile must be 'compact' or 'full', got {save_profile!r}."
        )

    seed_list = _resolve_seed_list(sweep_cfg["seeds"], base_seed)
    expected_run_keys = {
        f"size_{n_train}_seed_{seed}" for n_train in n_trains for seed in seed_list
    }
    mode_compare = list(analysis_cfg.get("mode_compare", []))
    mode_projection_k_max = int(analysis_cfg.get("mode_projection_k_max", K_max))

    base_save_dir = Path(exp_cfg.get("save_dir", "results"))
    exp_name = exp_cfg["name"]
    resume = bool(exp_cfg.get("resume", False))
    resume_run_id = exp_cfg.get("resume_run_id")

    if resume:
        if resume_run_id:
            save_dir = (base_save_dir / exp_name / str(resume_run_id)).resolve()
            if not save_dir.exists():
                raise FileNotFoundError(
                    f"resume_run_id={resume_run_id!r} does not exist under "
                    f"{base_save_dir / exp_name}."
                )
        else:
            save_dir = _latest_or_none(base_save_dir, exp_name)

        if save_dir is None:
            save_dir = make_run_dir(base_save_dir, exp_name)
            print("No previous run found; starting new run directory.")
        else:
            _set_latest_symlink(base_save_dir, exp_name, save_dir)
            print(f"Resuming existing run directory: {save_dir}")
    else:
        save_dir = make_run_dir(base_save_dir, exp_name)

    if resume and (save_dir / "config.yaml").exists():
        try:
            prev_cfg = yaml.safe_load((save_dir / "config.yaml").read_text())
            if prev_cfg != cfg:
                print("WARNING: resume config differs from existing config.yaml in run dir.")
        except Exception:
            pass

    write_config_copy(save_dir, cfg)

    print("=== Kernel circle preconditioning run ===")
    print(f"Saving results to: {save_dir}\n")
    print(f"Artifact profile: {save_profile}\n")
    print(f"Expected run files for this config: {len(expected_run_keys)}\n")
    t0 = time.time()

    ft = FourierTarget(Ks=Ks, amps=amps, phases=phases)

    # ------------------------------------------------------------
    # Shared dense probe grid for function-space visualization
    # ------------------------------------------------------------
    gamma_probe = jnp.linspace(0.0, TWO_PI, num=n_probe, endpoint=False)
    X_probe = _circle_points_from_gamma(gamma_probe)
    y_target_probe = f_star_gamma(gamma_probe, ft)

    probe_basis = build_real_fourier_basis(gamma_probe, K_max=K_max)
    Phi_probe = probe_basis["Phi"]

    probe_path = save_dir / "probe_geometry.npz"
    runs_dir = save_dir / "runs"
    ensure_dir(runs_dir)

    if not (resume and probe_path.exists()):
        save_npz(
            probe_path,
            gamma_probe=np.asarray(gamma_probe),
            X_probe=np.asarray(X_probe),
            y_target_probe=np.asarray(y_target_probe),
        )

    all_existing_keys = {p.stem for p in sorted(runs_dir.glob("size_*_seed_*.npz"))}
    extra_existing = sorted(all_existing_keys - expected_run_keys)
    if extra_existing:
        print(
            "Found existing run files from other n/seed settings; "
            "they will be ignored for this manifest."
        )
        print(f"Ignored run file count: {len(extra_existing)}")

    runs_manifest = {}
    for run_key in sorted(expected_run_keys):
        run_path = runs_dir / f"{run_key}.npz"
        if run_path.exists():
            runs_manifest[run_key] = f"runs/{run_path.name}"

    if runs_manifest:
        print(f"Found {len(runs_manifest)} existing run files; completed seeds will be skipped.")

    for n_train in n_trains:
        print(f"=== n_train = {n_train} ===")

        for seed in seed_list:
            run_key = f"size_{n_train}_seed_{seed}"
            run_path = runs_dir / f"{run_key}.npz"

            if run_key in runs_manifest and run_path.exists():
                print(f"---- seed = {seed} ---- [skip existing]")
                continue

            print(f"---- seed = {seed} ----")

            # ------------------------------------------------------------
            # 1. Sample train points and build target
            # ------------------------------------------------------------
            key = jr.PRNGKey(seed)
            gamma_train = _sample_uniform_gamma(key, n_train)
            X_train = _circle_points_from_gamma(gamma_train)

            y_train = f_star_gamma(gamma_train, ft)
            if noise_std > 0:
                noise = noise_std * jr.normal(jr.fold_in(key, 1), shape=y_train.shape)
                y_train = y_train + noise

            # ------------------------------------------------------------
            # 2. Build sampled Fourier basis up to K_max
            # ------------------------------------------------------------
            basis = build_real_fourier_basis(gamma_train, K_max=K_max)
            Phi = basis["Phi"]
            Phi_unit = basis["Phi_unit"]
            mode_names = basis["mode_names"]
            mode_freqs = basis["mode_freqs"]
            mode_types = basis["mode_types"]

            if mode_projection_k_max == K_max:
                basis_ext = basis
            else:
                basis_ext = build_real_fourier_basis(
                    gamma_train,
                    K_max=mode_projection_k_max,
                )

            Phi_unit_ext = basis_ext["Phi_unit"]
            mode_names_ext = basis_ext["mode_names"]
            mode_freqs_ext = basis_ext["mode_freqs"]
            mode_types_ext = basis_ext["mode_types"]

            # ------------------------------------------------------------
            # 3. Build empirical operator A = K / n
            # ------------------------------------------------------------
            K = kernel_matrix_from_gamma(gamma_train, kernel=kernel_name)
            A = K / n_train

            # Probe-to-train normalized cross operator for baseline predictions
            K_probe = kernel_cross_matrix_from_gamma(
                gamma_probe,
                gamma_train,
                kernel=kernel_name,
            )
            A_probe = K_probe / n_train

            # ------------------------------------------------------------
            # 4. Build theory diagonal Lambda^(n)
            # ------------------------------------------------------------
            lambda_by_k = continuum_fourier_eigenvalues_bias(jnp.arange(K_max + 1))
            g0 = _kernel_diag_value(kernel_name)
            lambda_n_diag = finite_n_diagonal_benchmark(
                mode_freqs=mode_freqs,
                lambda_by_k=lambda_by_k,
                g0=g0,
                n=n_train,
            )
            Lambda_n = jnp.diag(lambda_n_diag)

            # ------------------------------------------------------------
            # 5. Build lemma objects and empirical compressed operator C
            # ------------------------------------------------------------
            lemma = compute_lemma_objects(
                Phi=Phi,
                A=A,
                lambda_n_diag=lambda_n_diag,
            )
            G = lemma["G"]
            H = lemma["H"]

            C = compressed_operator_from_lemma_objects(G, H, reg=reg)

            # ------------------------------------------------------------
            # 6. Build theoretical preconditioned operator B_th
            # ------------------------------------------------------------
            th = build_theory_preconditioned_operators(
                A_train=A,
                Phi_train=Phi,
                lambda_n_diag=lambda_n_diag,
                tau=tau,
                reg=reg,
            )
            M_th = th["M"]
            P_th = th["P"]
            G_inv = th["G_inv"]
            B_th = th["B_train"]

            # ------------------------------------------------------------
            # 7. Build empirical preconditioned operator B_emp
            # ------------------------------------------------------------
            emp = _build_empirical_preconditioned_operator(
                A=A,
                Phi=Phi,
                G_inv=G_inv,
                C=C,
                tau=tau,
                reg=reg,
            )
            M_emp = emp["M_emp"]
            P_emp = emp["P_emp"]
            B_emp = emp["B_emp"]

            # ------------------------------------------------------------
            # 8. Build probe-space operators for dense prediction plotting
            # ------------------------------------------------------------
            P_probe_th = (Phi_probe @ M_th @ G_inv @ Phi.T) / n_train
            P_probe_emp = (Phi_probe @ M_emp @ G_inv @ Phi.T) / n_train

            B_probe_th = P_probe_th @ A
            B_probe_emp = P_probe_emp @ A

            # ------------------------------------------------------------
            # 9. Run the three flows with the same eta
            # ------------------------------------------------------------
            out_base = run_operator_gd(
                A_train=A,
                y_train=y_train,
                eta=eta,
                steps=steps,
                y_pred_train_0=jnp.zeros_like(y_train),
            )

            out_th = run_operator_gd(
                A_train=B_th,
                y_train=y_train,
                eta=eta,
                steps=steps,
                y_pred_train_0=jnp.zeros_like(y_train),
            )

            out_emp = run_operator_gd(
                A_train=B_emp,
                y_train=y_train,
                eta=eta,
                steps=steps,
                y_pred_train_0=jnp.zeros_like(y_train),
            )

            # ------------------------------------------------------------
            # 10. Project residuals onto sampled Fourier basis columns
            # ------------------------------------------------------------
            mode_proj_base = out_base["r_train"] @ Phi_unit
            mode_proj_th = out_th["r_train"] @ Phi_unit
            mode_proj_emp = out_emp["r_train"] @ Phi_unit

            mode_curves_base = _normalized_abs_curves(mode_proj_base)
            mode_curves_th = _normalized_abs_curves(mode_proj_th)
            mode_curves_emp = _normalized_abs_curves(mode_proj_emp)

            mode_proj_ext_base = out_base["r_train"] @ Phi_unit_ext
            mode_proj_ext_th = out_th["r_train"] @ Phi_unit_ext
            mode_proj_ext_emp = out_emp["r_train"] @ Phi_unit_ext

            mode_curves_ext_base = _normalized_abs_curves(mode_proj_ext_base)
            mode_curves_ext_th = _normalized_abs_curves(mode_proj_ext_th)
            mode_curves_ext_emp = _normalized_abs_curves(mode_proj_ext_emp)

            # ------------------------------------------------------------
            # 11. Reconstruct final dense probe predictions
            # ------------------------------------------------------------
            y_pred_probe_base_final = _final_probe_prediction_from_residuals(
                B_probe=A_probe,
                r_train=out_base["r_train"],
                eta=eta,
            )
            y_pred_probe_th_final = _final_probe_prediction_from_residuals(
                B_probe=B_probe_th,
                r_train=out_th["r_train"],
                eta=eta,
            )
            y_pred_probe_emp_final = _final_probe_prediction_from_residuals(
                B_probe=B_probe_emp,
                r_train=out_emp["r_train"],
                eta=eta,
            )

            # ------------------------------------------------------------
            # 12. Diagnostics: compare theory vs empirical objects
            # ------------------------------------------------------------
            C_vs_Lambda = matrix_error_metrics(C - Lambda_n)
            M_th_vs_emp = matrix_error_metrics(M_th - M_emp)
            P_th_vs_emp = matrix_error_metrics(P_th - P_emp)
            B_th_vs_emp = matrix_error_metrics(B_th - B_emp)

            H_th = (Phi.T @ B_th @ Phi) / n_train
            H_emp = (Phi.T @ B_emp @ Phi) / n_train

            C_th = compressed_operator_from_lemma_objects(G, H_th, reg=reg)
            C_emp = compressed_operator_from_lemma_objects(G, H_emp, reg=reg)

            C_th_vs_emp = matrix_error_metrics(C_th - C_emp)

            arrays_to_save = {
                # geometry / target
                "gamma_train": np.asarray(gamma_train),
                "X_train": np.asarray(X_train),
                "y_train": np.asarray(y_train),
                # basis
                "Phi": np.asarray(Phi),
                "Phi_unit": np.asarray(Phi_unit),
                "mode_names": np.asarray(mode_names),
                "mode_freqs": np.asarray(mode_freqs),
                "mode_types": np.asarray(mode_types),
                "Phi_unit_ext": np.asarray(Phi_unit_ext),
                "mode_names_ext": np.asarray(mode_names_ext),
                "mode_freqs_ext": np.asarray(mode_freqs_ext),
                "mode_types_ext": np.asarray(mode_types_ext),
                # compact operator/theory objects
                "G": np.asarray(G),
                "H": np.asarray(H),
                "C": np.asarray(C),
                "Lambda_n": np.asarray(Lambda_n),
                "lambda_by_k": np.asarray(lambda_by_k),
                "lambda_n_diag": np.asarray(lambda_n_diag),
                "g0": np.asarray([g0]),
                # compact preconditioner objects
                "M_th": np.asarray(M_th),
                "M_emp": np.asarray(M_emp),
                "C_th": np.asarray(C_th),
                "C_emp": np.asarray(C_emp),
                # diagnostics
                "C_vs_Lambda_max": np.asarray(C_vs_Lambda["max"]),
                "C_vs_Lambda_fro": np.asarray(C_vs_Lambda["fro"]),
                "M_th_vs_emp_max": np.asarray(M_th_vs_emp["max"]),
                "M_th_vs_emp_fro": np.asarray(M_th_vs_emp["fro"]),
                "P_th_vs_emp_max": np.asarray(P_th_vs_emp["max"]),
                "P_th_vs_emp_fro": np.asarray(P_th_vs_emp["fro"]),
                "B_th_vs_emp_max": np.asarray(B_th_vs_emp["max"]),
                "B_th_vs_emp_fro": np.asarray(B_th_vs_emp["fro"]),
                "C_th_vs_emp_max": np.asarray(C_th_vs_emp["max"]),
                "C_th_vs_emp_fro": np.asarray(C_th_vs_emp["fro"]),
                # dynamics
                "eta": np.asarray([eta]),
                "loss_base": np.asarray(out_base["loss"]),
                "loss_th": np.asarray(out_th["loss"]),
                "loss_emp": np.asarray(out_emp["loss"]),
                # mode projections and normalized mode curves
                "mode_proj_base": np.asarray(mode_proj_base),
                "mode_proj_th": np.asarray(mode_proj_th),
                "mode_proj_emp": np.asarray(mode_proj_emp),
                "mode_curves_base": np.asarray(mode_curves_base),
                "mode_curves_th": np.asarray(mode_curves_th),
                "mode_curves_emp": np.asarray(mode_curves_emp),
                "mode_proj_ext_base": np.asarray(mode_proj_ext_base),
                "mode_proj_ext_th": np.asarray(mode_proj_ext_th),
                "mode_proj_ext_emp": np.asarray(mode_proj_ext_emp),
                "mode_curves_ext_base": np.asarray(mode_curves_ext_base),
                "mode_curves_ext_th": np.asarray(mode_curves_ext_th),
                "mode_curves_ext_emp": np.asarray(mode_curves_ext_emp),
                # dense probe predictions
                "y_pred_probe_base_final": np.asarray(y_pred_probe_base_final),
                "y_pred_probe_th_final": np.asarray(y_pred_probe_th_final),
                "y_pred_probe_emp_final": np.asarray(y_pred_probe_emp_final),
            }

            if save_profile == "full":
                arrays_to_save.update(
                    {
                        # large dense operators
                        "A": np.asarray(A),
                        "P_th": np.asarray(P_th),
                        "P_emp": np.asarray(P_emp),
                        "B_th": np.asarray(B_th),
                        "B_emp": np.asarray(B_emp),
                        # full train residual trajectories
                        "r_train_base": np.asarray(out_base["r_train"]),
                        "r_train_th": np.asarray(out_th["r_train"]),
                        "r_train_emp": np.asarray(out_emp["r_train"]),
                    }
                )

            save_npz(run_path, **arrays_to_save)

            runs_manifest[run_key] = f"runs/{run_key}.npz"

    runs_manifest = {}
    for run_key in sorted(expected_run_keys):
        run_path = runs_dir / f"{run_key}.npz"
        if run_path.exists():
            runs_manifest[run_key] = f"runs/{run_path.name}"

    missing_run_keys = sorted(expected_run_keys - set(runs_manifest.keys()))
    if missing_run_keys:
        partial_manifest = {
            "probe_geometry": "probe_geometry.npz",
            "runs": runs_manifest,
            "missing_runs": missing_run_keys,
            "meta": {
                "n_trains": n_trains,
                "seeds": seed_list,
                "K_max": K_max,
                "kernel": kernel_name,
                "steps": steps,
                "eta": eta,
                "tau": tau,
                "reg": reg,
                "n_probe": n_probe,
                "target_Ks": target_cfg["Ks"],
                "target_amps": target_cfg["amps"],
                "target_phases": target_cfg["phases"],
                "noise_std": noise_std,
                "mode_compare": mode_compare,
                "mode_projection_k_max": mode_projection_k_max,
                "save_profile": save_profile,
                "large_dense_keys_saved": save_profile == "full",
            },
            "runtime_sec": round(time.time() - t0, 2),
        }
        save_json(save_dir / "manifest.partial.json", partial_manifest)
        print(
            "\nRun incomplete. Wrote manifest.partial.json and skipped manifest.json. "
            "Rerun with resume enabled to continue."
        )
        print(f"Missing run file count: {len(missing_run_keys)}")
        print(f"First missing runs: {missing_run_keys[:10]}")
        return

    manifest = {
        "probe_geometry": "probe_geometry.npz",
        "runs": runs_manifest,
        "meta": {
            "n_trains": n_trains,
            "seeds": seed_list,
            "K_max": K_max,
            "kernel": kernel_name,
            "steps": steps,
            "eta": eta,
            "tau": tau,
            "reg": reg,
            "n_probe": n_probe,
            "target_Ks": target_cfg["Ks"],
            "target_amps": target_cfg["amps"],
            "target_phases": target_cfg["phases"],
            "noise_std": noise_std,
            "mode_compare": mode_compare,
            "mode_projection_k_max": mode_projection_k_max,
            "save_profile": save_profile,
            "large_dense_keys_saved": save_profile == "full",
        },
        "runtime_sec": round(time.time() - t0, 2),
    }

    partial_manifest_path = save_dir / "manifest.partial.json"
    if partial_manifest_path.exists():
        try:
            partial_manifest_path.unlink()
        except Exception:
            pass

    save_json(save_dir / "manifest.json", manifest)
    print(f"\nDone. Preconditioning artifacts saved to {save_dir}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(
            "Usage: python3 -m experiments.kernel_circle_preconditioning "
            "configs/kernel_circle_preconditioning.yaml"
        )
    else:
        run(sys.argv[1])
