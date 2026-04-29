# ------------------------------------------------------------
# experiments/kernel_gd_circle_theory.py
# ------------------------------------------------------------

import sys
import time

import jax
import numpy as np
import yaml

from core.data import FourierTarget, f_star_gamma, make_probe_circle
from core.kernel_analysis import (
    continuum_fourier_eigenvalues_bias,
    continuum_fourier_eigenvalues_nobias,
    kernel_eigendecomposition,
    project_residuals_onto_eigenvectors,
    project_residuals_onto_fourier_modes,
)
from core.kernel_circle import kernel_matrix_from_gamma
from core.kernel_dynamics import gp_init_from_gamma, run_kernel_gd, zero_init_from_gamma
from utils.artifacts import make_run_dir, save_json, save_npz, write_config_copy


def _resolve_n_trains(n_train_cfg) -> list[int]:
    if isinstance(n_train_cfg, (list, tuple)):
        n_trains = [int(n) for n in n_train_cfg]
    else:
        n_trains = [int(n_train_cfg)]

    if len(n_trains) == 0:
        raise ValueError("data.n_train must provide at least one value.")
    if any(n <= 0 for n in n_trains):
        raise ValueError(f"All n_train values must be positive, got {n_trains}.")
    return n_trains


def _continuum_lambda_by_k(kernel_name: str, ks_compare: np.ndarray) -> np.ndarray:
    if kernel_name == "bias":
        return np.asarray(continuum_fourier_eigenvalues_bias(ks_compare))
    if kernel_name == "nobias":
        return np.asarray(continuum_fourier_eigenvalues_nobias(ks_compare))
    raise ValueError(f"Unknown kernel='{kernel_name}'. Expected 'bias' or 'nobias'.")


def run(config_path: str):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    exp_cfg = cfg["experiment"]
    data_cfg = cfg["data"]
    target_cfg = cfg["target"]
    kernel_cfg = cfg["kernel"]
    train_cfg = cfg["train"]
    analysis_cfg = cfg["analysis"]

    seed = int(exp_cfg["seed"])
    n_trains = _resolve_n_trains(data_cfg["n_train"])

    Ks = np.array(target_cfg["Ks"], dtype=float)
    amps = np.array(target_cfg["amps"], dtype=float)
    phases = np.array(target_cfg["phases"], dtype=float)

    kernel_name = kernel_cfg["name"]
    init_name = kernel_cfg["init"]

    steps = int(train_cfg["steps"])
    eta_cfg = train_cfg.get("eta", None)
    eta_scale = float(train_cfg.get("eta_scale", 0.05))

    compare_first_k = int(analysis_cfg.get("compare_first_k", 20))
    fourier_modes = list(analysis_cfg.get("fourier_modes", [0, 1, 2, 3, 4, 8]))

    save_dir = make_run_dir(exp_cfg.get("save_dir", "results"), exp_cfg["name"])
    write_config_copy(save_dir, cfg)

    print("=== Kernel GD circle theory-validation run ===")
    print(f"Saving results to: {save_dir}\n")
    t0 = time.time()

    ft = FourierTarget(Ks=Ks, amps=amps, phases=phases)
    runs_manifest = {}
    single_layout = len(n_trains) == 1

    for n_train in n_trains:
        print(f"=== n_train = {n_train} ===")

        run_key = f"size_{n_train}"
        run_dir = save_dir if single_layout else save_dir / "runs" / run_key

        # ------------------------------------------------------------
        # Evenly spaced circle grid
        # ------------------------------------------------------------
        gamma_train, X_train, _ = make_probe_circle(n_train)
        y_train = f_star_gamma(gamma_train, ft)

        save_npz(
            run_dir / "training_task.npz",
            gamma_train=np.asarray(gamma_train),
            X_train=np.asarray(X_train),
            y_train=np.asarray(y_train),
        )

        # ------------------------------------------------------------
        # Kernel matrix + eigendecomposition
        # ------------------------------------------------------------
        theta_xx = kernel_matrix_from_gamma(gamma_train, kernel=kernel_name)
        evals_emp, evecs = kernel_eigendecomposition(theta_xx)

        # For theory validation on analytic circle kernels, compare discrete
        # eigenvalues against n * lambda_k(cont).
        has_continuum_spectrum = False
        if kernel_name in {"bias", "nobias"}:
            ks_compare = np.arange(compare_first_k + 1)
            lambda_cont = _continuum_lambda_by_k(kernel_name, ks_compare)
            lambda_disc_pred = n_train * lambda_cont

            save_npz(
                run_dir / "continuum_spectrum.npz",
                ks_compare=ks_compare,
                lambda_cont=lambda_cont,
                lambda_disc_pred=lambda_disc_pred,
            )
            has_continuum_spectrum = True

        # ------------------------------------------------------------
        # Step size
        # ------------------------------------------------------------
        lam_max = float(np.asarray(evals_emp[0]))
        eta = float(eta_cfg) if eta_cfg is not None else float(eta_scale / lam_max)

        # ------------------------------------------------------------
        # Initialization
        # ------------------------------------------------------------
        if init_name == "zero":
            y_pred_0 = zero_init_from_gamma(gamma_train)
        elif init_name == "gp":
            y_pred_0 = gp_init_from_gamma(jax.random.PRNGKey(seed), gamma_train)
        else:
            raise ValueError(f"Unknown init='{init_name}'. Expected 'zero' or 'gp'.")

        # ------------------------------------------------------------
        # Kernel GD
        # ------------------------------------------------------------
        out = run_kernel_gd(
            gamma=gamma_train,
            y=y_train,
            eta=eta,
            steps=steps,
            kernel=kernel_name,
            y_pred_0=y_pred_0,
        )

        save_npz(
            run_dir / "kernel_objects.npz",
            theta_xx=np.asarray(theta_xx),
            evals_emp=np.asarray(evals_emp),
            evecs=np.asarray(evecs),
        )

        save_npz(
            run_dir / "kernel_dynamics.npz",
            y_pred=np.asarray(out["y_pred"]),
            r=np.asarray(out["r"]),
            loss=np.asarray(out["loss"]),
            eta=np.asarray([eta]),
        )

        # ------------------------------------------------------------
        # Mode projections
        # ------------------------------------------------------------
        eig_coeffs = project_residuals_onto_eigenvectors(out["r"], evecs)
        fourier_proj = project_residuals_onto_fourier_modes(
            out["r"],
            gamma_train,
            ks=fourier_modes,
        )

        mode_payload = {
            "eig_coeffs": np.asarray(eig_coeffs),
        }
        for name, values in fourier_proj.items():
            mode_payload[name] = np.asarray(values)

        save_npz(
            run_dir / "mode_projections.npz",
            **mode_payload,
        )

        if single_layout:
            training_task_rel = "training_task.npz"
            kernel_objects_rel = "kernel_objects.npz"
            kernel_dynamics_rel = "kernel_dynamics.npz"
            mode_projections_rel = "mode_projections.npz"
            continuum_spectrum_rel = "continuum_spectrum.npz"
        else:
            training_task_rel = f"runs/{run_key}/training_task.npz"
            kernel_objects_rel = f"runs/{run_key}/kernel_objects.npz"
            kernel_dynamics_rel = f"runs/{run_key}/kernel_dynamics.npz"
            mode_projections_rel = f"runs/{run_key}/mode_projections.npz"
            continuum_spectrum_rel = f"runs/{run_key}/continuum_spectrum.npz"

        runs_manifest[run_key] = {
            "training_task": training_task_rel,
            "kernel_objects": kernel_objects_rel,
            "kernel_dynamics": kernel_dynamics_rel,
            "mode_projections": mode_projections_rel,
            "meta": {
                "kernel": kernel_name,
                "init": init_name,
                "n_train": n_train,
                "steps": steps,
                "eta": eta,
                "eta_scale": eta_scale,
                "lambda_max_emp": lam_max,
                "target_Ks": target_cfg["Ks"],
                "target_amps": target_cfg["amps"],
                "target_phases": target_cfg["phases"],
                "fourier_modes": fourier_modes,
            },
        }
        if has_continuum_spectrum:
            runs_manifest[run_key]["continuum_spectrum"] = continuum_spectrum_rel

    # ------------------------------------------------------------
    # Manifest
    # ------------------------------------------------------------
    manifest = {
        "runs": runs_manifest,
        "meta": {
            "kernel": kernel_name,
            "init": init_name,
            "n_trains": n_trains,
            "steps": steps,
            "eta_config": eta_cfg,
            "eta_scale": eta_scale,
            "target_Ks": target_cfg["Ks"],
            "target_amps": target_cfg["amps"],
            "target_phases": target_cfg["phases"],
            "fourier_modes": fourier_modes,
        },
        "runtime_sec": round(time.time() - t0, 2),
    }

    # Backward-compatible top-level keys when there is a single n_train.
    if single_layout:
        single_key = f"size_{n_trains[0]}"
        single = runs_manifest[single_key]
        manifest["training_task"] = single["training_task"]
        manifest["kernel_objects"] = single["kernel_objects"]
        manifest["kernel_dynamics"] = single["kernel_dynamics"]
        manifest["mode_projections"] = single["mode_projections"]
        if "continuum_spectrum" in single:
            manifest["continuum_spectrum"] = single["continuum_spectrum"]

        manifest["meta"] = {
            **single["meta"],
            "n_trains": n_trains,
        }

    save_json(save_dir / "manifest.json", manifest)

    print(f"Done. Theory-validation artifacts saved to {save_dir}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(
            "Usage: python3 -m experiments.kernel_gd_circle_theory "
            "configs/kernel_gd_circle_theory.yaml"
        )
    else:
        run(sys.argv[1])
