# ------------------------------------------------------------
# experiments/kernel_circle_lemma_validation.py
# ------------------------------------------------------------

import sys
import time

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import yaml

from core.data import FourierTarget, f_star_gamma, make_probe_circle
from core.fourier_basis_circle import build_real_fourier_basis
from core.kernel_analysis import (
    compute_lemma_error_metrics,
    compute_lemma_objects,
    continuum_fourier_eigenvalues_bias,
    continuum_fourier_eigenvalues_nobias,
    finite_n_diagonal_benchmark,
    per_mode_action_relative_errors,
)
from core.kernel_circle import (
    TWO_PI,
    kernel_cross_matrix_from_gamma,
    kernel_matrix_from_gamma,
)
from core.kernel_dynamics import run_operator_gd
from utils.artifacts import make_run_dir, save_json, save_npz, write_config_copy


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


def _continuum_lambda_by_k(kernel_name: str, K_max: int) -> jnp.ndarray:
    ks = jnp.arange(K_max + 1)
    if kernel_name == "bias":
        return continuum_fourier_eigenvalues_bias(ks)
    if kernel_name == "nobias":
        return continuum_fourier_eigenvalues_nobias(ks)
    raise ValueError(f"Unknown kernel='{kernel_name}'. Expected 'bias' or 'nobias'.")


def _build_mode_decay_reference(
    r_train: jnp.ndarray,
    Phi_unit: jnp.ndarray,
    lambda_n_diag: jnp.ndarray,
    eta: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Observed sampled-mode coefficients and ideal diagonal decay reference.

    Observed:
        alpha_obs[t, p] = <r_t, phi_unit_p>

    Ideal:
        alpha_ideal[t, p] = (1 - eta * lambda_p^(n))^t alpha_obs[0, p]
    """
    alpha_obs = r_train @ Phi_unit
    t = jnp.arange(r_train.shape[0], dtype=lambda_n_diag.dtype)[:, None]
    decay = (1.0 - eta * lambda_n_diag[None, :]) ** t
    alpha_ideal = decay * alpha_obs[0:1, :]
    return alpha_obs, alpha_ideal


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
    analysis_cfg = cfg.get("analysis", {})
    artifacts_cfg = cfg.get("artifacts", {})

    base_seed = int(exp_cfg.get("seed", 0))
    n_trains = [int(n) for n in data_cfg["n_trains"]]
    n_eval = int(data_cfg.get("n_eval", 720))
    noise_std = float(data_cfg.get("noise_std", 0.0))

    K_max = int(basis_cfg["K_max"])

    Ks = np.array(target_cfg["Ks"], dtype=float)
    amps = np.array(target_cfg["amps"], dtype=float)
    phases = np.array(target_cfg["phases"], dtype=float)

    kernel_name = kernel_cfg["name"]

    steps = int(train_cfg.get("steps", 0))
    eta_cfg = train_cfg.get("eta", None)
    eta_scale = float(train_cfg.get("eta_scale", 0.05))
    save_every = int(train_cfg.get("save_every", 100))
    run_dynamics = bool(train_cfg.get("run_dynamics", True))

    save_full_matrices = bool(artifacts_cfg.get("save_full_matrices", True))

    seed_list = _resolve_seed_list(sweep_cfg["seeds"], base_seed)
    mode_compare = list(analysis_cfg.get("mode_compare", [0, 1, 2, 3, 4, 5]))

    save_dir = make_run_dir(exp_cfg.get("save_dir", "results"), exp_cfg["name"])
    write_config_copy(save_dir, cfg)

    print("=== Kernel circle lemma-validation run ===")
    print(f"Saving results to: {save_dir}\n")
    print(f"Run dynamics: {run_dynamics}")
    print(f"Save full matrices: {save_full_matrices}\n")
    t0 = time.time()

    # ------------------------------------------------------------
    # Dense evaluation grid (saved once)
    # ------------------------------------------------------------
    gamma_eval, X_eval, _ = make_probe_circle(n_eval)
    ft = FourierTarget(Ks=Ks, amps=amps, phases=phases)
    y_eval = f_star_gamma(gamma_eval, ft)

    save_npz(
        save_dir / "probe_geometry.npz",
        gamma_eval=np.asarray(gamma_eval),
        X_eval=np.asarray(X_eval),
        y_eval=np.asarray(y_eval),
    )

    runs_manifest = {}
    runs_dir = save_dir / "runs"

    # ------------------------------------------------------------
    # Sweep over sample sizes and seeds
    # ------------------------------------------------------------
    for n_train in n_trains:
        print(f"=== n_train = {n_train} ===")

        for seed in seed_list:
            print(f"---- seed = {seed} ----")

            key = jr.PRNGKey(seed)
            gamma_train = _sample_uniform_gamma(key, n_train)

            X_train = (
                _circle_points_from_gamma(gamma_train) if save_full_matrices else None
            )

            y_train = None
            if run_dynamics or save_full_matrices:
                y_train = f_star_gamma(gamma_train, ft)
                if noise_std > 0:
                    noise = noise_std * jr.normal(
                        jr.fold_in(key, 1), shape=y_train.shape
                    )
                    y_train = y_train + noise

            # ------------------------------------------------------------
            # Basis
            # ------------------------------------------------------------
            basis = build_real_fourier_basis(gamma_train, K_max=K_max)
            Phi = basis["Phi"]
            Phi_unit = basis["Phi_unit"]
            mode_names = basis["mode_names"]
            mode_freqs = basis["mode_freqs"]
            mode_types = basis["mode_types"]

            # ------------------------------------------------------------
            # Kernel/operator objects
            # ------------------------------------------------------------
            K = kernel_matrix_from_gamma(gamma_train, kernel=kernel_name)
            A = K / n_train

            lambda_by_k = _continuum_lambda_by_k(kernel_name, K_max)
            g0 = _kernel_diag_value(kernel_name)
            lambda_n_diag = finite_n_diagonal_benchmark(
                mode_freqs=mode_freqs,
                lambda_by_k=lambda_by_k,
                g0=g0,
                n=n_train,
            )

            lemma_objs = compute_lemma_objects(
                Phi=Phi,
                A=A,
                lambda_n_diag=lambda_n_diag,
            )
            metrics = compute_lemma_error_metrics(
                G=lemma_objs["G"],
                H=lemma_objs["H"],
                E=lemma_objs["E"],
                Lambda_n=lemma_objs["Lambda_n"],
            )
            per_mode_relerr = per_mode_action_relative_errors(
                lemma_objs["E"],
                Phi,
            )

            eta = None
            lambda_max_A = None
            out = None
            mode_proj_train = None
            mode_proj_train_ideal = None

            if run_dynamics:
                if y_train is None:
                    raise ValueError(
                        "y_train must be available when train.run_dynamics=True."
                    )

                # ------------------------------------------------------------
                # Step size on normalized operator scale
                # ------------------------------------------------------------
                evals_A = jnp.linalg.eigvalsh(A)
                lambda_max_A = float(np.asarray(evals_A[-1]))
                eta = (
                    float(eta_cfg)
                    if eta_cfg is not None
                    else float(eta_scale / lambda_max_A)
                )

                # ------------------------------------------------------------
                # Eval-train operator
                # ------------------------------------------------------------
                A_eval_train = (
                    kernel_cross_matrix_from_gamma(
                        gamma_eval,
                        gamma_train,
                        kernel=kernel_name,
                    )
                    / n_train
                )

                # ------------------------------------------------------------
                # Zero-init operator GD
                # ------------------------------------------------------------
                out = run_operator_gd(
                    A_train=A,
                    y_train=y_train,
                    eta=eta,
                    steps=steps,
                    y_pred_train_0=jnp.zeros_like(y_train),
                    A_eval_train=A_eval_train,
                    y_pred_eval_0=jnp.zeros_like(y_eval),
                    save_every=save_every,
                )

                # ------------------------------------------------------------
                # Observed vs ideal sampled-mode decay
                # ------------------------------------------------------------
                mode_proj_train, mode_proj_train_ideal = _build_mode_decay_reference(
                    r_train=out["r_train"],
                    Phi_unit=Phi_unit,
                    lambda_n_diag=lambda_n_diag,
                    eta=eta,
                )

            # ------------------------------------------------------------
            # Save per-run NPZ
            # ------------------------------------------------------------
            run_key = f"size_{n_train}_seed_{seed}"
            run_path = runs_dir / f"{run_key}.npz"

            arrays_to_save = {
                # lemma scalar metrics
                "gram_err_max": np.asarray(metrics["gram_err_max"]),
                "gram_err_fro": np.asarray(metrics["gram_err_fro"]),
                "comp_err_max": np.asarray(metrics["comp_err_max"]),
                "comp_err_fro": np.asarray(metrics["comp_err_fro"]),
                "action_err_max": np.asarray(metrics["action_err_max"]),
                "action_err_fro": np.asarray(metrics["action_err_fro"]),
                # lemma vector metrics
                "per_mode_action_relerr": np.asarray(per_mode_relerr),
                "mode_names": np.asarray(mode_names),
                "mode_freqs": np.asarray(mode_freqs),
                "mode_types": np.asarray(mode_types),
                # compact theory metadata
                "Lambda_n_diag": np.asarray(lambda_n_diag),
                "g0": np.asarray([g0]),
            }

            if save_full_matrices:
                arrays_to_save.update(
                    {
                        # geometry
                        "gamma_train": np.asarray(gamma_train),
                        "X_train": np.asarray(X_train),
                        "y_train": np.asarray(y_train),
                        # basis and operator objects
                        "Phi": np.asarray(Phi),
                        "A": np.asarray(A),
                        "G": np.asarray(lemma_objs["G"]),
                        "H": np.asarray(lemma_objs["H"]),
                    }
                )

            if run_dynamics:
                arrays_to_save.update(
                    {
                        # dynamics
                        "y_pred_train": np.asarray(out["y_pred_train"]),
                        "r_train": np.asarray(out["r_train"]),
                        "loss": np.asarray(out["loss"]),
                        "eta": np.asarray([eta]),
                        "snapshot_steps": np.asarray(out["snapshot_steps"]),
                        "y_pred_eval_snapshots": np.asarray(
                            out["y_pred_eval_snapshots"]
                        ),
                        "y_pred_eval_final": np.asarray(out["y_pred_eval_final"]),
                        # decay diagnostics
                        "mode_proj_train": np.asarray(mode_proj_train),
                        "mode_proj_train_ideal": np.asarray(mode_proj_train_ideal),
                        # useful metadata
                        "lambda_max_A": np.asarray([lambda_max_A]),
                    }
                )

            save_npz(run_path, **arrays_to_save)

            runs_manifest[run_key] = f"runs/{run_key}.npz"

    # ------------------------------------------------------------
    # Top-level manifest
    # ------------------------------------------------------------
    manifest = {
        "probe_geometry": "probe_geometry.npz",
        "runs": runs_manifest,
        "meta": {
            "n_trains": n_trains,
            "seeds": seed_list,
            "K_max": K_max,
            "kernel": kernel_name,
            "run_dynamics": run_dynamics,
            "save_full_matrices": save_full_matrices,
            "steps": steps,
            "eta_config": eta_cfg,
            "eta_scale": eta_scale,
            "save_every": save_every,
            "target_Ks": target_cfg["Ks"],
            "target_amps": target_cfg["amps"],
            "target_phases": target_cfg["phases"],
            "mode_compare": mode_compare,
            "noise_std": noise_std,
        },
        "runtime_sec": round(time.time() - t0, 2),
    }

    save_json(save_dir / "manifest.json", manifest)
    print(f"\nDone. Lemma-validation artifacts saved to {save_dir}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(
            "Usage: python3 -m experiments.kernel_circle_lemma_validation "
            "configs/kernel_circle_lemma_validation.yaml"
        )
    else:
        run(sys.argv[1])
