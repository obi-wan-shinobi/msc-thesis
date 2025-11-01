# ------------------------------------------------------------
# experiments/ntk_regression_convergence.py
# ------------------------------------------------------------
"""
Experiment 2 & 3 — Finite-width convergence to analytic NTK predictor
on regression tasks defined on the unit circle.

Supports:
  - Simple task:    f*(x)=x1·x2
  - Fourier mixture: custom Ks, amps, phases
"""

import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
import yaml
from core.analysis import analytic_ntk_predictor, relerr
from core.data import (
    f_star_gamma,
    make_circle_regression_dataset,
    make_fourier_target,
    sample_train_on_circle,
)
from core.model import build_mlp
from core.train import train
from utils.artifacts import make_run_dir, save_json, save_npz, write_config_copy


# ------------------------------------------------------------
# Main experiment logic
# ------------------------------------------------------------
def run(config_path: str):
    # --- Load config ---
    cfg = yaml.safe_load(open(config_path))
    exp = cfg["experiment"]
    widths = cfg["sweep"]["widths"]
    n_seeds = int(cfg["sweep"]["seeds"])
    depth_hidden = cfg["model"]["depth_hidden"]
    b_std = cfg["model"]["b_std"]
    steps = cfg["train"]["steps"]
    lr = cfg["train"]["lr"]
    log_every = cfg["train"]["log_every"]
    reg = cfg["analysis"]["reg"]

    # --- Determine task type ---
    task_type = exp.get("task_type", "simple")  # 'simple' or 'fourier'

    # --- Timestamped run directory ---
    base_dir = exp.get("save_dir", "results")
    exp_name = exp["name"]
    save_dir = make_run_dir(base_dir, exp_name)
    write_config_copy(save_dir, cfg)

    print(f"=== NTK regression convergence ({task_type}) ===")
    print(f"Saving results to: {save_dir}\n")
    t0 = time.time()

    # ============================================================
    # 1. Dataset
    # ============================================================
    if task_type == "simple":
        gamma_eval, X_eval, gamma_train, X_train, y_train = (
            make_circle_regression_dataset(
                cfg["data"]["n_eval"], cfg["data"]["M_train"], exp["seed"]
            )
        )
        y_eval_true = X_eval[:, 0] * X_eval[:, 1]
        data_file = "data_simple.npz"
        save_npz(
            save_dir / data_file,
            gamma_eval=np.asarray(gamma_eval),
            X_eval=np.asarray(X_eval),
            gamma_train=np.asarray(gamma_train),
            X_train=np.asarray(X_train),
            y_train=np.asarray(y_train),
            y_eval_true=np.asarray(y_eval_true),
        )

    elif task_type == "fourier":
        Ks = jnp.array(cfg["target"]["Ks"])
        amps = jnp.array(cfg["target"]["amps"])
        phases = jnp.array(cfg["target"]["phases"])
        spec = make_fourier_target(Ks, amps, phases)

        N_eval = cfg["data"]["n_eval"]
        gamma_eval = jnp.linspace(-jnp.pi, jnp.pi, N_eval, endpoint=False)
        X_eval = jnp.stack([jnp.cos(gamma_eval), jnp.sin(gamma_eval)], axis=1)
        y_eval_true = f_star_gamma(gamma_eval, spec)

        M_train = cfg["data"]["M_train"]
        X_train, gamma_train, y_train = sample_train_on_circle(
            gamma_eval,
            X_eval,
            spec,
            M_train=M_train,
            key=jax.random.PRNGKey(exp["seed"]),
            add_noise_std=cfg["data"].get("noise_std", 0.0),
        )

        data_file = "data_fourier.npz"
        save_npz(
            save_dir / data_file,
            gamma_eval=np.asarray(gamma_eval),
            X_eval=np.asarray(X_eval),
            gamma_train=np.asarray(gamma_train),
            X_train=np.asarray(X_train),
            y_train=np.asarray(y_train),
            y_eval_true=np.asarray(y_eval_true),
            Ks=np.asarray(Ks),
            amps=np.asarray(amps),
            phases=np.asarray(phases),
        )

    else:
        raise ValueError(f"Unknown task_type: {task_type}")

    # ============================================================
    # 2. Analytic NTK predictor (∞-width)
    # ============================================================
    _, _, kernel_ref = build_mlp(width=512, b_std=b_std, depth_hidden=depth_hidden)
    y_inf = analytic_ntk_predictor(X_train, y_train, X_eval, kernel_ref, reg)
    save_npz(save_dir / "analytic_predictor.npz", y_inf=np.asarray(y_inf), reg=reg)

    # ============================================================
    # 3. Width × seed sweep
    # ============================================================
    predictions_dir = save_dir / "predictions"
    predictions_dir.mkdir(exist_ok=True)
    manifest_preds = {}

    for w in widths:
        print(f"Width {w}: averaging over {n_seeds} seeds ...")
        preds, errs = [], []

        init_fn, apply_fn, _ = build_mlp(
            width=w, b_std=b_std, depth_hidden=depth_hidden
        )

        for s in range(n_seeds):
            print(f"  Seed {s}")
            _, params0 = init_fn(jax.random.PRNGKey(s), X_train.shape)
            paramsT = train(
                params0,
                apply_fn,
                X_train,
                y_train,
                steps=steps,
                lr=lr,
                log_every=log_every,
            )
            y_pred = apply_fn(paramsT, X_eval).squeeze()
            preds.append(np.asarray(y_pred))
            errs.append(relerr(y_pred, y_inf))

        preds = np.stack(preds)
        errs = np.asarray(errs)
        save_npz(
            predictions_dir / f"width_{w}.npz",
            preds_all=preds,
            err_all=errs,
            pred_mean=preds.mean(0),
            pred_p10=np.percentile(preds, 10, axis=0),
            pred_p50=np.percentile(preds, 50, axis=0),
            pred_p90=np.percentile(preds, 90, axis=0),
        )
        manifest_preds[str(w)] = f"predictions/width_{w}.npz"

        mean_err = float(errs.mean())
        p10, p50, p90 = np.percentile(errs, [10, 50, 90])
        print(f"  [eval] RelErr mean={mean_err:.3e}, p50={p50:.3e}")

    # ============================================================
    # 4. Save manifest summary
    # ============================================================
    summary = {
        "dataset": data_file,
        "analytic_predictor": "analytic_predictor.npz",
        "predictions": manifest_preds,
        "meta": {
            "task_type": task_type,
            "depth_hidden": depth_hidden,
            "b_std": b_std,
            "train_steps": steps,
            "lr": lr,
            "reg": reg,
            "seeds": n_seeds,
            "widths": widths,
        },
        "runtime_sec": round(time.time() - t0, 2),
    }

    save_json(save_dir / "manifest.json", summary)
    print(f"\n Done. Saved results for {len(widths)} widths to {save_dir}")


# ------------------------------------------------------------
# Entry point
# ------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 -m experiments.ntk_regression_convergence <config.yaml>")
    else:
        run(sys.argv[1])
