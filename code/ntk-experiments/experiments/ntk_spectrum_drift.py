# ------------------------------------------------------------
# experiments/ntk_spectrum_drift.py
# ------------------------------------------------------------
"""
Experiment — NTK eigen-spectrum drift during training (Fourier mixture task)

Logs, for each (width, seed):
    - empirical NTK eigenvalues on the training subset of the circle
    - predictions on the full γ-grid
"""

import sys
import time
from pathlib import Path

import jax
import numpy as np
import yaml
from core.analysis import empirical_ntk_spectrum
from core.data import (
    FourierTarget,
    f_star_gamma,
    make_probe_circle,
    sample_train_on_circle,
)
from core.model import build_mlp
from core.train import train
from utils.artifacts import make_run_dir, save_json, save_npz, write_config_copy


# ------------------------------------------------------------
# Main experiment logic
# ------------------------------------------------------------
def run(config_path: str):
    cfg = yaml.safe_load(open(config_path))

    exp_cfg = cfg["experiment"]
    data_cfg = cfg["data"]
    target_cfg = cfg["target"]
    model_cfg = cfg["model"]
    train_cfg = cfg["train"]
    sweep_cfg = cfg["sweep"]

    widths = sweep_cfg["widths"]
    n_seeds = sweep_cfg["seeds"]

    steps = int(train_cfg["steps"])
    lr = float(train_cfg["lr"])
    log_every = int(train_cfg["log_every"])
    eval_every = int(train_cfg["eval_every"])

    n_eval = int(data_cfg["n_eval"])
    M_train = int(data_cfg["M_train"])
    noise_std = float(data_cfg.get("noise_std", 0.0))

    depth_hidden = int(model_cfg["depth_hidden"])
    b_std = float(model_cfg["b_std"])

    # Fourier target
    Ks = np.array(target_cfg["Ks"], dtype=float)
    amps = np.array(target_cfg["amps"], dtype=float)
    phases = np.array(target_cfg["phases"], dtype=float)
    ft = FourierTarget(Ks=Ks, amps=amps, phases=phases)

    # ------------------------------------------------------------
    # Create run dir
    # ------------------------------------------------------------
    save_dir = make_run_dir(exp_cfg.get("save_dir", "results"), exp_cfg["name"])
    write_config_copy(save_dir, cfg)

    print("=== NTK spectrum drift (Fourier task) ===")
    print(f"Saving results to: {save_dir}\n")
    t0 = time.time()

    # ------------------------------------------------------------
    # Full probe geometry
    # ------------------------------------------------------------
    gamma_eval, X_eval, _ = make_probe_circle(n_eval)
    y_eval = f_star_gamma(gamma_eval, ft)

    save_npz(
        save_dir / "probe_geometry.npz",
        gamma=gamma_eval,
        X_eval=X_eval,
        y_eval=y_eval,
    )

    # ------------------------------------------------------------
    # Training set from circle sampler
    # ------------------------------------------------------------
    X_train, gamma_train, y_train = sample_train_on_circle(
        gamma_eval,
        X_eval,
        ft,
        M_train=M_train,
        key=jax.random.PRNGKey(exp_cfg["seed"]),
        add_noise_std=noise_std,
    )

    save_npz(
        save_dir / "training_task.npz",
        X_train=X_train,
        gamma_train=gamma_train,
        y_train=y_train,
    )

    # ------------------------------------------------------------
    # Sweep: width × seed
    # ------------------------------------------------------------
    manifest = {}
    runs_dir = save_dir / "runs"
    runs_dir.mkdir(exist_ok=True)

    for w in widths:
        print(f"Width = {w}")

        spectra_list = []
        spectra_meta = []
        preds_list = []
        preds_meta = []

        for s in range(n_seeds):
            print(f"  Seed {s}")

            # Build model (fixed depth)
            init_fn, apply_fn, _ = build_mlp(
                width=w,
                depth_hidden=depth_hidden,
                b_std=b_std,
            )
            _, params = init_fn(jax.random.PRNGKey(s), X_train.shape)

            # ----- Snapshot at t = 0 -----
            lam0 = empirical_ntk_spectrum(apply_fn, params, X_train)
            spectra_list.append(np.asarray(lam0))
            spectra_meta.append([s, 0])

            pred0 = np.asarray(apply_fn(params, X_eval).squeeze())
            preds_list.append(pred0)
            preds_meta.append([s, 0])

            # ----- Training in eval_every-sized chunks -----
            current_step = 0
            remaining = steps

            while remaining > 0:
                chunk = min(eval_every, remaining)

                params = train(
                    params,
                    apply_fn,
                    X_train,
                    y_train,
                    steps=chunk,
                    lr=lr,
                    log_every=log_every,
                    return_history=False,
                )

                current_step += chunk
                remaining -= chunk

                # Snapshot
                lam_t = empirical_ntk_spectrum(apply_fn, params, X_train)
                spectra_list.append(np.asarray(lam_t))
                spectra_meta.append([s, current_step])

                pred_t = np.asarray(apply_fn(params, X_eval).squeeze())
                preds_list.append(pred_t)
                preds_meta.append([s, current_step])

                print(f"    step {current_step:6d} | snapshot saved")

        # Save this width block
        out_path = runs_dir / f"width_{w}.npz"
        save_npz(
            out_path,
            spectra=np.stack(spectra_list),
            spectra_meta=np.asarray(spectra_meta),
            preds=np.stack(preds_list),
            preds_meta=np.asarray(preds_meta),
            X_train=X_train,
            y_train=y_train,
        )

        manifest[str(w)] = str(out_path.relative_to(save_dir))

    # ------------------------------------------------------------
    # Manifest
    # ------------------------------------------------------------
    summary = {
        "probe_geometry": "probe_geometry.npz",
        "training_task": "training_task.npz",
        "runs": manifest,
        "meta": {
            "widths": widths,
            "seeds": n_seeds,
            "steps": steps,
            "lr": lr,
            "eval_every": eval_every,
            "depth_hidden": depth_hidden,
        },
        "runtime_sec": round(time.time() - t0, 2),
    }
    save_json(save_dir / "manifest.json", summary)

    print(f"\nDone. Spectrum + prediction logs saved to {save_dir}")


# ------------------------------------------------------------
# Entry
# ------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(
            "Usage: python3 -m experiments.ntk_spectrum_drift "
            "configs/ntk_spectrum_drift.yaml"
        )
    else:
        run(sys.argv[1])
