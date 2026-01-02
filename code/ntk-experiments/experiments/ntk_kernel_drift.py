# ------------------------------------------------------------
# experiments/ntk_kernel_drift.py
# ------------------------------------------------------------
"""
Experiment — NTK kernel drift during training (Fourier mixture task)

For each (width, seed), this logs:
    - empirical NTK on the training set K_train_train(t)
    - NTK between eval grid and training set K_eval_train(t)
    - network predictions on the eval grid

Goal: detect when K_train_train(t) stops changing, and check whether
late-time training is equivalent to kernel regression with a frozen kernel.
"""

import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import yaml
from core.analysis import empirical_ntk_matrix
from core.data import f_star_gamma, make_fourier_target, sample_train_on_circle
from core.model import build_mlp
from core.train import mse_loss, train
from utils.artifacts import make_run_dir, save_json, save_npz, write_config_copy


# ------------------------------------------------------------
# Main experiment logic
# ------------------------------------------------------------
def run(config_path: str):
    cfg = yaml.safe_load(open(config_path))

    # Parse config
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
    eval_every = int(train_cfg["eval_every"])  # snapshot interval

    n_eval = int(data_cfg["n_eval"])
    M_train = int(data_cfg["M_train"])
    noise_std = float(data_cfg.get("noise_std", 0.0))

    depth_hidden = int(model_cfg["depth_hidden"])
    b_std = float(model_cfg["b_std"])

    # ------------------------------------------------------------
    # Create run directory
    # ------------------------------------------------------------
    save_dir = make_run_dir(exp_cfg.get("save_dir", "results"), exp_cfg["name"])
    write_config_copy(save_dir, cfg)

    print("=== NTK kernel drift experiment ===")
    print(f"Saving results to: {save_dir}\n")
    t_start = time.time()

    # ------------------------------------------------------------
    # Build Fourier target + dataset
    # ------------------------------------------------------------
    Ks = jnp.array(target_cfg["Ks"], dtype=jnp.float32)
    amps = jnp.array(target_cfg["amps"], dtype=jnp.float32)
    phases = jnp.array(target_cfg["phases"], dtype=jnp.float32)

    spec = make_fourier_target(Ks, amps, phases)

    # Eval grid
    gamma_eval = jnp.linspace(-jnp.pi, jnp.pi, n_eval, endpoint=False)
    X_eval = jnp.stack([jnp.cos(gamma_eval), jnp.sin(gamma_eval)], axis=1)
    y_eval_true = f_star_gamma(gamma_eval, spec)

    # Training set
    X_train, gamma_train, y_train = sample_train_on_circle(
        gamma_eval,
        X_eval,
        spec,
        M_train=M_train,
        key=jax.random.PRNGKey(exp_cfg["seed"]),
        add_noise_std=noise_std,
    )

    # Save dataset once
    data_file = "data_fourier.npz"
    save_npz(
        save_dir / data_file,
        gamma_eval=np.asarray(gamma_eval),
        X_eval=np.asarray(X_eval),
        y_eval_true=np.asarray(y_eval_true),
        gamma_train=np.asarray(gamma_train),
        X_train=np.asarray(X_train),
        y_train=np.asarray(y_train),
        Ks=np.asarray(Ks),
        amps=np.asarray(amps),
        phases=np.asarray(phases),
    )

    # ------------------------------------------------------------
    # Sweep over widths × seeds
    # ------------------------------------------------------------
    manifest = {}
    runs_dir = save_dir / "runs"
    runs_dir.mkdir(exist_ok=True)

    for w in widths:
        print(f"Width = {w}")

        # Storage
        K_tt_list = []  # (snapshot, M_train, M_train)
        K_et_list = []  # (snapshot, n_eval, M_train)
        kernel_meta = []  # (snapshot, 2) -> (seed, step)

        preds_list = []  # (snapshot, n_eval)
        preds_meta = []  # (snapshot, 2) -> (seed, step)

        loss_meta = []  # (seed, step)
        loss_values = []  # scalar losses

        for s in range(n_seeds):
            print(f"  Seed {s}")

            # Build model
            init_fn, apply_fn, _ = build_mlp(
                width=w,
                depth_hidden=depth_hidden,
                b_std=b_std,
            )
            _, params = init_fn(jax.random.PRNGKey(s), X_train.shape)

            # ----- Snapshot at t = 0 -----
            K_tt0 = empirical_ntk_matrix(apply_fn, params, X_train)
            K_et0 = empirical_ntk_matrix(apply_fn, params, X_eval, X_train)
            K_tt_list.append(np.asarray(K_tt0))
            K_et_list.append(np.asarray(K_et0))
            kernel_meta.append([s, 0])

            pred0 = np.asarray(apply_fn(params, X_eval).squeeze())
            preds_list.append(pred0)
            preds_meta.append([s, 0])

            # Initial loss
            loss0 = mse_loss(apply_fn, params, X_train, y_train)
            loss_meta.append([s, 0])
            loss_values.append(loss0)

            # ----- Training in chunks -----
            current_step = 0
            remaining = steps

            while remaining > 0:
                chunk = min(eval_every, remaining)

                # Train with return_history=True
                params, history = train(
                    params,
                    apply_fn,
                    X_train,
                    y_train,
                    steps=chunk,
                    lr=lr,
                    log_every=None,
                    return_history=True,
                )

                current_step += chunk
                remaining -= chunk

                last_loss = float(history[-1][1])

                loss_meta.append([s, current_step])
                loss_values.append(last_loss)

                # Snapshot after this chunk
                K_tt = empirical_ntk_matrix(apply_fn, params, X_train)
                K_et = empirical_ntk_matrix(apply_fn, params, X_eval, X_train)
                K_tt_list.append(np.asarray(K_tt))
                K_et_list.append(np.asarray(K_et))
                kernel_meta.append([s, current_step])

                pred_t = np.asarray(apply_fn(params, X_eval).squeeze())
                preds_list.append(pred_t)
                preds_meta.append([s, current_step])

                print(f"    step {current_step:7d} | loss={last_loss:.6e}")

        # Save all results for this width
        out_path = runs_dir / f"width_{w}.npz"
        save_npz(
            out_path,
            K_train_train=np.stack(K_tt_list),
            K_eval_train=np.stack(K_et_list),
            kernel_meta=np.asarray(kernel_meta),
            preds=np.stack(preds_list),
            preds_meta=np.asarray(preds_meta),
            loss_meta=np.asarray(loss_meta),
            loss_values=np.asarray(loss_values),
            X_train=np.asarray(X_train),
            y_train=np.asarray(y_train),
        )

        manifest[str(w)] = str(out_path.relative_to(save_dir))

    # ------------------------------------------------------------
    # Manifest
    # ------------------------------------------------------------
    summary = {
        "data_file": data_file,
        "runs": manifest,
        "meta": {
            "widths": widths,
            "seeds": n_seeds,
            "steps": steps,
            "lr": lr,
            "eval_every": eval_every,
            "depth_hidden": depth_hidden,
            "M_train": M_train,
            "n_eval": n_eval,
            "noise_std": noise_std,
        },
        "runtime_sec": round(time.time() - t_start, 2),
    }
    save_json(save_dir / "manifest.json", summary)

    print(f"\nDone. Kernel drift logs saved to {save_dir}")


# ------------------------------------------------------------
# Entry point
# ------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(
            "Usage: python3 -m experiments.ntk_kernel_drift "
            "configs/ntk_kernel_drift.yaml"
        )
    else:
        run(sys.argv[1])
