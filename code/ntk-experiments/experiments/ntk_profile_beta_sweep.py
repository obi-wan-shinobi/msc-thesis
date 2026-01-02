# ------------------------------------------------------------
# experiments/ntk_profile_beta_sweep.py
# ------------------------------------------------------------
"""
Experiment — NTK profile convergence under varying depth/width ratio (β = d/n).
Extends the NTK profile experiment by logging:
  (1) empirical NTK profiles pre/post training
  (2) on-diagonal NTK moments at initialization
  (3) one-step kernel updates ΔK / K at initialization
All results are saved as NPZ artifacts for later analysis.
"""

import sys
import time

import jax
import numpy as np
import yaml
from core.analysis import (
    delta_ntk_single_update,
    empirical_ntk_diag,
    empirical_profile,
    on_diag_moments,
)
from core.data import make_gaussian_regression_task, make_probe_circle
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
    depths = cfg["sweep"]["depths"]
    n_seeds = int(cfg["sweep"]["seeds"])
    b_std = cfg["model"]["b_std"]
    steps = cfg["train"]["steps"]
    lr = cfg["train"]["lr"]
    log_every = cfg["train"]["log_every"]

    probe_m = cfg["data"].get("probe_m", 128)
    n_update = cfg["analysis"].get("n_update_samples", 64)

    # --- Create timestamped run directory ---
    base_dir = exp.get("save_dir", "results")
    exp_name = exp["name"]
    save_dir = make_run_dir(base_dir, exp_name)
    write_config_copy(save_dir, cfg)

    print("=== NTK β-sweep (depth/width ratio) experiment ===")
    print(f"Saving artifacts to: {save_dir}\n")
    t0 = time.time()

    # ------------------------------------------------------------
    # 1. Probe manifolds
    # ------------------------------------------------------------
    gamma, X_circle, x0 = make_probe_circle(cfg["data"]["n_profile"])
    gamma_probe, X_probe, _ = make_probe_circle(probe_m)
    save_npz(
        save_dir / "probe_geometry.npz",
        gamma=np.asarray(gamma),
        X_circle=np.asarray(X_circle),
        x0=np.asarray(x0),
    )
    save_npz(
        save_dir / "probe_small.npz",
        gamma_probe=np.asarray(gamma_probe),
        X_probe=np.asarray(X_probe),
    )

    # ------------------------------------------------------------
    # 2. Gaussian regression task f*(x)=x₁·x₂ (for training)
    # ------------------------------------------------------------
    key = jax.random.PRNGKey(exp["seed"])
    Xtr, ytr = make_gaussian_regression_task(cfg["data"]["M_train"], key)
    save_npz(save_dir / "training_task.npz", Xtr=np.asarray(Xtr), ytr=np.asarray(ytr))

    # ------------------------------------------------------------
    # 3. Sweep over width × depth × seeds
    # ------------------------------------------------------------
    results_manifest = {}

    for w in widths:
        for d in depths:
            print(f"\nWidth {w}, Depth {d}: averaging over {n_seeds} seeds ...")

            init_profiles, post_profiles = [], []
            diag_stack = []
            rel_updates, abs_updates = [], []

            init_fn, apply_fn, kernel_fn = build_mlp(
                width=w, b_std=b_std, depth_hidden=d, parameterization="standard"
            )

            # --- (a) On-diagonal moments at initialization ---
            for s in range(n_seeds):
                _, params0 = init_fn(jax.random.PRNGKey(s), (1, 2))
                diag_vals = empirical_ntk_diag(apply_fn, params0, X_probe)
                diag_stack.append(np.asarray(diag_vals))
            diag_stack = np.stack(diag_stack)
            mu, m2, S2 = on_diag_moments(diag_stack)

            # # --- (b) One-step ΔK / K update stats ---
            # for s in range(n_seeds):
            #     _, params0 = init_fn(jax.random.PRNGKey(10_000 + s), (1, 2))
            #     idx = jax.random.choice(
            #         jax.random.PRNGKey(20_000 + s),
            #         Xtr.shape[0],
            #         (n_update,),
            #         replace=False,
            #     )
            #     for j in np.asarray(idx):
            #         xj, yj = Xtr[j], ytr[j]
            #         K0_xx, K1_xx, dK, rel = delta_ntk_single_update(
            #             params0, apply_fn, xj, yj, lr=lr
            #         )
            #         rel_updates.append(rel)
            #         abs_updates.append(dK)
            # rel_updates = np.asarray(rel_updates)
            # abs_updates = np.asarray(abs_updates)
            #
            # # --- (c) Full NTK profile before/after training ---
            # for s in range(n_seeds):
            #     _, params0 = init_fn(jax.random.PRNGKey(100 + s), (1, 2))
            #     prof0 = empirical_profile(apply_fn, params0, X_circle, x0)
            #     paramsT = train(
            #         params0, apply_fn, Xtr, ytr, steps=steps, lr=lr, log_every=log_every
            #     )
            #     profT = empirical_profile(apply_fn, paramsT, X_circle, x0)
            #     init_profiles.append(np.asarray(prof0))
            #     post_profiles.append(np.asarray(profT))
            #
            # init_arr = np.stack(init_profiles)
            # post_arr = np.stack(post_profiles)

            # Save per (width, depth)
            subdir = save_dir / "profiles"
            subdir.mkdir(exist_ok=True)
            file_name = f"width_{w}_depth_{d}.npz"
            save_npz(
                subdir / file_name,
                # init_all=init_arr,
                # post_all=post_arr,
                # init_mean=init_arr.mean(0),
                # init_std=init_arr.std(0),
                # post_mean=post_arr.mean(0),
                # post_std=post_arr.std(0),
                diag_init_all=diag_stack,
                diag_mu=mu,
                diag_m2=m2,
                diag_S2=S2,
                deltaK_rel_all=rel_updates,
                deltaK_abs_all=abs_updates,
                beta=float(d / w),
                width=w,
                depth=d,
            )
            results_manifest[f"{w}-{d}"] = str(
                (subdir / file_name).relative_to(save_dir)
            )

    # ------------------------------------------------------------
    # 5. Save manifest summary
    # ------------------------------------------------------------
    summary = {
        "probe_geometry": "probe_geometry.npz",
        "training_task": "training_task.npz",
        "analytic_profile": "analytic_profile.npz",
        "profiles": results_manifest,
        "meta": {
            "b_std": b_std,
            "train_steps": steps,
            "lr": lr,
            "seeds": n_seeds,
            "widths": widths,
            "depths": depths,
            "probe_m": probe_m,
            "n_update_samples": n_update,
            "input_dim": 2,
        },
        "runtime_sec": round(time.time() - t0, 2),
    }
    save_json(save_dir / "manifest.json", summary)

    print(
        f"\n✓ Done. Saved {len(widths) * len(depths)} (width,depth) profiles to {save_dir}"
    )


# ------------------------------------------------------------
# Entry point
# ------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(
            "Usage: python3 -m experiments.ntk_profile_beta_sweep configs/ntk_profile_beta.yaml"
        )
    else:
        run(sys.argv[1])
