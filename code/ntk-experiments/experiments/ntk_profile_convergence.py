# ------------------------------------------------------------
# experiments/ntk_profile_convergence.py
# ------------------------------------------------------------
"""
Experiment 1 — Reproduction of NTK paper
Convergence of the NTK profile Θ(x₀, x(γ)) as width → ∞.
Save-only version with timestamped results.
"""

import sys
import time

import jax
import numpy as np
import yaml
from core.analysis import empirical_profile
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
    n_seeds = int(cfg["sweep"]["seeds"])
    depth_hidden = cfg["model"]["depth_hidden"]
    b_std = cfg["model"]["b_std"]
    steps = cfg["train"]["steps"]
    lr = cfg["train"]["lr"]
    log_every = cfg["train"]["log_every"]

    # --- Create timestamped run directory ---
    base_dir = exp.get("save_dir", "results")
    exp_name = exp["name"]
    save_dir = make_run_dir(base_dir, exp_name)
    write_config_copy(save_dir, cfg)

    print("=== NTK profile convergence experiment ===")
    print(f"Saving artifacts to: {save_dir}\n")
    t0 = time.time()

    # ------------------------------------------------------------
    # 1. Probe manifold: unit circle
    # ------------------------------------------------------------
    gamma, X_circle, x0 = make_probe_circle(cfg["data"]["n_profile"])
    save_npz(
        save_dir / "probe_geometry.npz",
        gamma=np.asarray(gamma),
        X_circle=np.asarray(X_circle),
        x0=np.asarray(x0),
    )

    # ------------------------------------------------------------
    # 2. Gaussian regression task f*(x)=x₁·x₂ (for training)
    # ------------------------------------------------------------
    key = jax.random.PRNGKey(exp["seed"])
    Xtr, ytr = make_gaussian_regression_task(cfg["data"]["M_train"], key)
    save_npz(save_dir / "training_task.npz", Xtr=np.asarray(Xtr), ytr=np.asarray(ytr))

    # ------------------------------------------------------------
    # 3. Width × seed sweep — empirical NTK profiles
    # ------------------------------------------------------------
    results_manifest = {}
    for w in widths:
        print(f"Width {w}: averaging over {n_seeds} seeds ...")
        init_profiles, post_profiles = [], []

        init_fn, apply_fn, kernel_fn = build_mlp(
            width=w, b_std=b_std, depth_hidden=depth_hidden
        )

        for s in range(n_seeds):
            print(f"  Seed {s}")
            # Initialize network
            _, params0 = init_fn(jax.random.PRNGKey(s), (1, 2))

            # Profile at init (t=0)
            prof0 = empirical_profile(apply_fn, params0, X_circle, x0)
            init_profiles.append(np.asarray(prof0))

            # Train network and compute post-training profile
            paramsT = train(
                params0, apply_fn, Xtr, ytr, steps=steps, lr=lr, log_every=log_every
            )
            profT = empirical_profile(apply_fn, paramsT, X_circle, x0)
            post_profiles.append(np.asarray(profT))

        # Aggregate across seeds
        init_arr = np.stack(init_profiles)
        post_arr = np.stack(post_profiles)

        # Save per-width results
        width_file = save_dir / "profiles" / f"width_{w}.npz"
        width_file.parent.mkdir(exist_ok=True)
        save_npz(
            width_file,
            init_all=init_arr,
            post_all=post_arr,
            init_mean=init_arr.mean(0),
            init_std=init_arr.std(0),
            post_mean=post_arr.mean(0),
            post_std=post_arr.std(0),
        )
        results_manifest[str(w)] = str(width_file.relative_to(save_dir))

    # ------------------------------------------------------------
    # 4. Analytic (∞-width) NTK profile
    # ------------------------------------------------------------
    _, _, kernel_ref = build_mlp(
        width=widths[0], b_std=b_std, depth_hidden=depth_hidden
    )
    theta_inf = np.asarray(kernel_ref(x0, X_circle, get="ntk").squeeze())
    save_npz(save_dir / "analytic_profile.npz", theta_inf=theta_inf)

    # ------------------------------------------------------------
    # 5. Save manifest summary
    # ------------------------------------------------------------
    summary = {
        "probe_geometry": "probe_geometry.npz",
        "training_task": "training_task.npz",
        "analytic_profile": "analytic_profile.npz",
        "profiles": results_manifest,
        "meta": {
            "depth_hidden": depth_hidden,
            "b_std": b_std,
            "train_steps": steps,
            "lr": lr,
            "seeds": n_seeds,
            "widths": widths,
        },
        "runtime_sec": round(time.time() - t0, 2),
    }
    save_json(save_dir / "manifest.json", summary)

    print(f"\nDone. Saved {len(widths)} width profiles to {save_dir}")


# ------------------------------------------------------------
# Entry point
# ------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(
            "Usage: python3 -m experiments.ntk_profile_convergence configs/ntk_profile.yaml"
        )
    else:
        run(sys.argv[1])
