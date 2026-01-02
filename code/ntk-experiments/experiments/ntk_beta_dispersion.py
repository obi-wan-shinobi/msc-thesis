# ------------------------------------------------------------
# experiments/ntk_beta_dispersion.py
# ------------------------------------------------------------
"""
Experiment — NTK Dispersion under Explicit Width × Depth Grid
==============================================================

We compute:
    (1) NTK diagonal dispersion S2 at initialization
    (2) Optional NTK profiles across seeds

No training is performed.
"""

import sys
import time

import jax
import numpy as np
import yaml
from core.analysis import empirical_ntk_diag, empirical_profile, on_diag_moments
from core.data import make_probe_circle
from core.model import build_mlp
from jax import vmap
from utils.artifacts import make_run_dir, save_json, save_npz, write_config_copy


def run(config_path: str):

    # ------------------------------
    # Load config
    # ------------------------------
    cfg = yaml.safe_load(open(config_path))

    exp = cfg["experiment"]
    widths = cfg["sweep"]["widths"]
    depths = cfg["sweep"]["depths"]
    seeds = int(cfg["sweep"]["seeds"])

    b_std = cfg["model"]["b_std"]
    param = cfg["model"]["parameterization"]

    probe_m = cfg["data"]["probe_m"]
    n_profile = cfg["data"]["n_profile"]

    compute_profiles = cfg["analysis"]["compute_profiles"]

    # ------------------------------
    # Create run directory
    # ------------------------------
    save_dir = make_run_dir(exp["save_dir"], exp["name"])
    write_config_copy(save_dir, cfg)

    print("=== NTK Dispersion (Explicit Width × Depth Grid) ===")
    print(f"Saving to: {save_dir}\n")

    t0 = time.time()

    # ------------------------------
    # Probe sets
    # ------------------------------
    gamma, X_circle, x0 = make_probe_circle(n_profile)
    _, X_probe, _ = make_probe_circle(probe_m)

    save_npz(
        save_dir / "probe_geometry.npz",
        gamma=np.asarray(gamma),
        X_circle=np.asarray(X_circle),
        x0=np.asarray(x0),
        X_probe=np.asarray(X_probe),
    )

    # ------------------------------
    # Sweep widths × depths
    # ------------------------------
    results_manifest = {}

    for n in widths:
        for d in depths:

            print(f"[width={n:4d}, depth={d:4d}]")

            # Build model
            init_fn, apply_fn, kernel_fn = build_mlp(
                width=n,
                depth_hidden=d,
                b_std=b_std,
                parameterization=param,
            )

            # ------------------------------
            # A. Parallel compute diag NTK for all seeds
            # ------------------------------
            diag_list = []
            root_key = jax.random.PRNGKey(exp["seed"])

            for s in range(seeds):
                key_s = jax.random.fold_in(root_key, s)
                _, params0 = init_fn(key_s, (1, 2))
                diag_vals = empirical_ntk_diag(apply_fn, params0, X_probe)
                diag_list.append(np.asarray(diag_vals))

            diag_stack = np.stack(diag_list)  # (seeds, probe_m)

            mu, m2, S2 = on_diag_moments(diag_stack)

            # ------------------------------
            # B. Optional profiles
            # ------------------------------
            if compute_profiles:
                profile_list = []
                root_profile = jax.random.PRNGKey(exp["seed"] + 10_000)

                for s in range(seeds):
                    key_s = jax.random.fold_in(root_profile, s)
                    _, params0 = init_fn(key_s, (1, 2))
                    prof0 = empirical_profile(apply_fn, params0, X_circle, x0)
                    profile_list.append(np.asarray(prof0))

                profiles = np.stack(profile_list)
            else:
                profiles = None

            # ------------------------------
            # Save files
            # ------------------------------
            fname = f"width_{n}_depth_{d}.npz"
            subdir = save_dir / "results"
            subdir.mkdir(exist_ok=True)

            save_npz(
                subdir / fname,
                diag_init_all=diag_stack,
                diag_mu=mu,
                diag_m2=m2,
                diag_S2=S2,
                profiles=profiles,
                width=n,
                depth=d,
            )

            results_manifest[f"{n}-{d}"] = str((subdir / fname).relative_to(save_dir))

    # ------------------------------
    # Save manifest
    # ------------------------------
    summary = {
        "probe_geometry": "probe_geometry.npz",
        "results": results_manifest,
        "meta": {
            "widths": widths,
            "depths": depths,
            "seeds": seeds,
            "b_std": b_std,
            "parameterization": param,
            "probe_m": probe_m,
            "n_profile": n_profile,
        },
        "runtime_sec": round(time.time() - t0, 2),
    }
    save_json(save_dir / "manifest.json", summary)

    print("\n✓ Done.")
    print(f"Saved {len(results_manifest)} grid results to {save_dir}")


# ------------------------------
# Entry point
# ------------------------------
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(
            "Usage: python3 -m experiments.ntk_beta_dispersion configs/ntk_beta_dispersion.yaml"
        )
    else:
        run(sys.argv[1])
