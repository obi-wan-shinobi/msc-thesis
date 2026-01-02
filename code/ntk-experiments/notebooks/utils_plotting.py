# ------------------------------------------------------------
# notebooks/utils_plotting.py
# ------------------------------------------------------------
"""
Convenience functions for loading NTK regression experiment results
and visualizing convergence behavior from saved artifacts.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedFormatter, FixedLocator
from utils.artifacts_io import load_manifest, npz


def print_run_config(run_path: str):
    man = load_manifest(run_path)

    # --- meta / sweep / training ---
    meta = man.get("meta", {})
    widths = meta.get("widths")
    seeds = meta.get("seeds")
    lr = meta.get("lr")
    steps = meta.get("train_steps")
    reg = meta.get("reg")
    depth = meta.get("depth_hidden")
    b_std = meta.get("b_std")
    task = meta.get("task_type", "unknown")

    # --- dataset ---
    ds_rel = man.get("dataset")
    ds = npz(run_path, ds_rel)

    # common dataset fields (both simple & fourier configs store these)
    n_eval = len(ds["gamma_eval"]) if "gamma_eval" in ds else None
    m_train = len(ds["y_train"]) if "y_train" in ds else None
    noise_std = ds.get("noise_std", None)  # only saved for Fourier if you included it

    # --- fourier-specific (if present) ---
    Ks = ds.get("Ks", None)
    amps = ds.get("amps", None)
    phases = ds.get("phases", None)

    # --- analytic predictor info ---
    analytic_rel = man.get("analytic_predictor")
    reg_from_analytic = None
    if analytic_rel:
        analytic = npz(run_path, analytic_rel)
        reg_from_analytic = analytic.get("reg", None)

    print("─────────────────────────────────────────")
    print("Run:", run_path)
    print("Task:", task)
    print("Dataset file:", ds_rel)
    print(" ")
    print("Data:")
    print(f"  n_eval:   {n_eval}")
    print(f"  M_train:  {m_train}")
    if noise_std is not None:
        print(f"  noise_std: {float(noise_std):.4g}")
    if Ks is not None:
        # print Fourier spectrum compactly
        print("Fourier target:")

        print("  Ks:    ", np.asarray(Ks).astype(float).tolist())
        print("  amps:  ", [float(a) for a in np.asarray(amps)])
        print("  phases:", [float(p) for p in np.asarray(phases)])

    print(" ")
    print("Model:")
    print(f"  depth_hidden: {depth}")
    print(f"  b_std:        {b_std}")

    print(" ")
    print("Training:")
    print(f"  steps: {steps}")
    print(f"  lr:    {lr}")
    print(f"  seeds: {seeds}")

    print(" ")
    print("Sweep:")
    print(f"  widths: {widths}")

    print(" ")
    print("Analysis:")
    print(f"  reg (config): {reg}")
    if reg_from_analytic is not None:
        print(f"  reg (artifact): {reg_from_analytic}")
    print("─────────────────────────────────────────")


def plot_regression_convergence(
    run_path: str, show_predictions: bool = True, use_logx: bool = True
):
    """
    Load results from a finished NTK regression experiment and plot:
      (1) RelErr vs width
      (2) Finite-width vs analytic NTK predictions
    """
    # --- Load manifest ---
    man = load_manifest(run_path)

    # --- Load dataset and analytic NTK predictor ---
    data = npz(run_path, man["dataset"])
    gamma_eval = data["gamma_eval"]
    gamma_train = data["gamma_train"]
    X_eval = data["X_eval"]
    y_eval_true = data["y_eval_true"]
    X_train = data["X_train"]
    y_train = data["y_train"]

    analytic = npz(run_path, man["analytic_predictor"])
    y_inf = analytic["y_inf"]

    # --- Load predictions from each width ---
    predictions = {}
    errs_summary = []
    for w_str, rel_path in man["predictions"].items():
        w = int(w_str)
        pred = npz(run_path, rel_path)
        preds_all = pred["preds_all"]
        err_all = pred["err_all"]

        predictions[w] = pred
        mean_err = float(err_all.mean())
        p10, p50, p90 = np.percentile(err_all, [10, 50, 90])
        errs_summary.append((w, mean_err, p10, p50, p90))

    # --- Convert to arrays for plotting ---
    errs_summary = np.array(sorted(errs_summary, key=lambda x: x[0]))
    widths = errs_summary[:, 0].astype(int)
    merr, p10, p50, p90 = errs_summary[:, 1:5].T

    pos = np.arange(len(widths))

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    if use_logx:
        ax.loglog(widths, merr, marker="o", label="mean RelErr")
        ax.loglog(widths, p50, marker="s", label="median RelErr")
        ax.fill_between(
            widths, p10, p90, alpha=0.2, label="[p10, p90]", edgecolor="none"
        )
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(FixedLocator(widths))
        ax.xaxis.set_major_formatter(FixedFormatter([str(w) for w in widths]))
    else:
        ax.semilogy(pos, merr, marker="o", label="mean RelErr")
        ax.semilogy(pos, p50, marker="s", label="median RelErr")
        ax.fill_between(pos, p10, p90, alpha=0.2, label="[p10, p90]", edgecolor="none")
        ax.set_xticks(pos)
        ax.set_xticklabels([str(w) for w in widths])

    ax.set_xlabel("width")
    ax.set_ylabel("RelErr (finite vs NTK)")
    ax.set_title("Convergence to NTK predictor (function space)")
    ax.grid(True, which="both", ls="--", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    plt.show()

    # ============================================================
    # 2. Overlay predictions
    # ============================================================
    if show_predictions:
        plt.figure(figsize=(9, 3.5))
        plt.plot(
            gamma_eval, y_eval_true, color="gray", alpha=0.6, label="f*(γ) on circle"
        )
        plt.plot(gamma_eval, y_inf, lw=2.2, label="NTK ∞-width")

        for w in sorted(predictions.keys()):
            s = predictions[w]
            plt.plot(gamma_eval, s["pred_mean"], "--", lw=1.4, label=f"finite (w={w})")

        plt.xlabel(r"angle $\gamma$")
        plt.ylabel("prediction")
        plt.title("Finite nets (mean across seeds) and analytic NTK")
        plt.legend(ncol=2, fontsize=9)
        plt.grid(True, ls="--", alpha=0.4)
        plt.tight_layout()
        plt.show()

    # --- Return everything for further analysis if needed ---
    return {
        "widths": widths,
        "errors": errs_summary,
        "predictions": predictions,
        "y_inf": y_inf,
        "gamma_eval": gamma_eval,
        "y_eval_true": y_eval_true,
    }

def aggregate_over_seeds(property_meta, property_values,
                         q_low=10, q_high=90):
    """
    Generic aggregator over seeds for any property recorded as:
        property_meta   : (T_snap, 2)  -> (seed, step)
        property_values : (T_snap, ...) values at each snapshot

    Returns:
        steps_unique : (T_steps,)
        mean_vals    : (T_steps, ...)
        lo_vals      : (T_steps, ...)
        hi_vals      : (T_steps, ...)
    """
    meta = np.asarray(property_meta)
    vals = np.asarray(property_values)

    steps_unique = np.sort(np.unique(meta[:, 1]))

    mean_list = []
    lo_list   = []
    hi_list   = []

    for step in steps_unique:
        mask = (meta[:, 1] == step)
        batch_vals = vals[mask]             # shape: (n_seeds, ...)

        mu = batch_vals.mean(axis=0)
        lo, hi = np.percentile(batch_vals, [q_low, q_high], axis=0)

        mean_list.append(mu)
        lo_list.append(lo)
        hi_list.append(hi)

    return (
        steps_unique,
        np.array(mean_list),
        np.array(lo_list),
        np.array(hi_list),
    )
