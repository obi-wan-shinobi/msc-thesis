from pathlib import Path

import jax
import numpy as np
from core.analysis import relerr
from core.model import build_mlp
from core.train import train


def run_width_sweep(
    widths, seeds, data_fn, target_fn, model_cfg, train_cfg, eval_fn, save_dir: Path
):
    """
    Generic loop for width × seed sweeps.
    Returns: stats[width] = {...}, errs_summary
    """
    stats, errs_summary = {}, []
    for w in widths:
        preds, errs = [], []
        print(f"\n=== width {w} ===")
        init_fn, apply_fn, _ = build_mlp(width=w, **model_cfg)
        for s in seeds:
            print(f"---- Seed {s} ---- ")
            _, params = init_fn(jax.random.PRNGKey(s), data_fn["X_train"].shape)
            params = train(
                params, apply_fn, data_fn["X_train"], data_fn["y_train"], **train_cfg
            )
            y_pred = eval_fn(apply_fn, params, data_fn["X_eval"]).squeeze()
            preds.append(np.asarray(y_pred))
            errs.append(relerr(y_pred, target_fn))
        preds = np.stack(preds)
        errs = np.asarray(errs)
        stats[w] = {
            "preds": preds,
            "pred_mean": preds.mean(0),
            "pred_p10": np.percentile(preds, 10, 0),
            "pred_p50": np.percentile(preds, 50, 0),
            "pred_p90": np.percentile(preds, 90, 0),
            "err_all": errs,
        }
        p10, p50, p90 = np.percentile(errs, [10, 50, 90])
        errs_summary.append((w, float(errs.mean()), p10, p50, p90))
    return stats, np.array(errs_summary)
