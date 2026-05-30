# ------------------------------------------------------------
# experiments/ntk_spectral_bias_architecture_comparison.py
# ------------------------------------------------------------
"""
Compare spectral-bias fitting dynamics across overparameterized architectures.

Architectures:
    - Single-layer MLP
    - Deep 3-layer MLP
    - Single attention block

For each model (and seed), full-batch SGD is run for a short window.
At snapshot times we log:
    - predicted function on evaluation grid,
    - residual function on evaluation grid,
    - residual projections onto target modes,
    - residual l2 mass per tracked frequency plane.
"""

import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
import yaml
from jax import jit, value_and_grad
from neural_tangents import stax

from core.fourier_basis_circle import build_real_fourier_basis
from core.model import build_mlp
from utils.artifacts import make_run_dir, save_json, save_npz, write_config_copy

TWO_PI = 2.0 * jnp.pi


def _resolve_seed_list(seed_cfg, base_seed: int) -> list[int]:
    if isinstance(seed_cfg, int):
        return [base_seed + s for s in range(seed_cfg)]
    return [int(s) for s in seed_cfg]


def _snapshot_steps(steps: int, eval_every: int) -> np.ndarray:
    if steps < 0:
        raise ValueError(f"train.steps must be nonnegative, got {steps}.")
    if eval_every <= 0:
        raise ValueError(f"train.eval_every must be positive, got {eval_every}.")

    snaps = list(range(0, steps + 1, eval_every))
    if snaps[-1] != steps:
        snaps.append(steps)
    return np.asarray(snaps, dtype=np.int32)


def _build_circle_grid(n_points: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    gamma = jnp.linspace(0.0, TWO_PI, num=n_points, endpoint=False)
    X = jnp.stack([jnp.cos(gamma), jnp.sin(gamma)], axis=1)
    return gamma, X


def _parse_target_terms(target_cfg: dict) -> list[dict]:
    terms = target_cfg.get("terms", None)
    if terms is None:
        raise ValueError("target.terms is required.")
    if len(terms) == 0:
        raise ValueError("target.terms must not be empty.")

    parsed = []
    freqs_seen = set()
    amps = []

    for item in terms:
        freq = int(item["freq"])
        amp = float(item["amp"])
        basis = str(item["basis"]).lower()

        if freq < 0:
            raise ValueError(f"target term frequency must be nonnegative, got {freq}.")
        if basis not in {"cos", "sin"}:
            raise ValueError(
                f"target term basis must be 'cos' or 'sin', got {basis}."
            )
        if freq in freqs_seen:
            raise ValueError(
                "target frequencies must be unique in target.terms for this experiment. "
                f"Found duplicate freq={freq}."
            )

        freqs_seen.add(freq)
        amps.append(amp)
        parsed.append({"freq": freq, "amp": amp, "basis": basis})

    amps = np.asarray(amps, dtype=np.float64)
    if np.any(np.diff(amps) > 0.0):
        raise ValueError("target.terms amplitudes must be in decreasing order.")

    return parsed


def _evaluate_target(gamma: jnp.ndarray, target_terms: list[dict]) -> jnp.ndarray:
    y = jnp.zeros_like(gamma)
    for term in target_terms:
        k = term["freq"]
        a = term["amp"]
        basis = term["basis"]

        if basis == "cos":
            y = y + a * jnp.cos(k * gamma)
        else:
            y = y + a * jnp.sin(k * gamma)

    return y


def _attention_input_adapter(X: jnp.ndarray) -> jnp.ndarray:
    # Convert (batch, 2) coordinates into a length-2 token sequence:
    # token 0 = x-coordinate, token 1 = y-coordinate.
    return X[:, :, None]


def _build_model(model_cfg: dict):
    model_type = str(model_cfg["type"]).lower()
    model_name = str(model_cfg["name"])

    if model_type == "mlp":
        width = int(model_cfg["width"])
        depth_hidden = int(model_cfg["depth_hidden"])
        b_std = float(model_cfg.get("b_std", 1.0))
        parameterization = str(model_cfg.get("parameterization", "ntk"))

        init_fn, apply_fn, _ = build_mlp(
            width=width,
            depth_hidden=depth_hidden,
            b_std=b_std,
            parameterization=parameterization,
        )

        resolved_cfg = {
            "name": model_name,
            "type": "mlp",
            "width": width,
            "depth_hidden": depth_hidden,
            "b_std": b_std,
            "parameterization": parameterization,
        }
        return init_fn, apply_fn, lambda X: X, resolved_cfg

    if model_type == "attention":
        d_model = int(model_cfg["d_model"])
        n_heads = int(model_cfg["n_heads"])
        d_key = int(model_cfg.get("d_key", max(1, d_model // max(1, n_heads))))
        n_chan_val = int(model_cfg.get("n_chan_val", d_model))
        b_std = float(model_cfg.get("b_std", 1.0))
        parameterization = str(model_cfg.get("parameterization", "ntk"))
        linear_scaling = bool(model_cfg.get("linear_scaling", True))
        attention_mechanism = str(model_cfg.get("attention_mechanism", "SOFTMAX"))

        layers = [
            stax.Dense(d_model, b_std=b_std, parameterization=parameterization),
            stax.GlobalSelfAttention(
                n_chan_out=d_model,
                n_chan_key=d_key,
                n_chan_val=n_chan_val,
                n_heads=n_heads,
                linear_scaling=linear_scaling,
                b_std=b_std,
                attention_mechanism=attention_mechanism,
            ),
            stax.Relu(),
            stax.Flatten(),
            stax.Dense(1, b_std=None, parameterization=parameterization),
        ]
        init_fn, apply_fn, _ = stax.serial(*layers)
        apply_fn = jit(apply_fn)

        resolved_cfg = {
            "name": model_name,
            "type": "attention",
            "d_model": d_model,
            "n_heads": n_heads,
            "d_key": d_key,
            "n_chan_val": n_chan_val,
            "b_std": b_std,
            "parameterization": parameterization,
            "linear_scaling": linear_scaling,
            "attention_mechanism": attention_mechanism,
        }
        return init_fn, apply_fn, _attention_input_adapter, resolved_cfg

    raise ValueError(
        f"Unknown model type '{model_type}' for model '{model_name}'. "
        "Use 'mlp' or 'attention'."
    )


def _count_params(params) -> int:
    leaves = jax.tree_util.tree_leaves(params)
    return int(sum(np.asarray(leaf).size for leaf in leaves))


def _build_mode_bank(
    gamma_eval: jnp.ndarray,
    target_terms: list[dict],
) -> dict:
    max_k = max(int(t["freq"]) for t in target_terms)
    basis = build_real_fourier_basis(gamma_eval, K_max=max_k)

    Phi_unit = np.asarray(basis["Phi_unit"], dtype=np.float32)
    mode_names = np.asarray(basis["mode_names"])
    mode_freqs = np.asarray(basis["mode_freqs"], dtype=np.int32)
    mode_types = np.asarray(basis["mode_types"])

    tracked_idx = []
    tracked_names = []
    tracked_freqs = []
    tracked_types = []
    tracked_amps = []

    for term in target_terms:
        k = int(term["freq"])
        t = str(term["basis"])
        idx = np.where((mode_freqs == k) & (mode_types == t))[0]
        if len(idx) != 1:
            raise ValueError(f"Could not find unique basis mode for freq={k}, type={t}.")
        j = int(idx[0])

        tracked_idx.append(j)
        tracked_names.append(str(mode_names[j]))
        tracked_freqs.append(k)
        tracked_types.append(t)
        tracked_amps.append(float(term["amp"]))

    unique_freqs = []
    for term in target_terms:
        k = int(term["freq"])
        if k not in unique_freqs:
            unique_freqs.append(k)

    freq_plane_indices = []
    for k in unique_freqs:
        idx = np.where(mode_freqs == k)[0]
        expected_dim = 1 if k == 0 else 2
        if len(idx) != expected_dim:
            raise ValueError(
                f"Expected {expected_dim} basis modes for frequency {k}, got {len(idx)}."
            )
        freq_plane_indices.append(np.asarray(idx, dtype=np.int32))

    return {
        "Phi_unit": Phi_unit,
        "tracked_mode_idx": np.asarray(tracked_idx, dtype=np.int32),
        "tracked_mode_names": np.asarray(tracked_names),
        "tracked_mode_freqs": np.asarray(tracked_freqs, dtype=np.int32),
        "tracked_mode_types": np.asarray(tracked_types),
        "tracked_mode_amps": np.asarray(tracked_amps, dtype=np.float32),
        "freq_plane_freqs": np.asarray(unique_freqs, dtype=np.int32),
        "freq_plane_indices": freq_plane_indices,
    }


def _train_with_snapshots(
    params0,
    apply_fn,
    X_train,
    y_train,
    X_eval,
    y_eval_true,
    target_mode_matrix_eval: np.ndarray,
    freq_plane_mats_eval: list[np.ndarray],
    use_train_as_eval: bool,
    steps: int,
    lr: float,
    eval_every: int,
    log_every: int | None,
):
    snapshot_steps = _snapshot_steps(steps, eval_every)
    n_snap = len(snapshot_steps)
    n_eval = int(X_eval.shape[0])
    n_mode = int(target_mode_matrix_eval.shape[1])
    n_freq = len(freq_plane_mats_eval)

    pred_eval = np.full((n_snap, n_eval), np.nan, dtype=np.float32)
    residual_eval = np.full((n_snap, n_eval), np.nan, dtype=np.float32)
    target_mode_coeff_eval = np.full((n_snap, n_mode), np.nan, dtype=np.float32)
    target_mode_abs_eval = np.full((n_snap, n_mode), np.nan, dtype=np.float32)
    freq_plane_l2_eval = np.full((n_snap, n_freq), np.nan, dtype=np.float32)

    train_loss = np.full((n_snap,), np.nan, dtype=np.float32)
    train_mse = np.full((n_snap,), np.nan, dtype=np.float32)
    eval_mse = np.full((n_snap,), np.nan, dtype=np.float32)

    y_train_np = np.asarray(y_train, dtype=np.float32)
    y_eval_np = np.asarray(y_eval_true, dtype=np.float32)
    n_train = y_train_np.shape[0]

    opt = optax.sgd(lr)
    opt_state = opt.init(params0)

    @jit
    def step_fn(params, state):
        def loss_fn(p):
            residual = apply_fn(p, X_train).squeeze() - y_train
            return 0.5 * jnp.sum(residual**2)

        loss, grads = value_and_grad(loss_fn)(params)
        updates, state = opt.update(grads, state, params)
        params = optax.apply_updates(params, updates)
        return params, state, loss

    _ = step_fn(params0, opt_state)

    params = params0
    current_step = 0
    snap_idx = 0

    while True:
        pred_train_now = np.asarray(apply_fn(params, X_train).squeeze(), dtype=np.float32)
        residual_train_now = pred_train_now - y_train_np
        sq_norm_train = np.sum(residual_train_now**2)
        train_loss[snap_idx] = 0.5 * sq_norm_train
        train_mse[snap_idx] = sq_norm_train / float(n_train)

        if use_train_as_eval:
            pred_eval_now = pred_train_now
            residual_eval_now = residual_train_now
        else:
            pred_eval_now = np.asarray(apply_fn(params, X_eval).squeeze(), dtype=np.float32)
            residual_eval_now = pred_eval_now - y_eval_np

        pred_eval[snap_idx, :] = pred_eval_now
        residual_eval[snap_idx, :] = residual_eval_now
        eval_mse[snap_idx] = np.mean(residual_eval_now**2)

        coeff = residual_eval_now @ target_mode_matrix_eval
        target_mode_coeff_eval[snap_idx, :] = coeff
        target_mode_abs_eval[snap_idx, :] = np.abs(coeff)

        for f_idx, plane in enumerate(freq_plane_mats_eval):
            c = residual_eval_now @ plane
            freq_plane_l2_eval[snap_idx, f_idx] = float(np.linalg.norm(c))

        if current_step >= steps:
            break

        chunk = min(eval_every, steps - current_step)
        last_loss = np.nan
        for _ in range(chunk):
            params, opt_state, loss = step_fn(params, opt_state)
            last_loss = float(loss)

        current_step += chunk
        snap_idx += 1

        if log_every and ((current_step % log_every == 0) or (current_step == steps)):
            print(
                f"      [train] step {current_step:5d} | "
                f"train_ls={last_loss:.6e}",
                flush=True,
            )

    if snap_idx != n_snap - 1:
        raise RuntimeError(
            "Unexpected number of snapshots during training. "
            f"Expected {n_snap}, got {snap_idx + 1}."
        )

    return {
        "snapshot_steps": snapshot_steps,
        "pred_eval": pred_eval,
        "residual_eval": residual_eval,
        "target_mode_coeff_eval": target_mode_coeff_eval,
        "target_mode_abs_eval": target_mode_abs_eval,
        "freq_plane_l2_eval": freq_plane_l2_eval,
        "train_loss": train_loss,
        "train_mse": train_mse,
        "eval_mse": eval_mse,
    }


def run(config_path: str):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    exp_cfg = cfg["experiment"]
    data_cfg = cfg["data"]
    target_cfg = cfg["target"]
    train_cfg = cfg["train"]
    sweep_cfg = cfg["sweep"]
    model_cfgs = cfg["models"]
    analysis_cfg = cfg.get("analysis", {})

    base_seed = int(exp_cfg.get("seed", 0))
    seed_list = _resolve_seed_list(sweep_cfg["seeds"], base_seed)

    n_train = int(data_cfg["n_train"])
    n_eval = int(data_cfg["n_eval"])
    train_grid = str(data_cfg.get("train_grid", "equally_spaced"))

    if train_grid != "equally_spaced":
        raise ValueError(
            f"data.train_grid='{train_grid}' is not supported; use 'equally_spaced'."
        )

    target_terms = _parse_target_terms(target_cfg)

    probe_domain = str(analysis_cfg.get("probe_domain", "eval")).lower()
    if probe_domain not in {"train", "eval"}:
        raise ValueError(
            f"analysis.probe_domain must be 'train' or 'eval', got '{probe_domain}'."
        )

    steps = int(train_cfg["steps"])
    lr = float(train_cfg["lr"])
    eval_every = int(train_cfg["eval_every"])
    log_every = train_cfg.get("log_every", None)
    log_every = int(log_every) if log_every is not None else None

    snapshot_steps = _snapshot_steps(steps, eval_every)
    n_snap = len(snapshot_steps)

    gamma_train, X_train = _build_circle_grid(n_train)
    gamma_eval, X_eval = _build_circle_grid(n_eval)

    y_train = _evaluate_target(gamma_train, target_terms)
    y_eval_true = _evaluate_target(gamma_eval, target_terms)

    if probe_domain == "train":
        gamma_probe = gamma_train
        X_probe = X_train
        y_probe_true = y_train
        use_train_as_eval = True
    else:
        gamma_probe = gamma_eval
        X_probe = X_eval
        y_probe_true = y_eval_true
        use_train_as_eval = False

    mode_bank = _build_mode_bank(gamma_probe, target_terms)
    Phi_eval = mode_bank["Phi_unit"]
    target_mode_matrix_eval = Phi_eval[:, mode_bank["tracked_mode_idx"]]

    freq_plane_mats_eval = [
        Phi_eval[:, idx] for idx in mode_bank["freq_plane_indices"]
    ]

    save_dir = make_run_dir(exp_cfg.get("save_dir", "results"), exp_cfg["name"])
    write_config_copy(save_dir, cfg)

    print("=== Spectral Bias Architecture Comparison ===")
    print(f"Saving results to: {save_dir}")
    print(f"seeds={seed_list}")
    print(f"steps={steps}, lr={lr}, eval_every={eval_every}, snapshots={n_snap}")
    print(f"probe_domain={probe_domain}")
    print("target terms:", target_terms)
    print()
    t0 = time.time()

    save_npz(
        save_dir / "probe_geometry.npz",
        gamma_train=np.asarray(gamma_train),
        X_train=np.asarray(X_train),
        gamma_eval=np.asarray(gamma_eval),
        X_eval=np.asarray(X_eval),
        gamma_probe=np.asarray(gamma_probe),
        X_probe=np.asarray(X_probe),
    )

    save_npz(
        save_dir / "target_task.npz",
        y_train=np.asarray(y_train),
        y_eval_true=np.asarray(y_eval_true),
        y_probe_true=np.asarray(y_probe_true),
        target_freqs=np.asarray([t["freq"] for t in target_terms], dtype=np.int32),
        target_amps=np.asarray([t["amp"] for t in target_terms], dtype=np.float32),
        target_types=np.asarray([t["basis"] for t in target_terms]),
    )

    save_npz(
        save_dir / "mode_bank.npz",
        tracked_mode_names=mode_bank["tracked_mode_names"],
        tracked_mode_freqs=mode_bank["tracked_mode_freqs"],
        tracked_mode_types=mode_bank["tracked_mode_types"],
        tracked_mode_amps=mode_bank["tracked_mode_amps"],
        freq_plane_freqs=mode_bank["freq_plane_freqs"],
    )

    runs_dir = save_dir / "runs"
    runs_dir.mkdir(exist_ok=True)

    n_seed = len(seed_list)
    n_mode = target_mode_matrix_eval.shape[1]
    n_freq = len(freq_plane_mats_eval)
    n_probe = int(X_probe.shape[0])

    runs_manifest = {}
    resolved_model_cfgs = []

    for model_cfg in model_cfgs:
        model_name = str(model_cfg["name"])
        model_lr_scale = float(model_cfg.get("lr_scale", 1.0))
        model_lr = lr * model_lr_scale
        print(f"Model: {model_name}")
        print(f"  lr={model_lr:.3e} (scale={model_lr_scale:.3g})")

        init_fn, apply_fn, input_adapter, resolved_model_cfg = _build_model(model_cfg)
        resolved_model_cfg["lr"] = model_lr
        resolved_model_cfg["lr_scale"] = model_lr_scale
        resolved_model_cfgs.append(resolved_model_cfg)

        X_train_model = input_adapter(X_train)
        X_eval_model = input_adapter(X_probe)

        pred_eval_all = np.full((n_seed, n_snap, n_probe), np.nan, dtype=np.float32)
        residual_eval_all = np.full((n_seed, n_snap, n_probe), np.nan, dtype=np.float32)
        target_mode_coeff_all = np.full(
            (n_seed, n_snap, n_mode),
            np.nan,
            dtype=np.float32,
        )
        target_mode_abs_all = np.full((n_seed, n_snap, n_mode), np.nan, dtype=np.float32)
        freq_plane_l2_all = np.full((n_seed, n_snap, n_freq), np.nan, dtype=np.float32)

        train_loss_all = np.full((n_seed, n_snap), np.nan, dtype=np.float32)
        train_mse_all = np.full((n_seed, n_snap), np.nan, dtype=np.float32)
        eval_mse_all = np.full((n_seed, n_snap), np.nan, dtype=np.float32)
        param_count_all = np.full((n_seed,), -1, dtype=np.int64)

        for s_idx, seed in enumerate(seed_list):
            print(f"  Seed {seed}")
            _, params0 = init_fn(jax.random.PRNGKey(seed), X_train_model.shape)
            param_count_all[s_idx] = _count_params(params0)
            print(f"    params={int(param_count_all[s_idx])}")

            out = _train_with_snapshots(
                params0=params0,
                apply_fn=apply_fn,
                X_train=X_train_model,
                y_train=y_train,
                X_eval=X_eval_model,
                y_eval_true=y_probe_true,
                target_mode_matrix_eval=target_mode_matrix_eval,
                freq_plane_mats_eval=freq_plane_mats_eval,
                use_train_as_eval=use_train_as_eval,
                steps=steps,
                lr=model_lr,
                eval_every=eval_every,
                log_every=log_every,
            )

            pred_eval_all[s_idx, :, :] = out["pred_eval"]
            residual_eval_all[s_idx, :, :] = out["residual_eval"]
            target_mode_coeff_all[s_idx, :, :] = out["target_mode_coeff_eval"]
            target_mode_abs_all[s_idx, :, :] = out["target_mode_abs_eval"]
            freq_plane_l2_all[s_idx, :, :] = out["freq_plane_l2_eval"]
            train_loss_all[s_idx, :] = out["train_loss"]
            train_mse_all[s_idx, :] = out["train_mse"]
            eval_mse_all[s_idx, :] = out["eval_mse"]

        width_path = runs_dir / f"{model_name}.npz"
        save_npz(
            width_path,
            model_name=np.asarray([model_name]),
            seeds=np.asarray(seed_list, dtype=np.int32),
            snapshot_steps=snapshot_steps,
            tracked_mode_names=mode_bank["tracked_mode_names"],
            tracked_mode_freqs=mode_bank["tracked_mode_freqs"],
            tracked_mode_types=mode_bank["tracked_mode_types"],
            tracked_mode_amps=mode_bank["tracked_mode_amps"],
            freq_plane_freqs=mode_bank["freq_plane_freqs"],
            param_count=param_count_all,
            pred_eval=pred_eval_all,
            residual_eval=residual_eval_all,
            target_mode_coeff_eval=target_mode_coeff_all,
            target_mode_abs_eval=target_mode_abs_all,
            freq_plane_l2_eval=freq_plane_l2_all,
            train_loss=train_loss_all,
            train_mse=train_mse_all,
            eval_mse=eval_mse_all,
        )
        runs_manifest[model_name] = f"runs/{model_name}.npz"

    save_json(save_dir / "network_configs.json", {"models": resolved_model_cfgs})

    summary = {
        "probe_geometry": "probe_geometry.npz",
        "target_task": "target_task.npz",
        "mode_bank": "mode_bank.npz",
        "network_configs": "network_configs.json",
        "runs": runs_manifest,
        "meta": {
            "n_train": n_train,
            "n_eval": n_eval,
            "n_probe": n_probe,
            "probe_domain": probe_domain,
            "train_grid": train_grid,
            "seeds": seed_list,
            "steps": steps,
            "lr": lr,
            "eval_every": eval_every,
            "snapshot_steps": snapshot_steps.tolist(),
            "target_terms": target_terms,
            "loss_objective": "0.5 * ||f(X_train)-y_train||^2",
            "optimizer": "full_batch_sgd",
        },
        "runtime_sec": round(time.time() - t0, 2),
    }
    save_json(save_dir / "manifest.json", summary)

    print(f"\nDone. Spectral-bias comparison saved to {save_dir}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(
            "Usage: python3 -m experiments.ntk_spectral_bias_architecture_comparison "
            "configs/ntk_spectral_bias_architecture_comparison.yaml"
        )
    else:
        run(sys.argv[1])
