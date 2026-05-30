"""
Standalone Manim scenes for spectral-bias training dynamics.

Renders a two-panel animation:
  - left: fitted function vs target over time,
  - right: per-mode residual magnitude decay curves over steps (log y-scale).

Usage example:
  python -m manim scripts/spectral_bias_manim_scene.py SpectralBiasFitAndModes -qh
  python -m manim scripts/spectral_bias_manim_scene.py SpectralBiasModeLegend -s -qh

Optional environment variables:
  NTK_SB_RUN_DIR        Absolute/relative run dir (overrides auto-latest).
  NTK_SB_BASE_RESULTS   Results root (default: <repo>/results).
  NTK_SB_EXP_NAME       Experiment name (default: ntk_spectral_bias_architecture_comparison).
  NTK_SB_MODEL          Model name inside run (default: first configured model).
  NTK_SB_SEED_INDEX     Seed index (default: 0).
  NTK_SB_FRAME_STRIDE   Use every k-th snapshot (default: 1).
  NTK_SB_GAMMA_STRIDE   Downsample gamma by stride for plotting (default: 1).
  NTK_SB_RUNTIME_SEC    Scene run time in seconds (default: 20).
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np
from manim import (
    BLACK,
    BLUE_D,
    DOWN,
    GREEN_D,
    LEFT,
    ORANGE,
    PURPLE_D,
    RED_D,
    RIGHT,
    WHITE,
    Axes,
    Line,
    MathTex,
    Scene,
    ValueTracker,
    VGroup,
    VMobject,
    always_redraw,
    config,
    linear,
)

config.background_color = WHITE


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def _latest_run_dir(base_results: Path, exp_name: str) -> Path:
    exp_root = base_results / exp_name
    latest = exp_root / "latest"
    if latest.exists():
        return latest.resolve()

    runs = sorted([p for p in exp_root.iterdir() if p.is_dir()])
    if not runs:
        raise FileNotFoundError(f"No run directories found under {exp_root}")
    return runs[-1]


def _read_npz(path: Path) -> dict:
    return dict(np.load(path, allow_pickle=True))


def _resolve_model_name(manifest: dict, run_dir: Path) -> str:
    env_model = os.getenv("NTK_SB_MODEL")
    if env_model:
        if env_model not in manifest["runs"]:
            raise KeyError(
                f"NTK_SB_MODEL='{env_model}' not found in run models: "
                f"{sorted(manifest['runs'].keys())}"
            )
        return env_model

    network_cfg_file = run_dir / manifest.get("network_configs", "")
    if network_cfg_file.exists():
        cfg = json.loads(network_cfg_file.read_text())
        names = [
            str(m["name"])
            for m in cfg.get("models", [])
            if str(m["name"]) in manifest["runs"]
        ]
        if names:
            return names[0]

    return sorted(manifest["runs"].keys())[0]


def _prepare_scene_data() -> dict:
    repo_root = Path(__file__).resolve().parents[1]

    base_results = Path(
        os.getenv("NTK_SB_BASE_RESULTS", str(repo_root / "results"))
    ).expanduser()
    exp_name = os.getenv("NTK_SB_EXP_NAME", "ntk_spectral_bias_architecture_comparison")

    run_dir_env = os.getenv("NTK_SB_RUN_DIR")
    run_dir = (
        Path(run_dir_env).expanduser().resolve()
        if run_dir_env
        else _latest_run_dir(base_results, exp_name)
    )

    manifest = json.loads((run_dir / "manifest.json").read_text())
    probe = _read_npz(run_dir / manifest["probe_geometry"])
    target = _read_npz(run_dir / manifest["target_task"])
    mode_bank = _read_npz(run_dir / manifest["mode_bank"])

    model_name = _resolve_model_name(manifest, run_dir)
    run = _read_npz(run_dir / manifest["runs"][model_name])

    seed_index = int(os.getenv("NTK_SB_SEED_INDEX", "0"))
    seeds = np.asarray(run["seeds"], dtype=int)
    if seed_index < 0 or seed_index >= len(seeds):
        raise IndexError(
            f"NTK_SB_SEED_INDEX={seed_index} out of range for seeds={seeds.tolist()}"
        )

    gamma_key = "gamma_probe" if "gamma_probe" in probe else "gamma_eval"
    target_key = "y_probe_true" if "y_probe_true" in target else "y_eval_true"

    gamma = np.asarray(probe[gamma_key], dtype=float)
    y_true = np.asarray(target[target_key], dtype=float)

    pred = np.asarray(run["pred_eval"][seed_index], dtype=float)
    mode_abs = np.asarray(run["target_mode_abs_eval"][seed_index], dtype=float)
    steps = np.asarray(run["snapshot_steps"], dtype=float)
    mode_names = [str(x) for x in mode_bank["tracked_mode_names"]]

    frame_stride = max(1, int(os.getenv("NTK_SB_FRAME_STRIDE", "1")))
    frame_indices = list(range(0, len(steps), frame_stride))
    if frame_indices[-1] != len(steps) - 1:
        frame_indices.append(len(steps) - 1)

    gamma_stride = max(1, int(os.getenv("NTK_SB_GAMMA_STRIDE", "1")))
    gamma = gamma[::gamma_stride]
    y_true = y_true[::gamma_stride]
    pred = pred[:, ::gamma_stride]

    pred = pred[frame_indices]
    mode_abs = mode_abs[frame_indices]
    steps = steps[frame_indices]

    return {
        "run_dir": str(run_dir),
        "model_name": model_name,
        "seed": int(seeds[seed_index]),
        "probe_domain": str(manifest.get("meta", {}).get("probe_domain", "eval")),
        "gamma": gamma,
        "target": y_true,
        "pred": pred,
        "mode_abs": mode_abs,
        "steps": steps,
        "mode_names": mode_names,
        "runtime_sec": float(os.getenv("NTK_SB_RUNTIME_SEC", "20")),
    }


# ---------------------------------------------------------------------------
# Interpolation helpers
# ---------------------------------------------------------------------------


def _interp_rows(values: np.ndarray, t: float) -> np.ndarray:
    n = int(values.shape[0])
    if n == 1:
        return values[0]
    t = float(np.clip(t, 0.0, n - 1))
    i0 = int(np.floor(t))
    i1 = min(i0 + 1, n - 1)
    a = t - i0
    return (1.0 - a) * values[i0] + a * values[i1]


# ---------------------------------------------------------------------------
# Log-scale helpers for the right panel
# ---------------------------------------------------------------------------
# We manually transform to log10 space and shift so y starts at 0,
# keeping the right-panel x-axis at the bottom.

LOG_EPS = 1e-9  # safe floor to avoid log(0)


def _shifted_log10(x: np.ndarray) -> tuple[np.ndarray, float]:
    log_x = np.log10(np.maximum(x, LOG_EPS))
    log_floor = float(np.min(log_x))
    return log_x - log_floor, log_floor


def _mode_name_to_latex(name: str, fallback_index: int) -> str:
    digits = re.findall(r"\d+", name)
    k = int(digits[0]) if digits else fallback_index
    return rf"\cos({k}\gamma)"


# ---------------------------------------------------------------------------
# Scenes
# ---------------------------------------------------------------------------


class SpectralBiasFitAndModes(Scene):
    def construct(self):
        data = _prepare_scene_data()

        gamma = data["gamma"]
        target = data["target"]
        pred = data["pred"]
        mode_abs = data["mode_abs"]
        steps = data["steps"]

        # ------------------------------------------------------------------
        # Left panel: auto-range y to include negative values
        # ------------------------------------------------------------------
        all_fit = np.concatenate([target, pred.ravel()])
        y_min_raw = float(np.min(all_fit))
        y_max_raw = float(np.max(all_fit))
        y_span = max(y_max_raw - y_min_raw, 1e-9)
        pad = 0.08 * y_span
        y_bot = y_min_raw - pad
        y_top = y_max_raw + pad
        y_step = max(0.2, (y_top - y_bot) / 6.0)

        left_axes = Axes(
            x_range=[0.0, 2.0 * np.pi, np.pi / 2],
            y_range=[y_bot, y_top, y_step],
            x_length=5.95,
            y_length=5.15,
            axis_config={"include_numbers": False, "color": BLACK},
        ).to_edge(LEFT, buff=0.15)

        # ------------------------------------------------------------------
        # Right panel: log-scale y (manual transform)
        # ------------------------------------------------------------------
        mode_vals = np.clip(mode_abs, 0.0, None)
        log_mode, _ = _shifted_log10(mode_vals)  # shape: (frames, modes)

        log_max = float(np.max(log_mode))
        log_top = log_max + 0.08 * max(log_max, 1.0)
        if log_top <= 0.0:
            log_top = 1.0
        log_step = max(0.2, log_top / 6.0)

        right_axes = Axes(
            x_range=[
                float(steps[0]),
                float(steps[-1]),
                max(1.0, (steps[-1] - steps[0]) / 5),
            ],
            y_range=[0.0, log_top, log_step],
            x_length=5.95,
            y_length=5.15,
            axis_config={"include_numbers": False, "color": BLACK},
        ).to_edge(RIGHT, buff=0.15)

        # ------------------------------------------------------------------
        # Axis labels
        # ------------------------------------------------------------------
        left_y_label = MathTex(r"f(\gamma)", font_size=34, color=BLACK).rotate(
            np.pi / 2
        )
        left_y_label.move_to(left_axes.c2p(0.75, y_top - 0.18 * (y_top - y_bot)))
        left_x_label = MathTex(r"\gamma", font_size=36, color=BLACK).next_to(
            left_axes, DOWN, buff=0.2
        )
        right_y_label = (
            MathTex(
                r"\langle r_t(\gamma),\,\phi_k(\gamma)\rangle",
                font_size=22,
                color=BLACK,
            )
            .rotate(np.pi / 2)
            .next_to(right_axes, LEFT, buff=0.14)
        )
        right_x_label = MathTex(r"t", font_size=36, color=BLACK).next_to(
            right_axes, DOWN, buff=0.2
        )

        # ------------------------------------------------------------------
        # Static target curve (left panel)
        # ------------------------------------------------------------------
        target_curve = left_axes.plot_line_graph(
            x_values=gamma.tolist(),
            y_values=target.tolist(),
            add_vertex_dots=False,
            line_color=BLACK,
            stroke_width=4,
        )

        # ------------------------------------------------------------------
        # Animated fit curve (left panel)
        # ------------------------------------------------------------------
        frame = ValueTracker(0)

        def fit_curve_fn():
            y = _interp_rows(pred, frame.get_value())
            return left_axes.plot_line_graph(
                x_values=gamma.tolist(),
                y_values=y.tolist(),
                add_vertex_dots=False,
                line_color=BLUE_D,
                stroke_width=5,
            )

        fit_curve = always_redraw(fit_curve_fn)

        # ------------------------------------------------------------------
        # Animated residual-mode curves (right panel, log-space y)
        # ------------------------------------------------------------------
        mode_palette = [BLUE_D, GREEN_D, RED_D, PURPLE_D, ORANGE]
        mode_curves = VGroup()

        for m in range(log_mode.shape[1]):
            color = mode_palette[m % len(mode_palette)]

            def mode_curve_fn(m_idx=m, c=color):
                t = float(np.clip(frame.get_value(), 0.0, len(steps) - 1))
                i0 = int(np.floor(t))
                i1 = min(i0 + 1, len(steps) - 1)
                a = t - i0

                step_t = (1.0 - a) * steps[i0] + a * steps[i1]
                log_t = (1.0 - a) * log_mode[i0, m_idx] + a * log_mode[i1, m_idx]

                pts = [
                    right_axes.c2p(float(steps[j]), float(log_mode[j, m_idx]))
                    for j in range(i0 + 1)
                ]
                pts.append(right_axes.c2p(float(step_t), float(log_t)))
                if len(pts) == 1:
                    pts.append(pts[0])

                curve = VMobject(color=c, stroke_width=4)
                curve.set_points_as_corners(pts)
                return curve

            mode_curves.add(always_redraw(mode_curve_fn))

        # ------------------------------------------------------------------
        # Compose scene
        # ------------------------------------------------------------------
        self.add(
            left_axes,
            right_axes,
            left_y_label,
            left_x_label,
            right_y_label,
            right_x_label,
            target_curve,
            fit_curve,
            mode_curves,
        )
        self.play(
            frame.animate.set_value(len(steps) - 1),
            run_time=float(data["runtime_sec"]),
            rate_func=linear,
        )
        self.wait(0.3)


class SpectralBiasModeLegend(Scene):
    def construct(self):
        data = _prepare_scene_data()
        mode_names = data["mode_names"]

        mode_palette = [BLUE_D, GREEN_D, RED_D, PURPLE_D, ORANGE]
        legend_items = VGroup()

        for m, name in enumerate(mode_names):
            color = mode_palette[m % len(mode_palette)]
            line = Line(LEFT * 1.0, RIGHT * 1.0, color=color, stroke_width=8)
            label = MathTex(
                _mode_name_to_latex(name, fallback_index=m + 1),
                font_size=48,
                color=BLACK,
            )
            legend_items.add(VGroup(line, label).arrange(RIGHT, buff=0.35))

        legend_items.arrange(RIGHT, buff=0.5)
        max_width = config.frame_width - 0.8
        if legend_items.width > max_width:
            legend_items.scale_to_fit_width(max_width)
        legend_items.move_to([0, 0, 0])
        self.add(legend_items)
