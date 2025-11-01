# utils/artifacts_io.py
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def latest_run_dir(base: str | Path, exp_name: str) -> Path:
    base = Path(base) / exp_name
    if (base / "latest").exists():
        # prefer the symlink if present
        return (base / "latest").resolve()
    # else pick max timestamp-ish folder
    runs = [p for p in base.iterdir() if p.is_dir()]
    return sorted(runs)[-1] if runs else None


def load_manifest(run_dir: str | Path) -> dict:
    return json.loads((Path(run_dir) / "manifest.json").read_text())


def npz(run_dir: str | Path, rel_path: str) -> dict:
    return dict(np.load(Path(run_dir) / rel_path))


def list_widths(manifest: dict) -> list[int]:
    # keys in "profiles" or "predictions" are strings
    if "profiles" in manifest:
        return [int(w) for w in manifest["profiles"].keys()]
    if "predictions" in manifest:
        return [int(w) for w in manifest["predictions"].keys()]
    return []
