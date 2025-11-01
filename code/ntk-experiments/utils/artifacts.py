# utils/artifacts.py
from __future__ import annotations

import json
import os
import platform
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
import yaml


def ensure_dir(path: Path | str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_config_copy(save_dir: Path, cfg: dict) -> None:
    ensure_dir(save_dir)
    (save_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))


def save_npz(path: Path | str, **arrays) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    np.savez_compressed(path, **arrays)
    return path


def save_json(path: Path | str, data: dict) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=2))
    return path


def make_run_dir(base_dir: Path | str, exp_name: str, tz: str | None = None) -> Path:
    """
    Create a unique run directory: <base_dir>/<exp_name>/<YYYYMMDD-HHMMSS>-<shortid>
    - Uses local time by default; set tz="UTC" for UTC timestamps.
    - Also writes a lightweight run_info.json (env, host, python).
    """
    base = Path(base_dir) / exp_name
    ensure_dir(base)

    # timestamp
    if tz == "UTC":
        now = datetime.now(timezone.utc)
    else:
        now = datetime.now()  # local timezone (Europe/Amsterdam for you)
    ts = now.strftime("%Y%m%d-%H%M%S")

    # short unique id in case you launch multiple runs within the same second
    shortid = str(uuid4())[:8]
    run_dir = base / f"{ts}-{shortid}"
    ensure_dir(run_dir)

    # optional: maintain a 'latest' symlink for convenience (best-effort)
    try:
        latest = base / "latest"
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        latest.symlink_to(run_dir.name)  # relative symlink inside base
    except Exception:
        pass  # ok on Windows without dev mode

    # write basic env info for provenance
    env = {
        "timestamp": now.isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "cwd": str(Path.cwd()),
        "run_id": run_dir.name,
    }
    save_json(run_dir / "run_info.json", env)
    return run_dir
