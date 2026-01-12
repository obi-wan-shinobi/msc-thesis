# AGENTS.md

## Purpose
This repository contains JAX/Neural Tangents experiments for NTK behavior.
Scripts live under `experiments/` and are configured by YAML files in `configs/`.
There is no formal package build system or test suite configured.

## Quick Commands
### Run an experiment (primary entrypoint)
- `python -m experiments.ntk_kernel_drift configs/ntk_kernel_drift.yaml`
- `python -m experiments.ntk_beta_dispersion configs/ntk_beta_dispersion.yaml`
- `python -m experiments.ntk_spectrum_drift configs/ntk_spectrum_drift.yaml`
- `python -m experiments.ntk_profile_convergence configs/ntk_profile.yaml`
- `python -m experiments.ntk_regression_convergence configs/ntk_regression_simple.yaml`

### Run a single configuration “test”
- There is no automated test runner.
- For a fast sanity run, edit a YAML config to use small widths/seeds/steps.
- Example: reduce `sweep.widths`, `sweep.seeds`, and `train.steps`.

### Lint/format
- No linting or formatting tool is configured in this repo.
- Follow existing formatting and run `python -m compileall` only if needed.

### Tests
- No `tests/` directory or pytest config is present.
- If you add tests later, use: `pytest path/to/test_file.py::test_name`.

## Project Layout
- `core/`: shared model, data, analysis, training utilities.
- `experiments/`: CLI-style experiment scripts (invoked with `python -m`).
- `configs/`: YAML configurations for experiments.
- `utils/`: IO helpers for saving artifacts and run metadata.
- `notebooks/`: exploratory notebooks and plotting helpers.

## Execution Model
- Experiments are launched via `python -m experiments.<module> <config.yaml>`.
- Each experiment writes outputs to `results/<experiment_name>/...`.
- Use `utils.artifacts.make_run_dir` and `save_npz`/`save_json` for artifacts.
- Each experiment typically logs progress with `print` statements.

## Dependencies
- Python 3.10+ is assumed (type hints use `int | None`).
- Core libraries: JAX, Neural Tangents, Optax, NumPy, PyYAML.
- Install dependencies via your environment manager (no lockfile here).
- CPU runs are fine for small sweeps; GPU/TPU is optional.
- If using CUDA, ensure a matching `jaxlib` build is installed.

## Config Workflow
- YAML config keys are read directly; missing keys raise `KeyError`.
- Keep `experiment.name` unique for run directories.
- Use `experiment.seed` for data sampling.
- Use `train.log_every` for periodic loss prints.
- Use `train.eval_every` to control snapshot frequency.
- Use `sweep.seeds` for number of init seeds (0..n-1).

## Artifacts and Results
- Runs save under `results/<experiment>/<timestamp-id>/`.
- `manifest.json` summarizes run metadata and file locations.
- Per-width outputs are stored as `.npz` files under `runs/`.
- Use `utils.artifacts_io.latest_run_dir` to locate the latest run.
- Keep generated results out of version control unless needed.

## Suggested Validation
- Prefer a tiny config for quick checks.
- Example: set `train.steps=5` and `sweep.widths=[16]`.
- Run one experiment module as a sanity check.
- There is no benchmark or CI validation.

## Code Style Guidelines
### Imports
- Follow the existing import order:
  1. Standard library
  2. Third-party libraries (jax, numpy, yaml, optax, neural_tangents)
  3. Local modules (e.g., `from core...`, `from utils...`)
- Prefer explicit imports over `import *`.

### Formatting
- Use 4-space indentation.
- Keep lines reasonably short (≈88–100 chars) and wrap long calls.
- Use blank lines between logical sections (functions, blocks, sections).
- Keep module header comments when present (see `core/` and `experiments/`).

### Types and Annotations
- Use Python 3.10+ union syntax (`int | None`) as seen in `core/train.py`.
- Use `jnp.ndarray` or `np.ndarray` for array-like types when annotating.
- Prefer `Path | str` for filesystem paths (see `utils/artifacts.py`).
- Use `NamedTuple` for lightweight structured data (see `core/data.py`).

### Naming Conventions
- Functions and variables: `snake_case`.
- Constants: `UPPER_SNAKE_CASE` (rare in this repo).
- Modules: short, descriptive filenames (`analysis.py`, `train.py`).
- Config keys: lower snake case in YAML (`train.steps`, `model.depth_hidden`).

### JAX/NumPy Usage
- Use `jax.numpy as jnp` for JAX operations.
- Use `numpy as np` for host-side processing and saving arrays.
- Use `jax.random.PRNGKey(seed)` for reproducible RNG.
- Use `jax.jit` and `jax.grad` for performance/auto-diff; keep functions pure.
- Avoid mixing NumPy and JAX arrays inside JIT-compiled functions.

### Experiment Structure
- Parse configs with `yaml.safe_load(open(config_path))` (current pattern).
- Extract config sections into `exp_cfg`, `data_cfg`, `model_cfg`, etc.
- Keep run metadata and artifacts in `results/` via `utils/artifacts`.
- Use clear progress prints with step counters and loss values.

### Error Handling
- The codebase is lightweight and uses assertions where needed.
- Prefer simple, explicit checks (e.g., shape compatibility) over heavy exceptions.
- When adding new IO, ensure directories exist via `ensure_dir`.

### IO and Artifacts
- Use `utils.artifacts.save_npz` for arrays and `save_json` for metadata.
- Avoid writing ad-hoc files outside `results/` unless necessary.
- If you add new artifact types, keep their naming consistent and documented.

### Configs
- Keep YAML files in `configs/` with clear names.
- Use numeric scalars as plain YAML numbers, not strings.
- Group config sections as `experiment`, `data`, `target`, `model`, `train`, `sweep`.

### Notebooks
- Notebooks are exploratory; avoid heavy refactors in notebooks.
- Reuse logic from `core/` and `utils/` rather than duplicating it.
- Keep plotting helpers in `notebooks/utils_plotting.py`.

## Reproducibility
- Use explicit seeds from config files.
- Write run metadata (`run_info.json`) with `make_run_dir`.
- Save copies of configs using `write_config_copy`.

## Performance Notes
- Be mindful of JIT warm-up costs; existing code warms up once.
- Prefer batching and vectorized operations in JAX.
- Avoid host-device syncs inside tight loops unless necessary.

## Adding New Experiments
- Place new scripts under `experiments/` and add a YAML config.
- Follow the existing structure: `run(config_path)` + `__main__` entrypoint.
- Ensure output is saved via `utils.artifacts` helpers.

## Cursor/Copilot Rules
- No `.cursor/rules/`, `.cursorrules`, or `.github/copilot-instructions.md` files were found.
- If you add such rules later, mirror them here.
