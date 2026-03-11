# AGENTS

## Scope
- Applies to the `interim_presentation` repository.
- No other `AGENTS.md`, Cursor, or Copilot rules were found here.
- This guide focuses on Manim slide code in `deck.py` and helpers in `theme.py`.

## Project Summary
- Manim + manim-slides project for thesis presentation slides.
- Slide classes live in `deck.py` and inherit from `manim_slides.Slide`.
- Visual helper functions live in `theme.py`.
- Assets (images) are stored in `assets/`.
- Rendered output is stored in `media/` and `slides/` (generated).

## Directory Map
- `deck.py`: main slide definitions and animation timeline.
- `theme.py`: reusable slide helpers (title/bullets/figure).
- `assets/`: logos and images used in slides.
- `media/`: Manim render output (generated).
- `slides/`: manim-slides output (generated).
- `venv/`: local Python virtual environment (generated).

## Environment Setup
- Use the local venv if present: `source venv/bin/activate`.
- Python version in venv: 3.13.x (from `venv/pyvenv.cfg`).
- Dependencies are not pinned in this repo.
- Install as needed (commonly `manim` and `manim-slides`).
- Prefer `pip install` in the active venv.

## Dependency Notes
- Manim has system dependencies; follow Manim install docs if needed.
- Keep graphics/audio tooling local to the dev machine.
- Avoid adding new dependency managers without updating this guide.
- Document new runtime requirements in `README.md`.

## Build / Render Commands
- Quick render (low quality): `manim -pql deck.py ThesisIntro`.
- Full-quality render: `manim -pqh deck.py ThesisIntro`.
- Render all scenes: `manim -pql deck.py`.
- Select a different scene class by name, e.g. `manim -pql deck.py SomeScene`.
- If you use manim-slides CLI, check `manim-slides --help` for subcommands.
- Rendered videos appear under `media/` and `slides/`.

## Lint / Format Commands
- No lint or formatter config found (no `pyproject.toml`, `setup.cfg`, or `.pre-commit`).
- If you add one, document it here and keep it consistent with existing style.
- Keep formatting close to PEP 8 and the current file layout.

## Test Commands
- No test framework or test files found in this repo.
- If tests are added later, prefer `pytest`.
- Full test run (if present): `python -m pytest`.
- Single test example: `python -m pytest path/to/test_file.py::test_name`.
- Single test class example: `python -m pytest path/to/test_file.py::TestClass::test_name`.

## Code Style Guide

### Imports
- Keep Manim imports first: `from manim import *`.
- Keep `from manim_slides import Slide` close to Manim imports.
- Avoid unused imports; remove helpers you no longer use.
- Do not introduce wildcard imports outside Manim usage.

### Formatting
- Use 4-space indentation.
- Keep lines short and readable (around 88 chars where practical).
- Use trailing commas in multi-line calls to simplify diffs.
- Align method chains vertically as in `theme.py`.
- Use blank lines to separate logical blocks.

### Naming
- Slide classes: `PascalCase` (e.g., `ThesisIntro`).
- Helper functions: `snake_case` (e.g., `title_slide`).
- Variables: descriptive `snake_case` (e.g., `author_group`).
- Constants: `UPPER_CASE` only if truly constant.

### Types
- Current files are untyped; avoid adding type hints unless needed.
- If adding type hints, use standard `typing` and stay consistent.
- Prefer clear variable names over heavy typing in short animations.

### Scene Structure
- One scene class per conceptual slide or section.
- In `construct`, group layout objects before animations.
- Use `VGroup` for related elements and align with `to_edge` / `next_to`.
- Prefer explicit `font_size` and `color` arguments for consistency.
- Keep animation timing readable (`run_time`, `buff`, etc.).

### Layout & Typography
- Use `Text`, `MathTex`, or `MarkupText` with explicit `font_size`.
- Stick to Manim color constants (e.g., `BLUE`, `WHITE`) when possible.
- Favor `to_edge`, `next_to`, and `arrange` over absolute coordinates.
- Keep margins consistent with `buff` values (0.2–0.6 typical).
- Align related elements into a `VGroup` before positioning.

### Animation & Timing
- Use `FadeIn`/`FadeOut` for simple reveals and exits.
- Provide `run_time` when animating multiple elements together.
- For talk pauses, call `self.next_slide()` before `self.wait()`.
- Avoid overlapping animations unless the visual focus is clear.
- Keep animation sequences short and readable in the timeline.

### Helpers
- Keep reusable slide helpers in `theme.py`.
- Pass `scene` explicitly (no globals).
- Return created mobjects when follow-up positioning is needed.
- Avoid helper side effects beyond the intended `scene.play` calls.

### Assets
- Keep asset paths relative to repo root (e.g., `assets/logo.png`).
- Store new images in `assets/` with descriptive filenames.
- If adding new asset types, update this guide.

### Error Handling
- Avoid silent failures; raise or fail fast on missing assets.
- Prefer explicit checks when paths are computed dynamically.
- Keep error handling minimal and clear for render-time failures.

### Generated Outputs
- Treat `media/`, `slides/`, and `__pycache__/` as generated.
- Do not manually edit generated files.
- Avoid committing large render outputs unless explicitly requested.

### Documentation
- Keep `README.md` accurate for usage expectations.
- If you add new build or render steps, update this file too.

## Suggested Workflow
- Activate venv.
- Render a single scene with `manim -pql` for quick iteration.
- Switch to `-pqh` for final-quality export.
- Commit only source files (`deck.py`, `theme.py`, `assets/`).

## Quality Checklist
- Run a quick render after changes: `manim -pql deck.py SceneName`.
- Confirm assets load correctly and paths are valid.
- Verify slide pacing with `self.next_slide()` markers.
- Keep text readable at typical projector resolution.
- Remove temporary debugging objects before final render.

## Notes for Agents
- Keep changes minimal and focused on slide content.
- Preserve the existing visual style unless asked to redesign.
- Avoid reorganizing generated folders.
- Confirm new dependencies before adding config files.
