import os
import sys
from datetime import date

import yaml

# ---------- user parameters ----------
EXPERIMENTS_DIR = "experiments"
# ------------------------------------


def next_experiment_id(base_dir):
    """Return next experiment ID as EXP###."""
    os.makedirs(base_dir, exist_ok=True)
    existing = [
        d
        for d in os.listdir(base_dir)
        if d.startswith("EXP") and os.path.isdir(os.path.join(base_dir, d))
    ]
    nums = [int(d[3:6]) for d in existing if d[3:6].isdigit()]
    next_num = max(nums) + 1 if nums else 1
    return f"EXP{next_num:03d}"


def create_experiment(title, goal):
    exp_id = next_experiment_id(EXPERIMENTS_DIR)
    folder = os.path.join(EXPERIMENTS_DIR, f"{exp_id}_{title}")
    os.makedirs(folder, exist_ok=True)
    os.makedirs(os.path.join(folder, "plots"), exist_ok=True)

    # --- markdown log template ---
    readme = f"""# Experiment {exp_id} — {goal}
**Date:** {date.today().isoformat()}  
**Script:** <add notebook/script>  
**Paper(s):** <add reference>

---

## 1 Goal
{goal}

---

## 2 Setup
| Component | Description |
|------------|--------------|
| Dataset |  |
| Model |  |
| Optimizer |  |
| Epochs |  |
| Seeds |  |

---

## 3 Metrics
-  
---

## 4 Results
| Param | Train | Test | Note |
|-------|-------:|-----:|------|
|  |  |  |  |

**Figures:** `plots/`

---

## 5 Observations
-

---

## 6 Next Steps
-

---

## 7 Outcome
-
"""
    with open(os.path.join(folder, "README.md"), "w") as f:
        f.write(readme)

    # --- config.yaml stub ---
    cfg = {
        "id": exp_id,
        "date": date.today().isoformat(),
        "title": title,
        "goal": goal,
        "status": "pending",
        "metrics": [],
        "results": {},
    }
    with open(os.path.join(folder, "config.yaml"), "w") as f:
        yaml.dump(cfg, f, sort_keys=False)

    print(f"Created new experiment: {folder}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python make_experiment.py <short_title> <goal>")
        sys.exit(1)
    create_experiment(sys.argv[1], " ".join(sys.argv[2:]))
