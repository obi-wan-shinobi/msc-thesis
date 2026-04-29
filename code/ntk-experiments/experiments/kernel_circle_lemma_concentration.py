# ------------------------------------------------------------
# experiments/kernel_circle_lemma_concentration.py
# ------------------------------------------------------------

import sys

from experiments.kernel_circle_lemma_validation import run


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(
            "Usage: python3 -m experiments.kernel_circle_lemma_concentration "
            "configs/kernel_circle_lemma_concentration.yaml"
        )
    else:
        run(sys.argv[1])
