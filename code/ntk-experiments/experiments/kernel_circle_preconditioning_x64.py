import sys

import jax


jax.config.update("jax_enable_x64", True)

from experiments.kernel_circle_preconditioning import run as _run_preconditioning


def run(config_path: str):
    return _run_preconditioning(config_path)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(
            "Usage: python3 -m experiments.kernel_circle_preconditioning_x64 "
            "configs/kernel_circle_preconditioning_high_frequency_target_without_highest_frequencies_x64.yaml"
        )
    else:
        run(sys.argv[1])
