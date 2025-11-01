# ------------------------------------------------------------
# core/model.py
# ------------------------------------------------------------
import jax
from jax import jit
from neural_tangents import stax


def build_mlp(width: int, b_std: float = 0.05, depth_hidden: int = 2):
    """
    Build an NTK-parameterized fully-connected ReLU network.
    Returns (init_fn, apply_fn, kernel_fn) consistent with neural_tangents.stax.
    """
    layers = []
    for _ in range(depth_hidden):
        layers += [
            stax.Dense(width, b_std=b_std, parameterization="ntk"),
            stax.Relu(),
        ]
    layers += [stax.Dense(1, b_std=b_std, parameterization="ntk")]
    init_fn, apply_fn, kernel_fn = stax.serial(*layers)
    apply_fn = jit(apply_fn)  # speed up forward passes
    return init_fn, apply_fn, kernel_fn
