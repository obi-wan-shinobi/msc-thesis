import jax
import jax.numpy as jnp
from jax import random, jit, vmap, value_and_grad, jacrev
from jax.flatten_util import ravel_pytree
import optax
import matplotlib.pyplot as plt
import numpy as np

import neural_tangents as nt
from neural_tangents import stax

from typing import Callable, Optional, Tuple, List, Union


# ------------------------------------------------------------
# Build NTK-parameterized MLP for a given width
# ------------------------------------------------------------
def build_mlp(width: int, b_std: float = 0.05, depth_hidden: int = 2):
    layers = []
    for _ in range(depth_hidden):
        layers += [
            stax.Dense(width, b_std=b_std, parameterization='ntk'),
            stax.Relu(),
        ]
    layers += [stax.Dense(1, b_std=b_std, parameterization='ntk')]
    init_fn, apply_fn, kernel_fn = stax.serial(*layers)
    apply_fn = jit(apply_fn)  # speed
    return init_fn, apply_fn, kernel_fn

def train(
    params,
    apply_fn,
    Xtr,
    ytr,
    steps: int = 200,
    lr: float = 1.0,
    log_every: int | None = None,   # e.g., 50 -> print every 50 steps
    return_history: bool = False,   # return list of (step, loss)
):
    opt = optax.sgd(lr)
    opt_state = opt.init(params)

    def loss_fn(p):
        preds = apply_fn(p, Xtr).squeeze()
        return jnp.mean((preds - ytr) ** 2)

    @jit
    def step(p, s):
        l, g = value_and_grad(loss_fn)(p)
        up, s = opt.update(g, s, p)
        p = optax.apply_updates(p, up)
        return p, s, l

    # Warm-up JIT
    _ = step(params, opt_state)

    history = [] if return_history else None
    for it in range(1, steps + 1):
        params, opt_state, loss = step(params, opt_state)

        should_log = (
            log_every is not None
            and (it % log_every == 0 or it == steps)
        )
        if should_log:
            print(f"[train] step {it:5d} | loss={float(loss):.6e}", flush=True)

        if return_history:
            history.append((it, float(loss)))

    return (params, history) if return_history else params
