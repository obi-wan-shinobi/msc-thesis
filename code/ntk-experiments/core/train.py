# ------------------------------------------------------------
# core/train.py
# ------------------------------------------------------------
import jax
import jax.numpy as jnp
import optax
from jax import jit, value_and_grad


def mse_loss(apply_fn, params, X, y):
    preds = apply_fn(params, X).squeeze()
    return jnp.mean((preds - y) ** 2)


def train(
    params,
    apply_fn,
    Xtr,
    ytr,
    steps: int = 200,
    lr: float = 1.0,
    log_every: int | None = None,
    return_history: bool = False,
):
    """
    Generic SGD training loop for MSE regression.
    """
    opt = optax.sgd(lr)
    opt_state = opt.init(params)

    @jit
    def step(p, s):
        l, g = value_and_grad(lambda p: mse_loss(apply_fn, p, Xtr, ytr))(p)
        updates, s = opt.update(g, s, p)
        p = optax.apply_updates(p, updates)
        return p, s, l

    # Warm up JIT
    _ = step(params, opt_state)

    history = [] if return_history else None
    for it in range(1, steps + 1):
        params, opt_state, loss = step(params, opt_state)
        if log_every and (it % log_every == 0 or it == steps):
            print(f"[train] step {it:5d} | loss={float(loss):.6e}", flush=True)
        if return_history:
            history.append((it, float(loss)))

    return (params, history) if return_history else params
