import jax
import jax.numpy as jnp
from jax import random, jit, vmap, value_and_grad, jacrev
from jax.flatten_util import ravel_pytree
import optax
import matplotlib.pyplot as plt
import numpy as np

import neural_tangents as nt
from neural_tangents import stax

from typing import Callable, Optional, Tuple, List, Union, NamedTuple


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

# ----------------------------
# Empirical NTK profile helper
# ----------------------------
def empirical_profile(apply_fn, params, X_eval, x0):
    """
    Return theta_emp(x0, X_eval) as a vector [len(X_eval)] using Neural Tangents' empirical NTK.
    """
    f = lambda p, x: apply_fn(p, x)           # [B,2] -> [B,1]
    emp_fn = nt.empirical_ntk_fn(f, trace_axes=(), vmap_axes=0)
    prof = emp_fn(x0, X_eval, params).squeeze()   # [1,N] -> [N]
    return prof

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

class FourierTarget(NamedTuple):
    """Parametrization of a Fourier-mixture target on the unit circle."""
    Ks: jnp.ndarray       # (m,) integer frequencies k >= 1 (sorted if you want partials by 'lowest first')
    amps: jnp.ndarray     # (m,) amplitudes a_k
    phases: jnp.ndarray   # (m,) phases φ_k in radians


def make_fourier_target(
    Ks: jnp.ndarray,
    amps: jnp.ndarray,
    phases: Optional[jnp.ndarray] = None,
) -> FourierTarget:
    """
    Build a FourierTarget. If phases is None, uses zeros.
    Shapes must match; no implicit sorting is done.
    """
    Ks = jnp.asarray(Ks)
    amps = jnp.asarray(amps)
    if phases is None:
        phases = jnp.zeros_like(Ks, dtype=Ks.dtype)
    else:
        phases = jnp.asarray(phases, dtype=Ks.dtype)
    assert Ks.shape == amps.shape == phases.shape, "Ks, amps, phases must have same shape"
    return FourierTarget(Ks=Ks, amps=amps, phases=phases)


def f_star_gamma(gamma: jnp.ndarray, spec: FourierTarget) -> jnp.ndarray:
    """
    Evaluate the target y(gamma) = Σ_k a_k * sin(k * gamma + φ_k)
    gamma: (N,)
    returns: (N,)
    """
    A = spec.Ks[:, None] * gamma[None, :] + spec.phases[:, None]     # (m, N)
    return jnp.einsum('m,mn->n', spec.amps, jnp.sin(A))


def f_star_X(X: jnp.ndarray, spec: FourierTarget) -> jnp.ndarray:
    """
    Evaluate the target on 2D points on the unit circle X = [(cos γ, sin γ)].
    X: (N, 2)
    returns: (N,)
    """
    gamma = jnp.arctan2(X[:, 1], X[:, 0])   # in [-π, π)
    return f_star_gamma(gamma, spec)


def f_star_gamma_partial(gamma: jnp.ndarray, spec: FourierTarget, k_count: int) -> jnp.ndarray:
    """
    Cumulative partial sum using the first k_count modes (in the current order of spec.Ks).
    k_count: 1..m
    returns: (N,)
    """
    Ksel = spec.Ks[:k_count][:, None]
    Asel = spec.amps[:k_count][:, None]
    Psel = spec.phases[:k_count][:, None]
    G    = gamma[None, :]
    return jnp.sum(Asel * jnp.sin(Ksel * G + Psel), axis=0)


def truncate_spec(spec: FourierTarget, k_count: int) -> FourierTarget:
    """Return a new spec keeping only the first k_count modes."""
    return FourierTarget(
        Ks=spec.Ks[:k_count],
        amps=spec.amps[:k_count],
        phases=spec.phases[:k_count],
    )


def sample_train_on_circle(
    gamma_eval: jnp.ndarray,
    X_eval: jnp.ndarray,
    spec: FourierTarget,
    M_train: int,
    key: random.PRNGKey,
    add_noise_std: float = 0.0,
):
    """
    Sample M_train indices from the eval grid for training and return (X_train, gamma_train, y_train).
    """
    idx = random.choice(key, gamma_eval.shape[0], shape=(M_train,), replace=False)
    X_train = X_eval[idx]
    gamma_train = gamma_eval[idx]
    y_train = f_star_gamma(gamma_train, spec)
    if add_noise_std and add_noise_std > 0:
        y_train = y_train + add_noise_std * random.normal(random.PRNGKey(1), y_train.shape)
    return X_train, gamma_train, y_train