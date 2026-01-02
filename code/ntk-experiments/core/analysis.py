# ------------------------------------------------------------
# core/analysis.py
# ------------------------------------------------------------
import jax
import jax.numpy as jnp
import neural_tangents as nt


def empirical_ntk_matrix(apply_fn, params, X, X_other=None):
    """Compute full empirical NTK matrix Θ(x_i, x_j)."""
    f = lambda p, x: apply_fn(p, x)
    emp_fn = nt.empirical_ntk_fn(f, trace_axes=(), vmap_axes=0)

    if X_other is not None:
        return emp_fn(X, X_other, params)

    return emp_fn(X, X, params)  # [N, N]


# --- Empirical NTK profile ---
def empirical_profile(apply_fn, params, X_eval, x0):
    """theta_emp(x0, X_eval) using empirical NTK."""
    f = lambda p, x: apply_fn(p, x)
    emp_fn = nt.empirical_ntk_fn(f, trace_axes=(), vmap_axes=0)
    prof = emp_fn(x0, X_eval, params).squeeze()
    return prof


# --- Analytic NTK predictor ---
def analytic_ntk_predictor(X_train, y_train, X_eval, kernel_fn, reg=1e-6):
    """f_infinity(x) = K_xX (K_XX + lambdaI)^-1 y_train"""
    K_tt = kernel_fn(X_train, X_train, get="ntk")
    K_xt = kernel_fn(X_eval, X_train, get="ntk")
    alpha = jnp.linalg.solve(K_tt + reg * jnp.eye(K_tt.shape[0]), y_train[:, None])
    return (K_xt @ alpha).squeeze()


# --- Kernel drift ---
def kernel_drift(K0, KT):
    """Relative Frobenius drift."""
    diff = jnp.linalg.norm(KT - K0)
    base = jnp.linalg.norm(K0)
    return float(diff / (base + 1e-12))


# --- Relative error metric ---
def relerr(a, b):
    """Relative L2 error between predictions."""
    return float(jnp.linalg.norm(a - b) / (jnp.linalg.norm(b) + 1e-12))


def empirical_ntk_spectrum(apply_fn, params, X):
    """
    Returns eigenvalues (sorted descending) of empirical NTK on X.
    """
    K = empirical_ntk_matrix(apply_fn, params, X)
    # K is PSD, but we use eigh for numerical stability
    evals, _ = jnp.linalg.eigh(K.squeeze())
    evals = jnp.flip(evals)  # largest first
    return jnp.asarray(evals)


def empirical_ntk_diag(apply_fn, params, X):
    """Return only the diagonal elements of the empirical NTK."""
    K = empirical_ntk_matrix(apply_fn, params, X)
    K = jnp.squeeze(K)
    return jnp.diag(K)


def on_diag_moments(diag_K_stack):
    """
    Compute mean, second moment, and normalized second moment S2 = E[K^2]/E[K]^2.
    diag_K_stack: array [S, M] or flattened.
    """
    diag = jnp.asarray(diag_K_stack).reshape(-1)
    mu = jnp.mean(diag)
    m2 = jnp.mean(diag**2)
    S2 = (m2 / (mu**2 + 1e-12)).item()
    return float(mu), float(m2), float(S2)


def delta_ntk_single_update(params, apply_fn, x, y, lr=1.0):
    """
    Perform ONE SGD step on a single (x, y) pair (square loss) and compute:
        - initial K(x,x)
        - updated K(x,x)
        - absolute and relative change
    Returns: K0_xx, K1_xx, ΔK, ΔK / K0_xx
    """
    x = x.reshape(1, -1)
    y = y.reshape(
        1,
    )

    f = lambda p, z: apply_fn(p, z).squeeze()

    emp_fn = nt.empirical_ntk_fn(
        lambda p, z: apply_fn(p, z), trace_axes=(), vmap_axes=0
    )
    K0_xx = emp_fn(x, x, params).squeeze()

    def loss_single(p):
        pred = f(p, x)
        return jnp.mean((pred - y) ** 2)

    g = jax.grad(loss_single)(params)
    params1 = jax.tree_util.tree_map(lambda p, gg: p - lr * gg, params, g)

    K1_xx = emp_fn(x, x, params1).squeeze()
    dK = K1_xx - K0_xx
    rel = float(dK / (K0_xx + 1e-12))
    return float(K0_xx), float(K1_xx), float(dK), rel
