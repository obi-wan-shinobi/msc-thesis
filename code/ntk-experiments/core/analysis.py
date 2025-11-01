# ------------------------------------------------------------
# core/analysis.py
# ------------------------------------------------------------
import jax.numpy as jnp
import neural_tangents as nt


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
