# ------------------------------------------------------------
# core/model.py
# ------------------------------------------------------------
import jax.numpy as jnp
from jax import jit, random
from neural_tangents import stax


def CustomNTKDense(out_dim, W_std=1.0, b_std=1.0):
    """
    A Dense layer with NTK parameterization where biases
    are initialized to 0 but remain trainable and contribute to the NTK.
    """

    def init_fn(rng, input_shape):
        in_dim = input_shape[-1]

        # 1. Initialize weights from a standard normal distribution N(0, 1)
        W = random.normal(rng, (in_dim, out_dim))

        # 2. Initialize biases strictly to 0
        b = jnp.zeros((out_dim,))

        output_shape = input_shape[:-1] + (out_dim,)
        return output_shape, (W, b)

    def apply_fn(params, inputs, **kwargs):
        W, b = params
        in_dim = inputs.shape[-1]  # 'm'

        # 3. Apply the 1 / sqrt(m) NTK scaling to the weights dynamically
        W_scaled = W * W_std

        # 4. Scale the bias by b_std. Since `b` is initially 0, the output
        # is unaffected at initialization. However, this ensures the gradients
        # during backprop scale perfectly with the theoretical NTK.
        b_scaled = b * b_std

        return jnp.dot(inputs, W_scaled) + b_scaled

    def kernel_fn(k, **kwargs):
        in_dim = 2  # hardcoded for our case
        eff_W_std = W_std * jnp.sqrt(in_dim)
        # get analytical kernel assuming no bias
        _, _, base_dense_kernel = stax.Dense(
            out_dim, W_std=eff_W_std, b_std=0.0, parameterization="ntk"
        )
        k_out = base_dense_kernel(k, **kwargs)

        # now manually add the training contribution to the NTK
        corrected_ntk = k_out.ntk + (b_std**2)

        return k_out.replace(ntk=corrected_ntk)

    return init_fn, apply_fn, kernel_fn


def build_mlp_custom(
    width: int,
    b_std: float = 0.05,
    depth_hidden: int = 2,
    parameterization: str = "ntk",
):
    """
    Build an NTK-parameterized fully-connected ReLU network using custom dense layer.
    Returns (init_fn, apply_fn, kernel_fn) consistent with neural_tangents.stax.
    """
    layers = []
    for _ in range(depth_hidden):
        layers += [
            CustomNTKDense(width, W_std=1, b_std=b_std),
            stax.Relu(),
        ]
    layers += [stax.Dense(1, W_std=1, b_std=None, parameterization=parameterization)]
    init_fn, apply_fn, kernel_fn = stax.serial(*layers)
    apply_fn = jit(apply_fn)  # speed up forward passes
    return init_fn, apply_fn, kernel_fn


def build_mlp(
    width: int,
    b_std: float = 0.05,
    depth_hidden: int = 2,
    parameterization: str = "ntk",
):
    """
    Build an NTK-parameterized fully-connected ReLU network.
    Returns (init_fn, apply_fn, kernel_fn) consistent with neural_tangents.stax.
    """
    layers = []
    for _ in range(depth_hidden):
        layers += [
            stax.Dense(width, b_std=b_std, parameterization=parameterization),
            stax.Relu(),
        ]
    layers += [stax.Dense(1, b_std=None, parameterization=parameterization)]
    init_fn, apply_fn, kernel_fn = stax.serial(*layers)
    apply_fn = jit(apply_fn)  # speed up forward passes
    return init_fn, apply_fn, kernel_fn
