from typing import Any, Callable, Optional, Sequence, Union

import jax.numpy as jnp
import neural_tangents as nt
import numpy as np


class NTKAnalyzer:
    """
    A class to compute and analyze the Neural Tangent Kernel (NTK)
    for finite-width and finite-depth neural networks using JAX and neural-tangents.
    """

    def __init__(
        self,
        apply_fn: Callable,  # The network's apply function (e.g., from flax or nt.stax)
        params: Any,  # Network parameters from a specific initialization
        depth: Optional[int] = None,
        widths: Optional[Sequence[int]] = None,
    ):
        """
        Initializes the NTK analyzer with a given network and its parameters.

        Args:
            apply_fn: The forward pass function of the neural network.
                      It should take (params, inputs, **kwargs) and return outputs.
            params: A PyTree of the neural network parameters.
            depth: (Optional) Depth of the network.
            widths: (Optional) Sequence of widths for each layer.
        """
        self.apply_fn = apply_fn
        self.params = params
        self.depth = depth
        self.widths = widths  # e.g., [input_dim, n1, n2, ..., output_dim]

        # Calculate beta if width information is available
        self.beta = self._calculate_beta()

        # Initialize the empirical NTK function
        # trace_axes=() and diagonal_axes=() ensure the exact, unapproximated NTK.
        # trace_axes -> if N has multiple outputs this would sum the diagonal_axes
        # diagonal_axes -> compute only the diagonal elements of the NTK (ie. x_i, x_i)
        self.empirical_ntk_fn = nt.empirical_ntk_fn(
            self.apply_fn, trace_axes=(), diagonal_axes=()
        )

    def _calculate_beta(self) -> Optional[float]:
        """Calculates beta = sum(1/n_j) for hidden layers if width info is provided."""
        if self.widths and len(self.widths) > 2:  # Needs at least one hidden layer
            # Assuming widths include input/output, so slice for hidden layers
            hidden_widths = self.widths[1:-1]
            if hidden_widths:
                return sum(1.0 / w for w in hidden_widths)
        return None

    def get_beta(self) -> Optional[float]:
        """Returns the calculated beta value."""
        return self.beta

    def update_params(self, new_params: Any) -> None:
        self.params = new_params

    def compute_ntk_dataset(
        self,
        x1: jnp.ndarray,
        x2: Optional[jnp.ndarray] = None,
        params_override: Optional[Any] = None,
        **apply_fn_kwargs,
    ) -> jnp.ndarray:
        """
        Computes the empirical NTK matrix for given dataset(s).

        Args:
            x1: A JAX array of input data points (N1, ...).
            x2: (Optional) A JAX array of input data points (N2, ...).
                If None, computes K(x1, x1).
            **apply_fn_kwargs: Additional keyword arguments for the network's apply_fn.

        Returns:
            The NTK matrix, e.g., of shape (N1, N2) if output is scalar.
            If network has multiple outputs, shape is (N1, N2, output_dim, output_dim)
            or (N1, N2, output_dim) if 'trace_axes' was used differently.
            For single scalar output, it's (N1, N2).
        """
        current_params = params_override if params_override is not None else self.params
        kernel = self.empirical_ntk_fn(x1, x2, current_params, **apply_fn_kwargs)
        return kernel

    def compute_ntk_pair(
        self, x_single1: jnp.ndarray, x_single2: jnp.ndarray, **apply_fn_kwargs
    ) -> jnp.ndarray:
        """
        Computes the NTK for a single pair of inputs.
        Assumes x_single1 and x_single2 are individual data points (not batched).
        """
        # Add batch dimension if apply_fn expects it
        if x_single1.ndim == 1:  # Or based on expected input dims
            x_single1 = x_single1[None, ...]
        if x_single2.ndim == 1:
            x_single2 = x_single2[None, ...]
        kernel = self.empirical_ntk_fn(
            x_single1, x_single2, self.params, **apply_fn_kwargs
        )
        return kernel.squeeze()  # Remove batch dimension if added

    def compute_infinite_ntk_dataset(
        self,
        x1: jnp.ndarray,
        x2: Optional[jnp.ndarray] = None,
        kernel_fn: Optional[Callable] = None,  # Pass the stax.kernel_fn
        **apply_fn_kwargs,
    ) -> Optional[jnp.ndarray]:
        """
        Computes the infinite-width NTK for comparison.
        Requires the kernel_fn from the stax definition of the network.

        Args:
            x1: Input data.
            x2: (Optional) Second input data. If None, computes K(x1, x1).
            kernel_fn: The kernel_fn from nt.stax network definition.
            **apply_fn_kwargs: Keyword arguments for the kernel_fn.

        Returns:
            The infinite-width NTK matrix, or None if kernel_fn is not provided.
        """
        if not kernel_fn:
            raise AttributeError("Must pass kernel_fn in")
        kernel = kernel_fn(x1, x2, get="ntk", **apply_fn_kwargs)
        return kernel

    def _prepare_2d_ntk_for_analysis(
        self, x: jnp.ndarray, method: str = "trace", **apply_fn_kwargs
    ) -> jnp.ndarray:
        """Helper function to get a 2D NTK matrix for spectral analysis etc."""
        ntk_tensor = self.compute_ntk_dataset(x, None, **apply_fn_kwargs)  # K(x,x)
        N = x.shape[0]

        if ntk_tensor.ndim == 2:
            return ntk_tensor
        elif (
            ntk_tensor.ndim == 4
            and ntk_tensor.shape[0] == N
            and ntk_tensor.shape[1] == N
        ):
            D_out = ntk_tensor.shape[-1]
            if ntk_tensor.shape[-2] != D_out:
                raise ValueError(
                    f"Expected last two output dimensions of NTK to be equal, got {ntk_tensor.shape}"
                )

            if method == "trace":
                return jnp.trace(ntk_tensor, axis1=-2, axis2=-1)
            elif method == "full":
                reshaped_ntk = jnp.transpose(ntk_tensor, (0, 2, 1, 3))
                return reshaped_ntk.reshape((N * D_out, N * D_out))
            else:
                raise ValueError(
                    f"Unknown method for 4D NTK: {method}. Choose 'trace' or 'full'."
                )
        elif (
            ntk_tensor.ndim == 3
            and ntk_tensor.shape[0] == N
            and ntk_tensor.shape[1] == N
        ):
            print(
                f"Warning: NTK tensor is 3D: {ntk_tensor.shape}. Taking slice [..., 0]."
            )
            return ntk_tensor[..., 0]
        else:
            raise ValueError(
                f"Cannot process NTK of shape {ntk_tensor.shape} for 2D analysis."
            )

    def get_ntk_spectrum(
        self, x: jnp.ndarray, method: str = "trace", **apply_fn_kwargs
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Computes eigenvalues and eigenvectors of the (processed) NTK matrix.
        """
        ntk_matrix_2d = self._prepare_2d_ntk_for_analysis(x, method, **apply_fn_kwargs)
        if ntk_matrix_2d.shape[0] != ntk_matrix_2d.shape[1]:
            raise ValueError(
                f"Matrix for eigendecomposition must be square, got {ntk_matrix_2d.shape}"
            )
        eigenvalues, eigenvectors = jnp.linalg.eigh(ntk_matrix_2d)
        return eigenvalues, eigenvectors

    def get_ntk_diagonal(self, x: jnp.ndarray, **apply_fn_kwargs) -> jnp.ndarray:
        """
        Computes and returns the diagonal of the NTK matrix K(x, x).
        Each element is K(x_i, x_i) (potentially traced over output dimensions if multi-output).
        """
        ntk_full_matrix = self.compute_ntk_dataset(x, None, **apply_fn_kwargs)  # K(x,x)

        if ntk_full_matrix.ndim == 2:  # Scalar output case (N, N)
            return jnp.diag(ntk_full_matrix)
        elif ntk_full_matrix.ndim == 4:  # Multi-output case (N, N, D_out, D_out)
            # Returns sum_k K_iikk (diagonal of the "traced" N x N matrix)
            ntk_traced_for_diag_sum = jnp.trace(
                ntk_full_matrix, axis1=-2, axis2=-1
            )  # (N, N)
            return jnp.diag(ntk_traced_for_diag_sum)
        elif ntk_full_matrix.ndim == 3:
            # Assumes (N, N, D_out), diagonal of the first channel's K(x_i, x_i)
            return jnp.diag(ntk_full_matrix[..., 0])
        else:
            raise ValueError(
                f"Cannot compute diagonal for NTK of shape {ntk_full_matrix.shape}"
            )

    def get_ntk_diagonal_per_output(
        self, x: jnp.ndarray, **apply_fn_kwargs
    ) -> Optional[jnp.ndarray]:
        """
        For multi-output networks (NTK shape N, N, D_out, D_out),
        returns an (N, D_out) array where element (i, k) is K(x_i, x_i)_{kk}.
        """
        ntk_full_matrix = self.compute_ntk_dataset(x, None, **apply_fn_kwargs)  # K(x,x)
        if ntk_full_matrix.ndim == 4:
            # K_ii(k,k) can be obtained by taking the diagonal w.r.t sample axes (0 and 1)
            # then taking the diagonal w.r.t output axes (now 0 and 1 of the result)
            # Step 1: Get diagonal blocks K(x_i, x_i) which are (D_out, D_out) matrices. Result shape (N, D_out, D_out)
            diag_blocks_ii = jnp.diagonal(
                ntk_full_matrix, axis1=0, axis2=1
            ).T  # Transpose to make N the first axis
            # Step 2: Get diagonal elements K_ii(k,k) from these blocks. Result shape (N, D_out)
            return jnp.diagonal(diag_blocks_ii, axis1=1, axis2=2)
        elif ntk_full_matrix.ndim == 2:  # Scalar output
            return jnp.diag(ntk_full_matrix)[:, None]  # Add D_out=1 dimension
        else:
            print(
                f"get_ntk_diagonal_per_output is designed for NTK of shape (N,N,D_out,D_out) or (N,N). Got {ntk_full_matrix.shape}"
            )
            return None

    def get_ntk_trace(
        self, x: jnp.ndarray, method: str = "trace", **apply_fn_kwargs
    ) -> float:
        """Computes the trace of the (processed) NTK matrix K(x,x)."""
        ntk_matrix_2d = self._prepare_2d_ntk_for_analysis(x, method, **apply_fn_kwargs)
        return jnp.trace(ntk_matrix_2d)

    def get_ntk_condition_number(
        self,
        x: jnp.ndarray,
        method: str = "trace",
        epsilon_rank_check: float = 1e-9,  # To avoid division by zero for singular matrices
        **apply_fn_kwargs,
    ) -> float:
        """Computes the condition number of the (processed) NTK matrix K(x,x)."""
        eigenvalues, _ = self.get_ntk_spectrum(x, method, **apply_fn_kwargs)
        # Filter out very small eigenvalues that might be due to numerical precision
        # and could lead to an artificially large or infinite condition number
        # Only consider eigenvalues that are a significant fraction of the max eigenvalue
        abs_eigenvalues = jnp.abs(eigenvalues)
        max_eig = jnp.max(abs_eigenvalues)
        min_eig_nz = jnp.min(
            abs_eigenvalues[abs_eigenvalues > epsilon_rank_check * max_eig]
        )

        if (
            min_eig_nz <= 0 or max_eig <= 0
        ):  # Or if no eigenvalues > threshold * max_eig
            print(
                f"Warning: Could not compute valid condition number. Min non-zero eig: {min_eig_nz}, Max eig: {max_eig}"
            )
            return jnp.inf

        return max_eig / min_eig_nz

    def compute_ntk_change(
        self,
        new_params: Any,
        x: jnp.ndarray,
        norm_type: Union[str, int, float] = "fro",  # As per jnp.linalg.norm
        **apply_fn_kwargs,
    ) -> float:
        """
        Computes the change between the NTK with current self.params and NTK with new_params.
        Note: This method updates self.params to new_params after computing the change.
              If you don't want to update self.params, pass params_override to compute_ntk_dataset.
        """
        ntk_t0 = self.compute_ntk_dataset(
            x, None, **apply_fn_kwargs
        )  # Uses self.params
        ntk_t1 = self.compute_ntk_dataset(
            x, None, params_override=new_params, **apply_fn_kwargs
        )

        if ntk_t0.shape != ntk_t1.shape:
            raise ValueError(f"NTK shapes mismatch: {ntk_t0.shape} vs {ntk_t1.shape}")

        # Ensure they are at least 2D for norm calculation if they are higher order
        # (e.g. for multi-output, compare the full (N,N,D_out,D_out) tensors)
        diff_norm = jnp.linalg.norm(
            (ntk_t1 - ntk_t0).reshape(-1), ord=self._map_norm_type_to_ord(norm_type)
        )

        return diff_norm

    def _map_norm_type_to_ord(
        self, norm_type: Union[str, int, float]
    ) -> Optional[Union[int, float, str]]:
        if isinstance(norm_type, (int, float)):
            # These are valid for vector norms (e.g., 1 for L1, 2 for L2, jnp.inf)
            return norm_type
        if isinstance(norm_type, str):
            norm_type_low = norm_type.lower()
            # For flattened vectors, Frobenius norm is equivalent to L2 norm.
            if norm_type_low == "fro" or norm_type_low == "frobenius":
                return 2
            # 'nuc' (nuclear) is not for vectors.
            if norm_type_low == "nuc":
                print(
                    f"Warning: Nuclear norm ('nuc') is for matrices. Using L2 for flattened vector as fallback."
                )
                return 2  # Fallback to L2 for vectorized nuc
            try:  # Try to convert '1', '2' etc. to int for vector p-norms
                return int(norm_type_low)
            except ValueError:
                if norm_type_low == "inf":
                    return jnp.inf
                elif norm_type_low == "-inf":  # Max norm for vectors
                    return -jnp.inf
        # Default for unknown string types or other unhandled types when norming a vector
        print(
            f"Warning: Unknown norm type '{norm_type}' for vector, defaulting to L2 (ord=2)."
        )
        return 2
