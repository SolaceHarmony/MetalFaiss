"""
Experimental: Port of Ember ML SVD power-iteration kernel for MLX fast.metal_kernel.

Notes
- This variant mirrors the Ember ML structure with a single-threadgroup launch
  by default (grid=(1,1,1), threadgroup=(1,1,1)) to validate compilation and
  basic behavior before parallelizing.
- It is not used in production; see svd.py (MLX path) for the stable SVD.
"""

from __future__ import annotations
from typing import Tuple
import mlx.core as mx

_HEADER = """#include <metal_stdlib>\nusing namespace metal;\n"""

_POWER_ITER_SRC = r"""
    // Inputs:
    // A: (n,n) symmetric (ATA or AAT)
    // Q_init: (n,k) initial orthonormal basis
    // shapeParams = [n, k]
    // iterParams  = [num_iterations]
    // tolParams   = [tolerance]

    // We keep a simple single-threaded structure here to validate correctness
    // before enabling parallelism and simdgroup reductions.

    uint n = shapeParams[0];
    uint k = shapeParams[1];
    uint num_iterations = iterParams[0];
    float tolerance = tolParams[0];

    // Copy Q_init -> Q_out
    for (uint idx = 0; idx < n * k; ++idx) {
        Q_out[idx] = Q_init[idx];
    }

    // Power iteration with Gram-Schmidt orthonormalization (two-pass per col)
    for (uint iter = 0; iter < num_iterations; ++iter) {
        // Z = A * Q_out
        threadgroup float Zshared[4096]; // guard: only for small tests
        // compute into Zshared then write back to Q_out after orthonormalization
        for (uint col = 0; col < k; ++col) {
            for (uint row = 0; row < n; ++row) {
                float sum = 0.0f;
                for (uint i = 0; i < n; ++i) {
                    sum = fma(A[row * n + i], Q_out[i * k + col], sum);
                }
                Zshared[row * k + col] = sum;
            }
        }
        // Orthonormalize columns of Zshared -> Q_out
        for (uint col = 0; col < k; ++col) {
            // Project out previous columns
            for (uint j = 0; j < col; ++j) {
                float proj = 0.0f;
                for (uint row = 0; row < n; ++row) {
                    proj = fma(Q_out[row * k + j], Zshared[row * k + col], proj);
                }
                // Z[:,col] -= proj * Q_out[:,j]
                for (uint row = 0; row < n; ++row) {
                    Zshared[row * k + col] -= proj * Q_out[row * k + j];
                }
            }
            // Normalize Z[:,col]
            float normsq = 0.0f;
            for (uint row = 0; row < n; ++row) {
                float v = Zshared[row * k + col];
                normsq = fma(v, v, normsq);
            }
            float norm = sqrt(normsq);
            float invn = (norm > tolerance) ? (1.0f / norm) : 0.0f;
            for (uint row = 0; row < n; ++row) {
                Q_out[row * k + col] = Zshared[row * k + col] * invn;
            }
        }
    }
"""

_KERNEL_POWER = mx.fast.metal_kernel(
    name="svd_power_iter_serial",
    input_names=["A", "Q_init", "shapeParams", "iterParams", "tolParams"],
    output_names=["Q_out"],
    header=_HEADER,
    source=_POWER_ITER_SRC,
    ensure_row_contiguous=True,
)


def power_iter(A: mx.array, Q_init: mx.array, iters: int = 10, tol: float = 1e-10) -> mx.array:
    """Run a simple (serial) power iteration on symmetric A with initial Q.

    Args
    - A: (n,n)
    - Q_init: (n,k)
    - iters: iterations
    - tol: tolerance for normalization guard
    Returns
    - Q_out: (n,k)
    """
    n = int(A.shape[0])
    assert int(A.shape[1]) == n
    k = int(Q_init.shape[1])
    assert int(Q_init.shape[0]) == n
    shapeParams = mx.array([n, k], dtype=mx.uint32)
    iterParams = mx.array([int(iters)], dtype=mx.uint32)
    tolParams = mx.array([float(tol)], dtype=mx.float32)
    grid = (1, 1, 1)
    threadgroup = (1, 1, 1)
    (Q_out,) = _KERNEL_POWER(
        inputs=[A, Q_init, shapeParams, iterParams, tolParams],
        output_shapes=[(n, k)],
        output_dtypes=[A.dtype],
        grid=grid,
        threadgroup=threadgroup,
    )
    return Q_out


def test_small(n: int = 32, k: int = 8, iters: int = 5) -> Tuple[float, float]:
    """Run a small test: returns (orthogonality_error, rayleigh_err)."""
    mx.random.seed(0)
    A = mx.random.normal(shape=(n, n)).astype(mx.float32)
    # Make symmetric positive semidefinite
    A = mx.matmul(mx.transpose(A), A)
    Q0 = mx.random.normal(shape=(n, k)).astype(mx.float32)
    # Orthonormalize Q0 via MLX QR
    from ..qr import pure_mlx_qr
    Q_init, _ = pure_mlx_qr(Q0)
    Q_init = Q_init[:, :k]

    Q = power_iter(A, Q_init, iters=iters)
    # Check orthogonality: Q^T Q ~ I
    QtQ = mx.matmul(mx.transpose(Q), Q)
    I = mx.eye(k)
    # Rayleigh: R = Q^T A Q should be diagonal-ish
    R = mx.matmul(mx.transpose(Q), mx.matmul(A, Q))
    offdiag = mx.subtract(R, mx.diag(mx.diag(R)))
    mx.eval(QtQ, I, offdiag)
    orth_err = float(mx.max(mx.abs(mx.subtract(QtQ, I))).item())
    rayleigh_off = float(mx.max(mx.abs(offdiag)).item())
    return orth_err, rayleigh_off
