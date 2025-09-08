"""
QR decomposition helpers (production: kernel-accelerated projections)

Exports
- `pure_mlx_qr`: Modified Gram–Schmidt QR using MLX ops with kernel projections.

Notes
- Production path always uses Metal kernels for projection/update steps
  (`qr_kernels.py`), as they measured fastest for tall matrices.

References
- docs/mlx/Comprehensive-MLX-Metal-Guide.md:1
- docs/metal/Shader-Optimization-Tips.md:7
"""

from typing import Tuple
import os
import mlx.core as mx
from .kernels.qr_kernels import project_coeffs, update_vector
from .device_guard import require_gpu
from .svd_dispatch import choose_qr_impl

def pure_mlx_qr(A: mx.array) -> Tuple[mx.array, mx.array]:
    """Modified Gram–Schmidt QR using only MLX ops (GPU‑resident).

    Parameters
    - `A (m,n)` (float32 preferred)

    Returns
    - `Q (m,m)`: first `n` columns form an orthonormal basis for `A`’s columns
    - `R (m,n)`: upper‑triangular

    Notes
    - Performs two‑pass re‑orthogonalization for stability.
    - Projection/update use kernel-accelerated paths.
    """
    # Enforce GPU usage unless explicitly allowed
    require_gpu("QR (pure_mlx_qr)")
    m, n = int(A.shape[0]), int(A.shape[1])
    K = min(m, n)
    Q = mx.zeros((m, m), dtype=A.dtype)
    R = mx.zeros((m, n), dtype=A.dtype)

    # Work on a copy to avoid modifying the input
    V = A
    impl = choose_qr_impl(m, K)
    for k in range(K):
        v = V[:, k]
        # Subtract projections onto previous Q columns (double re-orthogonalization)
        if k > 0:
            Qk = Q[:, :k]  # (m, k)
            if impl == "KERNEL_PROJ":
                # Kernel-accelerated projections
                c1 = project_coeffs(Qk, v)
                v = update_vector(Qk, c1, v)
                c2 = project_coeffs(Qk, v)
                v = update_vector(Qk, c2, v)
            else:
                # MLX projections: c = Qk^T v; v = v - Qk c (two-pass re-orthogonalization)
                c1 = mx.matmul(mx.transpose(Qk), v.reshape((m, 1))).reshape((k,))
                v = mx.subtract(v, mx.matmul(Qk, c1.reshape((k, 1))).reshape((m,)))
                c2 = mx.matmul(mx.transpose(Qk), v.reshape((m, 1))).reshape((k,))
                v = mx.subtract(v, mx.matmul(Qk, c2.reshape((k, 1))).reshape((m,)))
            R[:k, k] = mx.add(c1, c2)
        # Normalize
        rkk = mx.sqrt(mx.sum(mx.square(v)))
        # Avoid zero division; if zero, leave column as zeros
        ones = mx.ones_like(rkk)
        zeros = mx.zeros_like(rkk)
        inv = mx.where(rkk > 0, mx.divide(ones, rkk), zeros)
        qk = mx.multiply(v, inv)
        Q[:, k] = qk
        R[k, k] = rkk

    # Zero strictly lower part of R for cleanliness
    # Already ensured by construction; optional explicit mask skipped for performance
    return Q, R
