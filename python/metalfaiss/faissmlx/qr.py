"""
QR decomposition helpers (MLX + optional kernels)

Exports
- `pure_mlx_qr`: Modified Gram–Schmidt QR implemented with MLX ops (GPU‑resident).

Kernel Integration
- Projection/update steps can optionally use Metal kernels (`qr_kernels.py`) for
  speed when vectors are long or many columns are involved.
- Selection is controlled by heuristics (`dispatch.choose_qr_impl`) and env var:
  `METALFAISS_USE_QR_KERNEL=1` to force kernel projections.

References
- docs/mlx/Comprehensive-MLX-Metal-Guide.md:1
- docs/metal/Shader-Optimization-Tips.md:7
"""

from typing import Tuple
import os
import mlx.core as mx
from .kernels.qr_kernels import project_coeffs, update_vector
from .dispatch import choose_qr_impl

def pure_mlx_qr(A: mx.array) -> Tuple[mx.array, mx.array]:
    """Modified Gram–Schmidt QR using only MLX ops (GPU‑resident).

    Parameters
    - `A (m,n)` (float32 preferred)

    Returns
    - `Q (m,m)`: first `n` columns form an orthonormal basis for `A`’s columns
    - `R (m,n)`: upper‑triangular

    Notes
    - Performs two‑pass re‑orthogonalization for stability.
    - For `k>0` panel steps, projection/update can be done with Metal kernels
      when `METALFAISS_USE_QR_KERNEL=1` or heuristics select the kernel path.
    """
    m, n = int(A.shape[0]), int(A.shape[1])
    K = min(m, n)
    Q = mx.zeros((m, m), dtype=A.dtype)
    R = mx.zeros((m, n), dtype=A.dtype)

    # Work on a copy to avoid modifying the input
    V = A
    for k in range(K):
        v = V[:, k]
        # Subtract projections onto previous Q columns (double re-orthogonalization)
        if k > 0:
            Qk = Q[:, :k]                               # (m, k)
            impl = None
            if os.environ.get("METALFAISS_USE_QR_KERNEL", "0") == "1":
                impl = "KERNEL_PROJ"
            else:
                impl = choose_qr_impl(m, k)
            if impl == "KERNEL_PROJ":
                c1 = project_coeffs(Qk, v)
                v = update_vector(Qk, c1, v)
                c2 = project_coeffs(Qk, v)
                v = update_vector(Qk, c2, v)
            else:
                c1 = mx.matmul(mx.transpose(Qk), v)         # (k,)
                v = v - mx.matmul(Qk, c1)
                c2 = mx.matmul(mx.transpose(Qk), v)
                v = v - mx.matmul(Qk, c2)
            R[:k, k] = c1 + c2
        # Normalize
        rkk = mx.sqrt(mx.sum(v * v))
        # Avoid zero division; if zero, leave column as zeros
        inv = mx.where(rkk > 0, 1.0 / rkk, 0.0)
        qk = v * inv
        Q[:, k] = qk
        R[k, k] = rkk

    # Zero strictly lower part of R for cleanliness
    # Already ensured by construction; optional explicit mask skipped for performance
    return Q, R
