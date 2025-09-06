"""
qr.py - QR decomposition helpers for MLX

This module provides:
- pure_mlx_qr: Modified Gram–Schmidt QR implemented with MLX ops (runs on GPU).
"""

from typing import Tuple
import os
import mlx.core as mx
from .kernels.qr_kernels import project_coeffs

def pure_mlx_qr(A: mx.array) -> Tuple[mx.array, mx.array]:
    """QR via Modified Gram–Schmidt using only MLX ops.

    Args:
        A: (m, n) matrix (float32 preferred)

    Returns:
        Q: (m, m) with first n columns forming orthonormal basis of A's columns
        R: (m, n) upper-triangular
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
            if os.environ.get("METALFAISS_USE_QR_KERNEL", "0") == "1":
                c1 = project_coeffs(Qk, v)
                v = v - mx.matmul(Qk, c1)
                c2 = project_coeffs(Qk, v)
                v = v - mx.matmul(Qk, c2)
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
