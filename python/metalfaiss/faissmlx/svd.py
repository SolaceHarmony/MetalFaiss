"""
svd.py - MLX-only tiled SVD via block power iteration

Provides a GPU-friendly top-k SVD using subspace (block) power iteration.
This uses only MLX ops (matmul, norms, our MLX QR) and keeps compute on GPU.
"""

from typing import Tuple
import mlx.core as mx
from .qr import pure_mlx_qr

def topk_svd(A: mx.array, k: int, iters: int = 8) -> Tuple[mx.array, mx.array, mx.array]:
    """Approximate top-k SVD of A using block power iteration.

    Args:
        A: (m, n) matrix
        k: number of leading singular vectors/values to compute (k <= min(m,n))
        iters: number of subspace iteration steps

    Returns:
        U: (m, k), S: (k,), Vt: (k, n)
    """
    m, n = int(A.shape[0]), int(A.shape[1])
    k = min(k, m, n)

    # Initialize V with random normal and orthonormalize
    V0 = mx.random.normal(shape=(n, k)).astype(A.dtype)
    Qv, _ = pure_mlx_qr(V0)
    V = Qv[:, :k]  # (n, k)

    # Subspace iteration: repeatedly apply (A^T A) to V and re-orthonormalize
    for _ in range(max(1, iters)):
        AV = mx.matmul(A, V)            # (m, k)
        Z = mx.matmul(mx.transpose(A), AV)  # (n, k)
        Qz, _ = pure_mlx_qr(Z)
        V = Qz[:, :k]

    # Ritz values/vectors: compute U = A V, then singular values as norms
    AV = mx.matmul(A, V)    # (m, k)
    # Column norms for singular values
    s = mx.sqrt(mx.sum(AV * AV, axis=0))  # (k,)
    # Avoid div by zero
    inv = mx.where(s > 0, 1.0 / s, 0.0)
    U = AV * inv.reshape((1, -1))
    Vt = mx.transpose(V)
    return U, s, Vt
