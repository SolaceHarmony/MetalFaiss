"""
Orthogonality helpers (MLX): columns/rows, completion, initializer, and blocked variant.

These are production-safe wrappers around our QR implementation, with
straightforward MLX ops for completion and blocked orthogonalization.
"""

from __future__ import annotations
from typing import Tuple
import mlx.core as mx

from .qr import pure_mlx_qr


def orthonormal_columns(X: mx.array) -> mx.array:
    """Return a matrix with orthonormal columns spanning cols(X)."""
    Q, _ = pure_mlx_qr(X)
    return Q[:, : int(X.shape[1])]


def orthonormal_rows(X: mx.array) -> mx.array:
    """Return a matrix with orthonormal rows spanning rows(X)."""
    Qt, _ = pure_mlx_qr(mx.transpose(X))
    return mx.transpose(Qt[:, : int(X.shape[0])])


def complete_basis(Q: mx.array) -> mx.array:
    """Complete Q (m,r) with m-r additional orthonormal columns to (m,m)."""
    m, r = int(Q.shape[0]), int(Q.shape[1])
    k = m - r
    if k == 0:
        return Q
    R = Q
    for _ in range(k):
        v = mx.random.normal(shape=(m,), dtype=R.dtype)
        # two‑pass MGS projection
        c1 = mx.matmul(mx.transpose(R), v)
        v = v - mx.matmul(R, c1)
        c2 = mx.matmul(mx.transpose(R), v)
        v = v - mx.matmul(R, c2)
        nrm = mx.sqrt(mx.sum(mx.square(v)))
        denom = mx.where(nrm > 0, nrm, mx.ones_like(nrm))
        u = mx.divide(v, denom)
        R = mx.concatenate([R, u.reshape((m, 1))], axis=1)
    return R


def orthogonal(shape, gain: float = 1.0) -> mx.array:
    """Orthogonal initializer for any shape (QR of a square random, sliced)."""
    if isinstance(shape, mx.array):
        shp = tuple(int(d.item() if hasattr(d, "item") else d) for d in shape)
    else:
        shp = tuple(int(d) for d in shape)
    if len(shp) < 2:
        raise ValueError("Shape must have at least 2 dims")
    rows = shp[0]
    cols = 1
    for d in shp[1:]:
        cols *= d
    size = max(rows, cols)
    W = mx.random.normal(shape=(size, size)).astype(mx.float32)
    Q, _ = pure_mlx_qr(W)
    Qblock = Q[:rows, :cols]
    g = mx.array(gain, dtype=Qblock.dtype)
    return mx.multiply(g, Qblock.reshape(shp))


def orthogonalize_blocked(A: mx.array, B: int = 32) -> mx.array:
    """Blocked orthogonalization for tall/wide matrices using two‑pass MGS per block."""
    m, n = int(A.shape[0]), int(A.shape[1])
    Q = A
    for b in range(0, n, B):
        e = min(b + B, n)
        for j in range(b, e):
            v = Q[:, j]
            if j > b:
                Qb = Q[:, b:j]
                c1 = mx.matmul(mx.transpose(Qb), v)
                v = v - mx.matmul(Qb, c1)
                c2 = mx.matmul(mx.transpose(Qb), v)
                v = v - mx.matmul(Qb, c2)
            if b > 0:
                Qprev = Q[:, :b]
                c1 = mx.matmul(mx.transpose(Qprev), v)
                v = v - mx.matmul(Qprev, c1)
                c2 = mx.matmul(mx.transpose(Qprev), v)
                v = v - mx.matmul(Qprev, c2)
            nrm = mx.sqrt(mx.sum(mx.square(v)))
            denom = mx.where(nrm > 0, nrm, mx.ones_like(nrm))
            Q[:, j] = mx.divide(v, denom)
    return Q
