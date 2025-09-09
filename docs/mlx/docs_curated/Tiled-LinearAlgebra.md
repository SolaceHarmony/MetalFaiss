Tiled Linear Algebra on MLX + Metal

Overview

For large matrices, stage tiles of inputs into threadgroup memory, synchronize with barriers, and accumulate with FMA. On Apple GPUs, choose tile sizes aligned to the execution width (32) and keep threads per threadgroup ≤ 1024.

QR: Two-Pass MGS (Stable at fp32)

Two-pass Modified Gram–Schmidt stabilizes fp32 by re-projecting after the first subtraction. Use a small kernel for projections `c = Qᵀv`, but keep vector updates in MLX:

```python
def mgs_two_pass(A):
    m, n = int(A.shape[0]), int(A.shape[1])
    K = min(m, n)
    Q = mx.zeros((m, m), dtype=A.dtype)
    R = mx.zeros((m, n), dtype=A.dtype)
    for k in range(K):
        v = A[:, k]
        if k > 0:
            Qk = Q[:, :k]
            c1 = mx.matmul(Qk.T, v)
            v  = v - mx.matmul(Qk, c1)
            c2 = mx.matmul(Qk.T, v)
            v  = v - mx.matmul(Qk, c2)
            R[:k, k] = c1 + c2
        rkk = mx.sqrt(mx.sum(v*v))
        qk  = v / mx.where(rkk > 0, rkk, 1.0)
        Q[:, k] = qk; R[k, k] = rkk
    return Q[:, :n], R
```

SVD: Subspace (Block) Power Iteration

Iteratively apply `Z = Aᵀ(A V)` and re-orthonormalize columns of `V`. MLX GEMM is strong; for large tiles, use two GEMM-like kernels:

1) `B = A × V` with tiles of A(m×n) and V(n×k)
2) `Z = Aᵀ × B` with tiles of Aᵀ(n×m) and B(m×k)

We provide tiled kernels in-repo:

- `python/metalfaiss/faissmlx/kernels/gemm_kernels.py`

Autoswitching

Prefer MLX matmul for small/medium problems; switch to tiled kernels by size/device. See `python/metalfaiss/faissmlx/dispatch.py` for heuristics and environment overrides.

Householder/Cholesky Insights

For block algorithms (Cholesky/QR), diagonal panels are numerically sensitive — keep them on fewer threads or single-thread; trailing updates are highly parallel. The same philosophy improves SVD tiling when computing `Aᵀ(A V)` in two steps.

References

- This repository: `python/metalfaiss/faissmlx/kernels/gemm_kernels.py`, `python/metalfaiss/faissmlx/svd.py`
- Ember ML: `.../cholesky_ops.py`, `.../svd_ops.py`, `.../qr_ops.py`

