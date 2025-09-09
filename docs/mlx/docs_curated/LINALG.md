# Linear Algebra

High‑performance linear algebra kernels backed by Metal.

## Solves and Factorizations

```python
import mlx.core as mx

A = mx.random.normal((4, 4))
b = mx.random.normal((4,))
x = mx.linalg.solve(A, b)

L = mx.linalg.cholesky(A @ A.T + 1e-3 * mx.eye(4))
U, S, Vh = mx.linalg.svd(A)
Q, R = mx.linalg.qr(A)

pinv = mx.linalg.pinv(A)
```

Triangular tools: `solve_triangular`, `tri`, `triu`, `tril`, `tri_inv`.

## Norms and Utilities

```python
print(mx.linalg.norm(A))
print(mx.trace(A))
print(mx.tensordot(A, A, axes=1))
```

Eigen:

```python
w, V = mx.linalg.eigh(A @ A.T)
```

Numerical tips:

- Prefer well‑conditioned matrices; add small diagonal jitter for stability.
- Avoid inverting matrices explicitly; use `solve` when possible.

## For NumPy Users

- `numpy.linalg.solve` -> `mx.linalg.solve`
- `numpy.linalg.svd` -> `mx.linalg.svd`
- `numpy.linalg.pinv` -> `mx.linalg.pinv`
- `numpy.linalg.eigh`/`eigvalsh` -> `mx.linalg.eigh`/`mx.linalg.eigvalsh`
- `numpy.tensordot` -> `mx.tensordot`; `numpy.einsum` -> `mx.einsum`
