# Non‑Square Orthogonality (Left/Right + Completion)

Orthogonality isn’t just for square matrices. This note documents robust patterns for semi‑orthogonal matrices and completion to a full orthonormal basis, using MLX and GPU‑friendly kernels.

## Definitions

- Left‑orthonormal (orthonormal columns): Q ∈ R^{m×n}, m ≥ n, Q^T Q = I_n
- Right‑orthonormal (orthonormal rows): Q ∈ R^{m×n}, n ≥ m, Q Q^T = I_m

## Orthonormal Columns (Left)

```python
from metalfaiss.faissmlx.qr import pure_mlx_qr

def orthonormal_columns(X: mx.array) -> mx.array:
    # Two‑pass MGS via MLX QR builds Q with Q^T Q = I
    Q, _ = pure_mlx_qr(X)
    return Q[:, : X.shape[1]]
```

## Orthonormal Rows (Right)

```python
def orthonormal_rows(X: mx.array) -> mx.array:
    # Orthonormalize columns of X^T, then transpose back
    Qt, _ = pure_mlx_qr(mx.transpose(X))
    return mx.transpose(Qt[:, : X.shape[0]])
```

## Completing to a Full Basis

Append k = m − r new orthonormal columns to Q ∈ R^{m×r}:

```python
def complete_basis(Q: mx.array) -> mx.array:
    m, r = int(Q.shape[0]), int(Q.shape[1])
    k = m - r
    if k == 0:
        return Q
    R = Q
    for _ in range(k):
        v = mx.random.normal(shape=(m,), dtype=R.dtype)
        # two‑pass MGS projection
        c1 = mx.matmul(mx.transpose(R), v)
        v  = v - mx.matmul(R, c1)
        c2 = mx.matmul(mx.transpose(R), v)
        v  = v - mx.matmul(R, c2)
        nrm = mx.sqrt(mx.sum(v*v))
        u = v / mx.where(nrm > 0, nrm, 1)
        R = mx.concatenate([R, u.reshape((m, 1))], axis=1)
    return R
```

## Orthogonal Initializer (Any Shape)

This pattern initializes an orthogonal matrix for arbitrary shapes by building a square Q and slicing. It mirrors robust initializers used in ML frameworks.

```python
import mlx.core as mx
from metalfaiss.faissmlx.qr import pure_mlx_qr

def orthogonal(shape, gain: float = 1.0) -> mx.array:
    # shape can be a tuple or an MLX array of dims
    if isinstance(shape, mx.array):
        shp = tuple(int(d.item() if hasattr(d, 'item') else d) for d in shape)
    else:
        shp = tuple(int(d) for d in shape)
    if len(shp) < 2:
        raise ValueError("Shape must have at least 2 dims")

    rows, cols = shp[0], int(mx.prod(mx.array(shp[1:])).item())
    size = max(rows, cols)

    # Random square matrix, QR -> Q is square orthogonal
    W = mx.random.normal(shape=(size, size)).astype(mx.float32)
    Q, _ = pure_mlx_qr(W)  # Q: (size, size)

    # Take the leading block and reshape to target
    Qblock = Q[:rows, :cols]
    return gain * Qblock.reshape(shp)
```

Why it works
- QR of a square random matrix gives a numerically stable orthogonal Q. Slicing gives left/right semi‑orthogonality by construction.

## Block‑Based Non‑Square Orthogonalization (Large n)

For very wide/tall matrices, orthogonalize columns in blocks. Two‑pass MGS within a block, orthogonalize against previous blocks, and renormalize. Use compensated sums for norms/dots when needed.

Pseudocode (block columns of size B):

```python
def orthogonalize_blocked(A: mx.array, B: int = 32) -> mx.array:
    m, n = int(A.shape[0]), int(A.shape[1])
    Q = A
    for b in range(0, n, B):
        e = min(b + B, n)
        # Within‑block two‑pass MGS
        for j in range(b, e):
            v = Q[:, j]
            if j > b:
                Qb = Q[:, b:j]
                c1 = mx.matmul(mx.transpose(Qb), v)
                v  = v - mx.matmul(Qb, c1)
                c2 = mx.matmul(mx.transpose(Qb), v)
                v  = v - mx.matmul(Qb, c2)
            # Orthogonalize vs previous blocks
            if b > 0:
                Qprev = Q[:, :b]
                c1 = mx.matmul(mx.transpose(Qprev), v)
                v  = v - mx.matmul(Qprev, c1)
                c2 = mx.matmul(mx.transpose(Qprev), v)
                v  = v - mx.matmul(Qprev, c2)
            # Renormalize
            nrm = mx.sqrt(mx.sum(v * v))
            Q[:, j] = v / mx.where(nrm > 0, nrm, 1)
    return Q
```

Compensated sums (Kahan‑style) for norms/dots

```python
def kahan_sum_squares(x: mx.array) -> mx.array:
    hi = mx.zeros((), dtype=x.dtype)
    lo = mx.zeros((), dtype=x.dtype)
    for i in range(int(x.shape[0])):
        v = x[i] * x[i]
        t = hi + v
        e = (hi - t) + v
        hi = t
        lo = lo + e
    return hi + lo

def kahan_dot(x: mx.array, y: mx.array) -> mx.array:
    hi = mx.zeros((), dtype=x.dtype)
    lo = mx.zeros((), dtype=x.dtype)
    for i in range(int(x.shape[0])):
        v = x[i] * y[i]
        t = hi + v
        e = (hi - t) + v
        hi = t
        lo = lo + e
    return hi + lo
```

When to use
- Very ill‑conditioned data or long columns where naive reductions drift; otherwise, the standard two‑pass MGS is usually sufficient.

## GPU Notes

- Use the QR projection kernel (c = Q^T v) for large m,k to speed up re‑orthonormalization.
- Consider HPC16x8 limb accumulation for projections and norms when drift appears.
- Random rotations for non‑square transforms:
  - If d_in ≥ d_out: take first d_out columns of a left‑orthonormal Q.
  - If d_out > d_in: build right‑orthonormal rows and transpose.
