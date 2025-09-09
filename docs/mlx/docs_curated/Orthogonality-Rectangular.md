Rectangular Orthogonality: Left/Right and Completion

Definitions

- Left-orthonormal columns: Q ∈ R^{m×n} with QᵀQ = Iₙ (tall-skinny)
- Right-orthonormal rows: Q ∈ R^{m×n} with QQᵀ = Iₘ (short-fat)

Left (columns) via Two-Pass MGS

```python
def orthonormal_columns(X):
    Q, _ = mgs_two_pass(X)
    return Q[:, :X.shape[1]]
```

Right (rows) via transpose trick

```python
def orthonormal_rows(X):
    Qt, _ = mgs_two_pass(mx.transpose(X))
    return mx.transpose(Qt[:, :X.shape[0]])
```

Completion to a full orthonormal basis

```python
def complete_basis(Q):
    m, r = int(Q.shape[0]), int(Q.shape[1])
    for _ in range(m - r):
        v = mx.random.normal(shape=(m,), dtype=Q.dtype)
        for __ in range(2):
            c = mx.matmul(mx.transpose(Q), v)
            v = v - mx.matmul(Q, c)
        nrm = mx.sqrt(mx.sum(v*v))
        u = v / mx.where(nrm > 0, nrm, 1.0)
        Q = mx.concatenate([Q, u.reshape((m, 1))], axis=1)
    return Q
```

When to use kernels and HPC limbs

- Kernel projections (`c = Qᵀv`) help when m and current k are large.
- Limb-accumulating dot/norm helps when numerics drift at fp32.

References

- This repository: `python/metalfaiss/faissmlx/qr.py`, `.../kernels/qr_kernels.py`
- Ember ML: `orthogonal_nonsquare.py`

