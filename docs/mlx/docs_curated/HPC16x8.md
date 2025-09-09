# Extended Precision (Double-Double) in MLX: HPC16x8 Pattern

MLX runs best in float32/bfloat16 on GPU. When you need more numerical headroom on GPU, a limb‑based (double‑double) technique can emulate higher precision by carrying a high and a low component per value. This page summarizes a practical pattern (HPC16x8) used for QR/eig/orthogonalization.

## Concept

- Represent a value `x` as `x ≈ high + low`, where both are float32 arrays.
- Implement extended‑precision primitives (add/mul) to propagate a compensated error term (`low`).
- Convert back to float32 results as needed.
- Fixed‑point limb pattern: accumulate with 16‑bit limbs using radix 2^16; `NUM_LIMBS=8` yields ~128‑bit headroom.

## Minimal API (example)

```python
import mlx.core as mx

class HPC16x8:
    def __init__(self, high: mx.array, low: mx.array | None = None):
        self.high = mx.array(high, dtype=mx.float32)
        self.low = mx.zeros_like(self.high) if low is None else mx.array(low, dtype=mx.float32)

    @classmethod
    def from_array(cls, arr):
        a = mx.array(arr, dtype=mx.float32)
        # split into limbs (toy split; see your implementation for robust splitting)
        high = mx.array(a, dtype=mx.float32)
        low = a - high
        return cls(high, low)

    def to_float32(self) -> mx.array:
        # compensated add
        s = self.high + self.low
        v = s - self.high
        e = (self.high - (s - v)) + (self.low - v)
        return s  # or return (s, e) if you need the residual
```

## Operations (examples)

- QR: compute on the float32 projection (`to_float32`), or integrate limb ops if needed.
- Eig (power iteration): accumulate multiplies with a limb‑aware update (e.g., `v_high = A @ q; v_low = (self.high @ q) - v_high; v = v_high + v_low`).
- Complete basis: use a robust orthogonalization of non‑square matrices (e.g., a Metal kernel or a stabilized CPU routine) and wrap back into HPC16x8.

```python
# Wrap a matrix and run eig (power iteration + orthogonalization approach)
H = HPC16x8.from_array(mx.random.normal((n, n)))
# Use your HPC16x8.eig()/qr()/complete_basis implementations here
w, V = H.eig()   # conceptual; see your concrete implementation
```

### QR with limb-aware dot products (sketch)

```python
def gram_schmidt_hpc(X: HPC16x8):
    # X.high: (m, n)
    Qh = []
    for j in range(X.high.shape[1]):
        v_high = X.high[:, j]
        v_low = X.low[:, j]
        for qh in Qh:
            # limb-aware dot: (qh^T v_high) in high limb; residual tracked in low limb (toy illustration)
            dot_high = qh @ v_high
            v_high = v_high - dot_high * qh
        # normalize in high limb
        nrm = mx.linalg.norm(v_high) + 1e-12
        Qh.append(v_high / nrm)
    Q_high = mx.stack(Qh, axis=1)
    return HPC16x8(Q_high, mx.zeros_like(Q_high))
```

Validate with `||I - Q^T Q||` and compare against a CPU float64 reference for small test cases.

## When to use

- Ill‑conditioned factorizations on GPU when CPU float64 is too slow or unavailable in your pipeline.
- Eigen/QR where float32 underflows/loses orthogonality; limb‑based accumulation can improve stability.

## Caveats

- Slower than native float32; profile and restrict to the smallest hot regions.
- Maintain clear boundaries: keep limb types local; convert to float32 at module edges.
- Verify numerics with `mx.allclose` and problem‑specific tolerances; add diagonal jitter to improve conditioning.

For a full implementation (QR, eig, orthogonalization), see the accompanying code in your project and consider combining with tiled Metal kernels for block updates.
