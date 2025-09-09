MLX fast.metal_kernel: Body-Only Source + Header Patterns

Why this matters

MLX’s `fast.metal_kernel` compiles Metal code at runtime. To make kernels actually compile and run reliably, use a header for includes and keep the kernel “body” functionless (no signature), passing parameters via MLX buffers. This avoids duplicate include errors, “expected expression” failures, and JIT churn from string-templating.

Rules that work in production

- Put `#include <metal_stdlib>` and `using namespace metal;` in `header=...`
- Keep `source=...` body-only: no `kernel void ...` signature, only statements.
- Pass shape/params via small MLX buffers (e.g., `uint[3] shape`).
- Set `ensure_row_contiguous=True` when linearizing like `row*cols + col`.
- Explicit `grid=(x,y,z)` and `threadgroup=(tx,ty,tz)` sizing; multiples of 32; ≤ 1024 threads/tg.
- Avoid compiling kernels per call. Build once, reuse; feed scalars/buffers for shapes.

Example: Column-Parallel Projection c = Qᵀ v

This is a small but useful helper for QR (Modified Gram–Schmidt): each thread accumulates one coefficient.

```python
import mlx.core as mx

header = """#include <metal_stdlib>\nusing namespace metal;\n"""
body = r"""
    uint gid = thread_position_in_grid.x;
    uint m = (uint)shape[0];
    uint k = (uint)shape[1];
    if (gid >= k) return;
    float acc = 0.0f;
    for (uint i = 0; i < m; ++i) {
        acc += Q[i * k + gid] * v[i];
    }
    out[gid] = acc;
"""

kernel = mx.fast.metal_kernel(
    name="qr_col_dot",
    input_names=["Q", "v", "shape"],
    output_names=["out"],
    header=header,
    source=body,
    ensure_row_contiguous=True,
)

def project_coeffs(Q, v):
    m, k = int(Q.shape[0]), int(Q.shape[1])
    shape = mx.array([m, k], dtype=mx.uint32)
    tgroup = 64
    total = k
    nthreads = ((total + tgroup - 1) // tgroup) * tgroup
    grid = (nthreads, 1, 1)
    tg = (tgroup, 1, 1)
    (out,) = kernel(
        inputs=[Q, v, shape],
        output_shapes=[(k,)],
        output_dtypes=[Q.dtype],
        grid=grid,
        threadgroup=tg,
    )
    return out
```

References

- This repository: `python/metalfaiss/faissmlx/kernels/qr_kernels.py`
- Ember ML source of truth: `ember_ml/backend/mlx/linearalg/qr_ops.py`

