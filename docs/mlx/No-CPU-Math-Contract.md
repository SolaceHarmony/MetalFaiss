# No‑CPU‑Math Contract (Pure MLX Scalars)

Rule
- All numeric values participating in math on arrays must be MLX arrays, even
  when conceptually scalars (1, 0, eps, gains, norms, thresholds).

Why
- Keeps graphs fusable and device‑resident.
- Avoids accidental host pulls and dtype surprises.
- Plays nicely with `mx.compile` tracing and streams.

How
- Use MLX ops: `mx.square, mx.divide, mx.multiply, mx.where, mx.add, mx.subtract, mx.power, mx.sqrt`.
- Build MLX scalars explicitly: `mx.array(1.0, dtype=x.dtype)`, or `mx.ones_like(x)`, `mx.zeros_like(x)`.
- +inf without Python floats: `mx.divide(mx.ones((n,), dtype=mx.float32), mx.zeros((n,), dtype=mx.float32))`.
- Pass MLX shapes/flags/eps into kernels via small buffers (`shape`, `flags`, `eps`).
- Control parameters (e.g., k, nlist) can be Python ints until they enter math.

Forbidden in compute paths
- `float(…)`, `int(…)` on MLX arrays
- Python operators on arrays: `*`, `/`, `**` (use MLX ops)
- `.item()`, `.tolist()`, `.numpy()` in hot paths

Enforcement (dev tool)
- Run `python -m python.metalfaiss.tools.scan_pure_mlx` to list violations.

Related docs
- docs/mlx/Compile-Guide.md — compile + fusion patterns
- docs/mlx/Comprehensive-MLX-Metal-Guide.md — kernel contract and MLX usage
