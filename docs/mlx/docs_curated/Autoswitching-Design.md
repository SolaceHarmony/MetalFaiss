Autoswitching (Device + Size Aware)

Principles

- Prefer MLX vectorized ops for small/medium problems (no JIT overhead).
- Use Metal kernels for inner loops/tiles on larger shapes.
- Add limb (HPC16x8) paths when numeric drift is observed or shapes are “hard.”

Sketch

```python
def choose_qr_impl(m, k):
    if m*k < 256*1024:
        return "MLX_MGS"
    return "KERNEL_PROJ"

def choose_svd_impl(m, n, k):
    if m*n*k < 4*1024*1024:
        return "MLX_MATMUL"
    return "KERNEL_TILED"
```

Implementation in this repo

- `python/metalfaiss/faissmlx/dispatch.py` — heuristics + env overrides
- `python/metalfaiss/faissmlx/svd.py` — uses tiled kernels or MLX GEMM
- `python/metalfaiss/faissmlx/qr.py` — chooses kernel projections vs MLX

Attribution

Patterns adapted from Ember ML (The Solace Project), notably `eigen_ops.py` and `cholesky_ops.py`.

