# MetalFaiss Compile Guide (Practical Patterns)

This short guide shows how we apply `mx.compile` in a pure‑MLX/Metal project to cut Python overhead and fuse MLX ops, with real numbers from our benches.

Key rules
- MLX only: use `mx.square/mx.divide/mx.multiply/mx.where/mx.add/mx.subtract/mx.power`; no Python math on arrays, no `.item()/.tolist()/.numpy()`.
- Stable shapes/dtypes for reuse; use `shapeless=True` only with shape‑agnostic code.
- Build constants as MLX scalars (`mx.array(1.0, dtype=...)`), +inf via `mx.divide(mx.ones(...), mx.zeros(...))`.

Fast paths
- SVD MLX step (one iteration):
```python
import mlx.core as mx
from metalfaiss.faissmlx.qr import pure_mlx_qr

def svd_step(A, V):
    AV = mx.matmul(A, V)
    Z  = mx.matmul(mx.transpose(A), AV)
    Qz, _ = pure_mlx_qr(Z)
    return Qz

svd_step_c = mx.compile(svd_step)
```
Bench (M‑series): 512×256, k=32, iters=3 → ~1.6× faster compiled (≈0.036s → ≈0.023s median).

- Kernel wrapper (or banded Z‑step):
```python
from metalfaiss.faissmlx.kernels.gemm_kernels import gemm_av, gemm_at_b

def zstep_kernel(A, V):
    return gemm_at_b(A, gemm_av(A, V))

zstep_kernel_c = mx.compile(zstep_kernel)
```
Note: ≈ parity vs non‑compiled when kernels dominate runtime; helps when dispatching many small bands.

Mixed CPU/GPU inside compile (unified memory)
```python
@mx.compile
def mixed(A, V):
    B = mx.matmul(A, V, stream=mx.gpu)   # heavy on GPU
    eps = mx.array(1e-6, dtype=V.dtype)
    Vg = mx.where(mx.abs(V, stream=mx.cpu) < eps, mx.zeros_like(V), V)
    return B, Vg
```
MLX inserts cross‑device dependencies automatically.

Shapeless compilation
- `mx.compile(fun, shapeless=True)` compiles once for variable shapes. Use only if code is shape‑agnostic; prefer `flatten(start,end)` over `reshape` with captured dims.

Tips
- Cache compiled functions by (m,n,k,dtype,device). Don’t rebuild them inside loops.
- Keep streams explicit for overlap, but compile the outer function so calls reuse the graph.

See also
- ../mlx.core.compile.md (API)
- ../compile.md (user guide; shapeless, pitfalls)
- ../function_transforms.md (compose with grad/value_and_grad)
