# MLX Compile Guide for MetalFaiss

This guide documents how we use `mx.compile` in MetalFaiss to speed up repeated computations while keeping everything pure‑MLX/Metal. It also summarizes safe patterns, caveats, and concrete examples you can copy into new code.

Goals
- Reduce Python overhead by compiling hot functions (SVD iteration, QR drivers, small math helpers).
- Keep shapes/dtypes stable to reuse compiled graphs; avoid side effects during tracing.
- Mix CPU/GPU work in one compiled function when it helps overall throughput on Apple Silicon.

Key Rules (High‑Signal)
- Pure arrays only: no `.item()`, `.numpy()`, `.tolist()`, or Python `float()` pulls. Use MLX ops everywhere (`mx.divide`, `mx.multiply`, `mx.square`, `mx.sqrt`, `mx.where`).
- Stable shapes/dtypes: changes trigger recompiles unless you use `shapeless=True`. If you opt into shapeless, avoid shape‑dependent logic captured at trace time; prefer shape‑agnostic ops (e.g., `x.flatten(0,1)` instead of `x.reshape(x.shape[0]*x.shape[1], -1)`).
- No prints or host I/O in compiled functions. Return arrays you want to inspect.
- RNG: don’t seed inside compiled functions; pass keys/state or generate inputs outside.

What We Compile
- SVD iteration (MLX path): one step `Z = Aᵀ(A V)` then `QR(Z)` (see `faissmlx/svd.py`). Cached by `(m,n,k,dtype,device)`.
- SVD kernel path: a compiled wrapper around `gemm_av` → `gemm_at_b` (and banded variants) to cut Python overhead; kernels run unchanged.
- Future (optional): QR two‑pass MGS driver and IVF orchestration when shapes are stable.

Examples

1) Compile a math helper (elementwise fusion)
```python
import mlx.core as mx
def gelu(x):
    # Use MLX ops only — avoid Python math on arrays
    inv_rt2 = mx.divide(mx.ones_like(x), mx.sqrt(mx.array(2.0, dtype=x.dtype)))
    return mx.multiply(x, mx.divide(mx.add(mx.ones_like(x), mx.erf(mx.multiply(x, inv_rt2))), mx.array(2.0, dtype=x.dtype)))

gelu_c = mx.compile(gelu)
```

2) Compile SVD MLX step (already in `faissmlx/svd.py`)
```python
def _mlx_power_iter_once(A, V):
    AV = mx.matmul(A, V)
    Z = mx.matmul(mx.transpose(A), AV)
    Qz, _ = pure_mlx_qr(Z)
    return Qz

# Cached compiled version keyed by (m,n,k,dtype,device)
compiled_once = mx.compile(_mlx_power_iter_once)
```

3) Compile a kernel wrapper (AV then AᵀB)
```python
def zstep_kernel(A, V):
    B = gemm_av(A, V)
    return gemm_at_b(A, B)

zstep_kernel_c = mx.compile(zstep_kernel)
```

4) Mixed CPU/GPU inside compile (unified memory)
```python
@mx.compile
def mixed_step(A, V):
    B = mx.matmul(A, V, stream=mx.gpu)  # heavy on GPU
    # cheap guards on CPU
    eps = mx.array(1e-6, dtype=V.dtype)
    Vg = mx.where(mx.abs(V, stream=mx.cpu) < eps, mx.zeros_like(V), V)
    return gemm_at_b(A, B), Vg
```

Shapeless Mode (advanced)
- `mx.compile(fun, shapeless=True)` compiles once for variable shapes. Use only when your function is shape‑agnostic. Favor APIs like `flatten(start,end)` over `reshape` driven by cached shapes.

Benchmarks (this repo)
- SVD MLX path (512×256, k=32, iters=3): ~1.6× speedup with compile. Kernel path wrapper ≈ parity (kernels dominate runtime).

Pitfalls
- Python control flow that depends on array values is evaluated during tracing; avoid it or move the branch into MLX with `mx.where`/masks.
- Recompiles on any of: dtype change, rank change, shape change (unless shapeless), argument count change.
- Don’t create/destroy compiled lambdas in inner loops; compile once and reuse.

Integration Points in Code
- See `python/metalfaiss/faissmlx/svd.py` for compiled MLX and kernel wrapper paths.
- Bench harness adds `svd_compile.csv` to quantify compile wins (MLX vs kernel wrappers).

References
- MLX compile user guide and API docs (local curated docs and upstream MLX site).

