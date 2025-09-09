Curator's note: Prefer the human-authored MLX guides for clarity.
- ../docs_curated/README.md
- ../docs_curated/PYTORCH_DISSONANCE.md
- ../docs_curated/NUMPY_USERS.md
- ../docs_curated/COMMON_PITFALLS.md

<!--
Per-file analysis (linalg.md):
- API table for linalg; mirrors our curated LINALG guide.
- Add quick best practices to steer away from explicit inverses and toward solves.
-->

## Curated Notes

- Prefer `mx.linalg.solve` over `inv(A) @ b` for stability and performance.
- Add small diagonal jitter when decomposing nearly singular matrices: `A @ A.T + 1e-3 * mx.eye(n)`.
- When batching solves, broadcast shapes explicitly and check that leading dimensions align.
- No `device=` argument: control placement via default device or per‑op `stream` when supported.
  - Global: `mx.set_default_device(mx.cpu)` (or `mx.gpu`)
  - Scoped: `with mx.default_device(mx.cpu): U, S, Vt = mx.linalg.svd(A)`
  - Per‑op: some linalg ops accept `stream=mx.cpu`

### Cholesky on Metal (Tiled Pattern)

- Built‑in `mx.linalg.cholesky` often runs on CPU; for GPU, a custom tiled kernel can deliver speedups on large, well‑conditioned matrices.
- Strategy: compute the diagonal block with a numerically stable loop (single thread), then update trailing blocks in parallel; synchronize with `threadgroup_barrier` between phases.

```python
import mlx.core as mx

def block_cholesky(A, block_size=16):
    # See Custom Metal Kernels for a full kernel; this shows the call pattern
    nthreads = min(32, A.shape[0])
    k = mx.fast.metal_kernel(
        name="blk_chol", input_names=["A","blk","nth"], output_names=["out"],
        source="/* body omitted for brevity; diagonal then trailing updates with barriers */",
        header="#include <metal_stdlib>\nusing namespace metal;\n",
    )
    return k(
        inputs=[A, mx.array([block_size], dtype=mx.uint32), mx.array([nthreads], dtype=mx.uint32)],
        output_shapes=[A.shape], output_dtypes=[A.dtype],
        grid=(nthreads,1,1), threadgroup=(nthreads,1,1),
    )[0]

# Verify correctness against A ≈ L @ L.T
A = mx.random.normal((256, 256)); A = (A + A.T)/2 + 1e-1*mx.eye(256)
L = block_cholesky(A, block_size=16)
ok = mx.allclose(A, L @ L.T, rtol=1e-4, atol=1e-4)
```

Notes:
- Keep block sizes modest; prefer stability over maximal parallelism.
- Fall back to CPU `mx.linalg.cholesky` for ill‑conditioned matrices or when precision is critical.

### QR on Metal (Tiled + Guards)

- Built‑in `mx.linalg.qr` may run on CPU depending on dtype/backend; large GPU QR benefits from block/panel updates.
- Strategy: compute Householder transforms in panels, update trailing matrix in tiles; add stability guards and fallbacks.

```python
import mlx.core as mx

# Reduced QR example and validation
m, n = 512, 256
A = mx.random.normal((m, n))
Q, R = mx.linalg.qr(A)  # reduced by default if m >= n

# Orthogonality check: ||I - Q^T Q||_F / n
I = mx.eye(Q.shape[1])
orth_err = mx.linalg.norm(I - (Q.T @ Q)) / Q.shape[1]

# Triangularity check: lower part of R should be ~0
lower_mask = mx.tril(mx.ones_like(R), k=-1)
tri_err = mx.linalg.norm(lower_mask * R)

print("orth_err:", float(orth_err.item()), "tri_err:", float(tri_err.item()))

# CPU fallback with float64 for tough problems
with mx.default_device(mx.cpu):
    Q64, R64 = mx.linalg.qr(A.astype(mx.float64))
```

Notes and guards (for custom kernels):
- Use modest panel widths and threadgroup sizes; synchronize phases with `threadgroup_barrier`.
- Add diagonal jitter or pivot guards when norms are tiny; detect NaN/Inf and fall back to CPU.
- Re‑orthogonalize a panel if `||I - Q^T Q||` exceeds a threshold.
- Consider limb‑based accumulation (see HPC16x8) for inner products to maintain orthogonality on GPU.
- Thread planning (Apple GPUs): use execution width 32; cap threadgroup size ≤1024; launch ≤ one thread per element and round counts to a multiple of the execution width.

#### Limb‑based vᵀv accumulation (Metal pattern)

- For improved stability, accumulate `v^T v` in a wider fixed‑point format using 16‑bit limbs.
- Typical setup: `NUM_LIMBS = 8` with radix `2^16` gives a ~128‑bit accumulator across threads.

```c
// Each thread accumulates partial limbs for its slice
threadgroup uint thread_limbs[WARP_SIZE * NUM_LIMBS];
uint local_limb[NUM_LIMBS] = {0u};
for (uint i = k + tid; i < m; i += grid_sz) {
    uint bits = as_type<uint>(R_out[i*n + k]);
    ushort lo = bits & 0xFFFFu;
    ushort hi = (bits >> 16) & 0xFFFFu;
    uint p0 = uint(lo*lo), p1 = uint(hi*hi), pc = uint(lo*hi) << 1;
    local_limb[0] +=  p0 & 0xFFFFu;
    local_limb[1] += (p0 >> 16) + (pc & 0xFFFFu);
    local_limb[2] += (pc >> 16) + (p1 & 0xFFFFu);
    local_limb[3] +=  p1 >> 16;
}
// Write to shared, then reduce + carry propagate to get vtv as float
```

Notes:
- Reduce across threads, propagate carries, then convert limbs back to float via radix expansion.
- Use this for `v^T v` and optionally for dot products in reflector application when drift is observed.

#### Running the QR Metal kernel (wrapper)

```python
import signal, time
import mlx.core as mx

def run_qr_kernel(A: mx.array, kernel, debug=False, timeout_seconds=1.0):
    m, n = A.shape
    exec_width = 32
    tgroup = exec_width
    # one thread per element, conservatively capped
    total_elems = m*m + m*n + m*n
    max_total = 1024
    nthreads = min(max_total, total_elems)
    # round up to multiple of threadgroup
    nthreads = ((nthreads + tgroup - 1) // tgroup) * tgroup
    grid = (nthreads, 1, 1)
    threadgroup = (tgroup, 1, 1)

    # small timeout guard
    def timeout_handler(signum, frame):
        raise TimeoutError(f"QR kernel timed out after {timeout_seconds}s")
    orig = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(int(timeout_seconds))
    try:
        Q, R, dbg = kernel(
            inputs=[A],
            output_shapes=[(m, m), (m, n), (16,)],
            output_dtypes=[mx.float32, mx.float32, mx.float32],
            grid=grid, threadgroup=threadgroup, verbose=debug
        )
    finally:
        signal.alarm(0); signal.signal(signal.SIGALRM, orig)
    return Q, R, dbg
```

#### QR debug codes (dbg)

- `1`: matrix too large
- `2`: workload too large
- `3`: too many iterations
- `4`: numerical instability in norm calculation
- `5`: numerical instability in vᵀv calculation

#### Apple GPU sizing (heuristics)

- Execution width 32; threadgroup size ≤ 1024.
- Tiny (≤8): single threadgroup of 32.
- Small (≤32): ≈ one thread per element, cap ≤ 256; round to width.
- Medium+: cap total launched threads ≤ 1024 and round to width.

### SVD Notes (Shapes, Batching, Backend)

- Supports rectangular and batched inputs: for array shape `(..., m, n)`, SVD is applied to the last two dims for each leading batch index.
- Returns `U, S, Vt` when `compute_uv=True` and `S` only when `compute_uv=False`, satisfying `A = U @ diag(S) @ Vt`.
- If you hit backend errors on a given device, scope to CPU for SVD and continue the rest of the pipeline on GPU as needed.

### Examples

```python
import mlx.core as mx

# Solve vs inv(A) @ b
A = mx.random.normal((4, 4))
b = mx.random.normal((4,))
x = mx.linalg.solve(A, b)

# Batched solve: (B, M, M) @ (B, M)
Ab = mx.random.normal((8, 4, 4))
bb = mx.random.normal((8, 4))
xb = mx.linalg.solve(Ab, bb)

# SVD (optionally on CPU if your GPU backend errors)
with mx.default_device(mx.cpu):
    U, S, Vt = mx.linalg.svd(A)
print("square A shapes:", U.shape, S.shape, Vt.shape)

# Rectangular SVD
Arect = mx.random.normal((5, 3))
with mx.default_device(mx.cpu):
    U2, S2, Vt2 = mx.linalg.svd(Arect)
print("rect A shapes:", U2.shape, S2.shape, Vt2.shape)

# Singular values only
with mx.default_device(mx.cpu):
    S_only = mx.linalg.svd(Arect, compute_uv=False)
print("S only:", S_only.shape)

# Cholesky with jitter for PSD matrices
M = mx.random.normal((4, 4))
PSD = M @ M.T
L = mx.linalg.cholesky(PSD + 1e-4 * mx.eye(4))
```


<div id="main-content" class="bd-main" role="main">

<div class="sbt-scroll-pixel-helper">

</div>

<div class="bd-content">

<div class="bd-article-container">

<div class="bd-header-article d-print-none">

<div class="header-article-items header-article__inner">

<div class="header-article-items__start">

<div class="header-article-item">

<span class="fa-solid fa-bars"></span>

</div>

</div>

<div class="header-article-items__end">

<div class="header-article-item">

<div class="article-header-buttons">

<a href="https://github.com/ml-explore/mlx"
class="btn btn-sm btn-source-repository-button"
data-bs-placement="bottom" data-bs-toggle="tooltip" target="_blank"
title="Source repository"><span class="btn__icon-container"> <em></em>
</span></a>

<div class="dropdown dropdown-download-buttons">

- <a
  href="https://ml-explore.github.io/mlx/build/html/_sources/python/linalg.rst"
  class="btn btn-sm btn-download-source-button dropdown-item"
  data-bs-placement="left" data-bs-toggle="tooltip" target="_blank"
  title="Download source file"><span class="btn__icon-container">
  <em></em> </span> <span class="btn__text-container">.rst</span></a>
- <span class="btn__icon-container"> </span>
  <span class="btn__text-container">.pdf</span>

</div>

<span class="btn__icon-container"> </span>

</div>

</div>

</div>

</div>

</div>

<div id="jb-print-docs-body" class="onlyprint">

# Linear Algebra

<div id="print-main-content">

<div id="jb-print-toc">

</div>

</div>

</div>

<div id="searchbox">

</div>

<div id="linear-algebra" class="section">

<span id="linalg"></span>

# Linear Algebra<a href="https://ml-explore.github.io/mlx/build/html/#linear-algebra"
class="headerlink" title="Link to this heading">#</a>

<div class="pst-scrollable-table-container">

|  |  |
|----|----|
| <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.linalg.inv.html#mlx.core.linalg.inv"
class="reference internal" title="mlx.core.linalg.inv"><span
class="pre"><code class="sourceCode python">inv</code></span></a>(a, \*\[, stream\]) | Compute the inverse of a square matrix. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.linalg.tri_inv.html#mlx.core.linalg.tri_inv"
class="reference internal" title="mlx.core.linalg.tri_inv"><span
class="pre"><code class="sourceCode python">tri_inv</code></span></a>(a\[, upper, stream\]) | Compute the inverse of a triangular square matrix. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.linalg.norm.html#mlx.core.linalg.norm"
class="reference internal" title="mlx.core.linalg.norm"><span
class="pre"><code class="sourceCode python">norm</code></span></a>(a, /\[, ord, axis, keepdims, stream\]) | Matrix or vector norm. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.linalg.cholesky.html#mlx.core.linalg.cholesky"
class="reference internal" title="mlx.core.linalg.cholesky"><span
class="pre"><code class="sourceCode python">cholesky</code></span></a>(a\[, upper, stream\]) | Compute the Cholesky decomposition of a real symmetric positive semi-definite matrix. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.linalg.cholesky_inv.html#mlx.core.linalg.cholesky_inv"
class="reference internal" title="mlx.core.linalg.cholesky_inv"><span
class="pre"><code
class="sourceCode python">cholesky_inv</code></span></a>(L\[, upper, stream\]) | Compute the inverse of a real symmetric positive semi-definite matrix using it's Cholesky decomposition. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.linalg.cross.html#mlx.core.linalg.cross"
class="reference internal" title="mlx.core.linalg.cross"><span
class="pre"><code class="sourceCode python">cross</code></span></a>(a, b\[, axis, stream\]) | Compute the cross product of two arrays along a specified axis. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.linalg.qr.html#mlx.core.linalg.qr"
class="reference internal" title="mlx.core.linalg.qr"><span
class="pre"><code class="sourceCode python">qr</code></span></a>(a, \*\[, stream\]) | The QR factorization of the input matrix. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.linalg.svd.html#mlx.core.linalg.svd"
class="reference internal" title="mlx.core.linalg.svd"><span
class="pre"><code class="sourceCode python">svd</code></span></a>(a\[, compute_uv, stream\]) | The Singular Value Decomposition (SVD) of the input matrix. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.linalg.eigvalsh.html#mlx.core.linalg.eigvalsh"
class="reference internal" title="mlx.core.linalg.eigvalsh"><span
class="pre"><code class="sourceCode python">eigvalsh</code></span></a>(a\[, UPLO, stream\]) | Compute the eigenvalues of a complex Hermitian or real symmetric matrix. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.linalg.eigh.html#mlx.core.linalg.eigh"
class="reference internal" title="mlx.core.linalg.eigh"><span
class="pre"><code class="sourceCode python">eigh</code></span></a>(a\[, UPLO, stream\]) | Compute the eigenvalues and eigenvectors of a complex Hermitian or real symmetric matrix. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.linalg.lu.html#mlx.core.linalg.lu"
class="reference internal" title="mlx.core.linalg.lu"><span
class="pre"><code class="sourceCode python">lu</code></span></a>(a, \*\[, stream\]) | Compute the LU factorization of the given matrix <span class="pre">`A`</span>. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.linalg.lu_factor.html#mlx.core.linalg.lu_factor"
class="reference internal" title="mlx.core.linalg.lu_factor"><span
class="pre"><code class="sourceCode python">lu_factor</code></span></a>(a, \*\[, stream\]) | Computes a compact representation of the LU factorization. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.linalg.pinv.html#mlx.core.linalg.pinv"
class="reference internal" title="mlx.core.linalg.pinv"><span
class="pre"><code class="sourceCode python">pinv</code></span></a>(a, \*\[, stream\]) | Compute the (Moore-Penrose) pseudo-inverse of a matrix. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.linalg.solve.html#mlx.core.linalg.solve"
class="reference internal" title="mlx.core.linalg.solve"><span
class="pre"><code class="sourceCode python">solve</code></span></a>(a, b, \*\[, stream\]) | Compute the solution to a system of linear equations <span class="pre">`AX`</span>` `<span class="pre">`=`</span>` `<span class="pre">`B`</span>. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.linalg.solve_triangular.html#mlx.core.linalg.solve_triangular"
class="reference internal"
title="mlx.core.linalg.solve_triangular"><span class="pre"><code
class="sourceCode python">solve_triangular</code></span></a>(a, b, \*\[, upper, stream\]) | Computes the solution of a triangular system of linear equations <span class="pre">`AX`</span>` `<span class="pre">`=`</span>` `<span class="pre">`B`</span>. |

</div>

</div>

<div class="prev-next-area">

<a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fft.irfftn.html"
class="left-prev" title="previous page"><em></em></a>

<div class="prev-next-info">

previous

mlx.core.fft.irfftn

</div>

<a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.linalg.inv.html"
class="right-next" title="next page"></a>

<div class="prev-next-info">

next

mlx.core.linalg.inv

</div>

</div>

</div>

</div>

<div class="bd-footer-content__inner container">

<div class="footer-item">

By MLX Contributors

</div>

<div class="footer-item">

© Copyright 2023, MLX Contributors.  

</div>

<div class="footer-item">

</div>

<div class="footer-item">

</div>

</div>

</div>
