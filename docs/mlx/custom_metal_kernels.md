Curator's note: Prefer the human-authored MLX guides for clarity.
- ../docs_curated/README.md
- ../docs_curated/PYTORCH_DISSONANCE.md
- ../docs_curated/NUMPY_USERS.md
- ../docs_curated/COMMON_PITFALLS.md

<!--
Per-file analysis (custom_metal_kernels.md):
- Sphinx-converted developer doc; advanced audience targeting Metal shader authors.
- Helpful additions: profiling pointers and ABI expectations (shapes/strides).
-->

## Curated Notes

- Start with a CPU reference and assert equality before moving work to a Metal kernel.
- Use `mx.metal.start_capture()` / `mx.metal.stop_capture()` to collect command buffers for inspection in Xcode.
- Respect array strides: many views are non‑contiguous; either operate with strides or call `mx.contiguous` upfront.
- Document kernel preconditions (dtype, layout) and validate them in a Python wrapper before dispatch.

## Patterns and Examples

### Simple Elementwise Kernel (row‑contiguous)

```python
import mlx.core as mx

def exp_elementwise(a: mx.array):
    src = """
        uint elem = thread_position_in_grid.x;
        T tmp = inp[elem];
        out[elem] = metal::exp(tmp);
    """

    kernel = mx.fast.metal_kernel(
        name="myexp",
        input_names=["inp"],
        output_names=["out"],
        source=src,
    )
    out = kernel(
        inputs=[a],
        template=[("T", a.dtype)],  # dtype template
        grid=(a.size, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[a.shape],
        output_dtypes=[a.dtype],
    )[0]
    return out

a = mx.random.normal((4, 16)).astype(mx.float16)
b = exp_elementwise(a)
assert mx.allclose(b, mx.exp(a))
```

Notes:
- Pass only the kernel body in `source`; MLX generates the full signature from inputs/outputs and template params.
- Use `template=[("T", ...)]` to parameterize dtype/ints/bools.
- Tune `grid` and `threadgroup` (threads launched = prod(grid)).
- Add `verbose=True` to the kernel call to print the generated code.

### Strided Inputs (no row‑contiguous copy)

```python
def exp_elementwise_strided(a: mx.array):
    src = """
        uint elem = thread_position_in_grid.x;
        // utils.h helpers are available automatically
        uint loc = elem_to_loc(elem, inp_shape, inp_strides, inp_ndim);
        T tmp = inp[loc];
        // outputs are row contiguous
        out[elem] = metal::exp(tmp);
    """
    k = mx.fast.metal_kernel(
        name="myexp_strided", input_names=["inp"], output_names=["out"], source=src
    )
    return k(
        inputs=[a],
        template=[("T", a.dtype)],
        grid=(a.size, 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[a.shape], output_dtypes=[a.dtype],
        ensure_row_contiguous=False,  # opt‑out of input copy
    )[0]

x = mx.random.normal((4, 16)).astype(mx.float16)[::2]  # make non‑contiguous
y = exp_elementwise_strided(x)
assert mx.allclose(y, mx.exp(x))
```

### Custom Function + VJP (backprop) Sketch

```python
@mx.custom_function
def myop(x):
    # forward via metal kernel (like above)
    ...
    return y

@myop.vjp
def myop_vjp(primals, cotangent, aux):
    x, = primals
    # fused backward via metal kernel
    k = mx.fast.metal_kernel(
        name="myop_grad",
        input_names=["x", "cotangent"],
        output_names=["x_grad"],
        source="""/* atomic updates to x_grad */""",
        atomic_outputs=True,
    )
    xg = k(
        inputs=[x, cotangent],
        template=[("T", x.dtype)],
        output_shapes=[x.shape],
        output_dtypes=[x.dtype],
        grid=(x.size, 1, 1), threadgroup=(256, 1, 1),
        init_value=0,
    )[0]
    return (xg,)
```

Tips:
- `atomic_outputs=True` and `init_value=0` enable safe accumulation across threads.
- Consider padding channels to simd‑group multiples when doing simd reductions.

### Tiled/Block Cholesky (pattern)

Block‑based factorizations are a good fit for GPU when you keep the numerics stable. A practical approach is:

1) Provide a numerically safe single‑thread kernel for the diagonal block (or run diagonal updates on CPU), then
2) Use multiple threads to update trailing blocks, with threadgroup barriers to synchronize phases.

```python
# Header (math helpers)
hdr = """
#include <metal_stdlib>
#include <metal_math>
using namespace metal;
"""

src = """
    // thread and problem sizing
    uint tid = thread_position_in_grid.x;
    uint n = A_shape[0];
    uint blk = block_param[0];      // block size
    uint nblk = (n + blk - 1) / blk;
    uint nthreads = thread_count[0];

    for (uint k = 0; k < nblk; ++k) {
        uint b0 = k * blk, b1 = min(b0 + blk, n);
        // 1) diagonal block with single thread for stability
        if (tid == 0) {
            for (uint j = b0; j < b1; ++j) {
                float s = 0.0f;
                for (uint p = 0; p < j; ++p) s += out[j*n + p] * out[j*n + p];
                float d = A[j*n + j] - s; d = d <= 1e-10f ? 1e-10f : d;
                out[j*n + j] = sqrt(d);
                for (uint i = j+1; i < b1; ++i) {
                    float acc = 0.0f;
                    for (uint p = 0; p < j; ++p) acc += out[i*n + p] * out[j*n + p];
                    float denom = out[j*n + j];
                    out[i*n + j] = denom > 1e-10f ? (A[i*n + j] - acc) / denom : 0.0f;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_device);
        // 2) trailing updates: distribute rows across threads
        for (uint row = tid; row < n; row += nthreads) {
            if (row >= b1) {
                for (uint j = b0; j < b1; ++j) {
                    float acc = 0.0f;
                    for (uint p = 0; p < j; ++p) acc += out[row*n + p] * out[j*n + p];
                    float denom = out[j*n + j];
                    out[row*n + j] = denom > 1e-10f ? (A[row*n + j] - acc) / denom : 0.0f;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_device);
    }
"""

kernel = mx.fast.metal_kernel(
    name="block_cholesky_kernel",
    input_names=["A", "block_param", "thread_count"],
    output_names=["out"],
    source=src, header=hdr, ensure_row_contiguous=True,
)

def block_cholesky(A, block_size=16):
    nthreads = min(32, A.shape[0])
    return kernel(
        inputs=[A,
                mx.array([block_size], dtype=mx.uint32),
                mx.array([nthreads], dtype=mx.uint32)],
        output_shapes=[A.shape], output_dtypes=[A.dtype],
        grid=(nthreads,1,1), threadgroup=(nthreads,1,1),
    )[0]
```

Notes:
- Use a small block size and limited threads for stability; benchmark for your matrices.
- Verify correctness via `mx.allclose(A, L @ L.T)` and fall back to CPU for edge cases.

### Panel/Block QR (sketch)

- Pattern: build Householder reflectors for a panel (single thread or warp‑sized group for stability), then tile the trailing matrix update across threads; sync with `threadgroup_barrier`.
- Guards: cap panel width, reject tiny norms (add jitter), check for NaN/Inf, optionally re‑orthogonalize.

```python
hdr = """
#include <metal_stdlib>
using namespace metal;
"""

src = R"METAL(
    uint tid = thread_position_in_grid.x;
    uint m = A_shape[0];
    uint n = A_shape[1];
    uint bw = panel_shape[0];  // panel width

    for (uint k = 0; k < min(m, n); k += bw) {
        uint k1 = min(k + bw, n);
        // 1) form Householder vectors for columns k..k1-1 (single thread for stability)
        if (tid == 0) {
            // compute v, tau for each column; apply to panel
            // add small jitter if column norm is tiny to avoid breakdown
        }
        threadgroup_barrier(mem_flags::mem_device);
        // 2) apply block of reflectors to trailing matrix in parallel
        for (uint col = k1 + tid; col < n; col += grid_size.x) {
            // compute W = V^T * A[:, col]; A[:, col] -= V * (tau .* W)
        }
        threadgroup_barrier(mem_flags::mem_device);
    }
)METAL";

kernel = mx.fast.metal_kernel(
    name="panel_qr_kernel",
    input_names=["A", "panel"], output_names=["Q", "R"],
    source=src, header=hdr, ensure_row_contiguous=True,
)
```

Tips:
- Keep numerical work in stable scalar loops; parallelize level‑3 style updates.
- Validate with `||I - Q^T Q||` and lower‑triangular residual on R; fall back to CPU on failure.
- Use HPC16x8 limb accumulation for dot products in challenging cases.

### Debug and Safety Flags (pattern)

- Maintain a small `dbg` output buffer to signal health and capture metrics without crashing the kernel.
- Example mapping (adapt to your kernel):
- `dbg[0]`: start flag (1.0 if preflight checks passed)
- `dbg[1..3]`: matrix dims `m, n, min(m,n)`
- `dbg[5..7]`: threads per group, total threads, work per thread
- `dbg[8..11]`: intermediate values (e.g., sigma, norms, vTv)
- `dbg[12]`: iteration count
- `dbg[13]`: error code (1=matrix too large, 2=work too large, 3=too many iterations, 4/5=numerical instability)
- `dbg[14]`: error value (context)
- `dbg[15]`: success flag set at the end if all checks passed

Pattern:
- Perform preflight checks (dims, workload), set `dbg[0]`, and early‑return on failure.
- Guard inner computations; if NaN/Inf, set `dbg[13]`/`dbg[14]` and exit cleanly.
- Prefer to return a flagged `dbg` rather than crashing; host code can fallback to CPU.

### Apple GPU sizing (cheat sheet)

- Execution width: 32; choose threadgroup sizes as multiples of 32.
- Max threads per threadgroup: 1024 (cap here or lower).
- Tiny matrices (≤8): use one 32‑thread group.
- Small (≤32): up to one thread per element, cap around 256.
- Larger: cap total launched threads ≤ 1024; round to execution width.

### Wrapper with timeout (reusable)

```python
import signal

def launch_with_timeout(callable_kernel, args, kwargs=None, timeout_seconds=1.0):
    kwargs = {} if kwargs is None else kwargs
    def handler(signum, frame):
        raise TimeoutError(f"Kernel timed out after {timeout_seconds}s")
    orig = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(int(timeout_seconds))
    try:
        return callable_kernel(*args, **kwargs)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, orig)
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
  href="https://ml-explore.github.io/mlx/build/html/_sources/dev/custom_metal_kernels.rst"
  class="btn btn-sm btn-download-source-button dropdown-item"
  data-bs-placement="left" data-bs-toggle="tooltip" target="_blank"
  title="Download source file"><span class="btn__icon-container">
  <em></em> </span> <span class="btn__text-container">.rst</span></a>
- <span class="btn__icon-container"> </span>
  <span class="btn__text-container">.pdf</span>

</div>

<span class="btn__icon-container"> </span>

<span class="fa-solid fa-list"></span>

</div>

</div>

</div>

</div>

</div>

<div id="jb-print-docs-body" class="onlyprint">

# Custom Metal Kernels

<div id="print-main-content">

<div id="jb-print-toc">

<div>

## Contents

</div>

- <a href="https://ml-explore.github.io/mlx/build/html/#simple-example"
  class="reference internal nav-link">Simple Example</a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#using-shape-strides"
  class="reference internal nav-link">Using Shape/Strides</a>
- <a href="https://ml-explore.github.io/mlx/build/html/#complex-example"
  class="reference internal nav-link">Complex Example</a>
- <a href="https://ml-explore.github.io/mlx/build/html/#grid-sample-vjp"
  class="reference internal nav-link">Grid Sample VJP</a>

</div>

</div>

</div>

<div id="searchbox">

</div>

<div id="custom-metal-kernels" class="section">

<span id="id1"></span>

# Custom Metal Kernels<a
href="https://ml-explore.github.io/mlx/build/html/#custom-metal-kernels"
class="headerlink" title="Link to this heading">#</a>

MLX supports writing custom Metal kernels through the Python and C++
APIs.

<div id="simple-example" class="section">

## Simple Example<a href="https://ml-explore.github.io/mlx/build/html/#simple-example"
class="headerlink" title="Link to this heading">#</a>

Let’s write a custom kernel that computes <span class="pre">`exp`</span>
elementwise:

<div class="highlight-python notranslate">

<div class="highlight">

    def exp_elementwise(a: mx.array):
        source = """
            uint elem = thread_position_in_grid.x;
            T tmp = inp[elem];
            out[elem] = metal::exp(tmp);
        """

        kernel = mx.fast.metal_kernel(
            name="myexp",
            input_names=["inp"],
            output_names=["out"],
            source=source,
        )
        outputs = kernel(
            inputs=[a],
            template=[("T", mx.float32)],
            grid=(a.size, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[a.shape],
            output_dtypes=[a.dtype],
        )
        return outputs[0]

    a = mx.random.normal(shape=(4, 16)).astype(mx.float16)
    b = exp_elementwise(a)
    assert mx.allclose(b, mx.exp(a))

</div>

</div>

<div class="admonition note">

Note

We are only required to pass the body of the Metal kernel in
<span class="pre">`source`</span>.

</div>

The full function signature will be generated using:

- The shapes/dtypes of <span class="pre">`inputs`</span>  
  In the above, <span class="pre">`a`</span> is an
  <span class="pre">`mx.array`</span> of type
  <span class="pre">`mx.float16`</span> and we pass it with the key
  <span class="pre">`inp`</span> so we will add
  <span class="pre">`const`</span>` `<span class="pre">`device`</span>` `<span class="pre">`float16_t*`</span>` `<span class="pre">`inp`</span>
  to the signature. <span class="pre">`inp_shape`</span>,
  <span class="pre">`inp_strides`</span> and
  <span class="pre">`inp_ndim`</span> are also added for convenience if
  they are present in <span class="pre">`source`</span>.

- The list of <span class="pre">`output_dtypes`</span>  
  In the above, <span class="pre">`out`</span> is an
  <span class="pre">`mx.array`</span> of type
  <span class="pre">`mx.float16`</span> so we add
  <span class="pre">`device`</span>` `<span class="pre">`float16_t*`</span>` `<span class="pre">`out`</span>.

- Template parameters passed using <span class="pre">`template`</span>  
  In the above,
  <span class="pre">`template=[("T",`</span>` `<span class="pre">`mx.float32)]`</span>
  adds a template of
  <span class="pre">`template`</span>` `<span class="pre">`<typename`</span>` `<span class="pre">`T>`</span>
  to the function and instantiates the template with
  <span class="pre">`custom_kernel_myexp_float<float>`</span>. Template
  parameters can be <span class="pre">`mx.core.Dtype`</span>,
  <span class="pre">`int`</span> or <span class="pre">`bool`</span>.

- Metal attributes used in <span class="pre">`source`</span> such as <span class="pre">`[[thread_position_in_grid]]`</span>  
  These will be added as function arguments. All the attributes defined
  in Table 5.8 of the <a
  href="https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf"
  class="reference external">Metal Shading Language Specification</a>
  are supported.

Putting this all together, the generated function signature for
<span class="pre">`myexp`</span> is as follows:

<div class="highlight-cpp notranslate">

<div class="highlight">

    template <typename T>
    [[kernel]] void custom_kernel_myexp_float(
      const device float16_t* inp [[buffer(0)]],
      device float16_t* out [[buffer(1)]],
      uint3 thread_position_in_grid [[thread_position_in_grid]]) {

            uint elem = thread_position_in_grid.x;
            T tmp = inp[elem];
            out[elem] = metal::exp(tmp);

    }

    template [[host_name("custom_kernel_myexp_float")]] [[kernel]] decltype(custom_kernel_myexp_float<float>) custom_kernel_myexp_float<float>;

</div>

</div>

Note: <span class="pre">`grid`</span> and
<span class="pre">`threadgroup`</span> are parameters to the Metal <a
href="https://developer.apple.com/documentation/metal/mtlcomputecommandencoder/2866532-dispatchthreads"
class="reference external">dispatchThreads</a> function. This means we
will launch <span class="pre">`mx.prod(grid)`</span> threads, subdivided
into <span class="pre">`threadgroup`</span> size threadgroups. For
optimal performance, each thread group dimension should be less than or
equal to the corresponding grid dimension.

Passing <span class="pre">`verbose=True`</span> to
<span class="pre">`mx.fast.metal_kernel.__call__`</span> will print the
generated code for debugging purposes.

</div>

<div id="using-shape-strides" class="section">

## Using Shape/Strides<a
href="https://ml-explore.github.io/mlx/build/html/#using-shape-strides"
class="headerlink" title="Link to this heading">#</a>

<span class="pre">`mx.fast.metal_kernel`</span> supports an argument
<span class="pre">`ensure_row_contiguous`</span> which is
<span class="pre">`True`</span> by default. This will copy the
<span class="pre">`mx.array`</span> inputs if needed before the kernel
is launched to ensure that the memory layout is row contiguous.
Generally this makes writing the kernel easier, since we don’t have to
worry about gaps or the ordering of the dims when indexing.

If we want to avoid this copy, <span class="pre">`metal_kernel`</span>
automatically passes <span class="pre">`a_shape`</span>,
<span class="pre">`a_strides`</span> and
<span class="pre">`a_ndim`</span> for each input array
<span class="pre">`a`</span> if any are present in
<span class="pre">`source`</span>. We can then use MLX’s built in
indexing utils to fetch the right elements for each thread.

Let’s convert <span class="pre">`myexp`</span> above to support
arbitrarily strided arrays without relying on a copy from
<span class="pre">`ensure_row_contiguous`</span>:

<div class="highlight-python notranslate">

<div class="highlight">

    def exp_elementwise(a: mx.array):
        source = """
            uint elem = thread_position_in_grid.x;
            // Utils from `mlx/backend/metal/kernels/utils.h` are automatically included
            uint loc = elem_to_loc(elem, inp_shape, inp_strides, inp_ndim);
            T tmp = inp[loc];
            // Output arrays are always row contiguous
            out[elem] = metal::exp(tmp);
        """

        kernel = mx.fast.metal_kernel(
            name="myexp_strided",
            input_names=["inp"],
            output_names=["out"],
            source=source
        )
        outputs = kernel(
            inputs=[a],
            template=[("T", mx.float32)],
            grid=(a.size, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[a.shape],
            output_dtypes=[a.dtype],
            ensure_row_contiguous=False,
        )
        return outputs[0]

    a = mx.random.normal(shape=(4, 16)).astype(mx.float16)
    # make non-contiguous
    a = a[::2]
    b = exp_elementwise(a)
    assert mx.allclose(b, mx.exp(a))

</div>

</div>

</div>

<div id="complex-example" class="section">

## Complex Example<a href="https://ml-explore.github.io/mlx/build/html/#complex-example"
class="headerlink" title="Link to this heading">#</a>

Let’s implement a more complex example:
<span class="pre">`grid_sample`</span> in
<span class="pre">`"bilinear"`</span> mode.

We’ll start with the following MLX implementation using standard ops:

<div class="highlight-python notranslate">

<div class="highlight">

    def grid_sample_ref(x, grid):
        N, H_in, W_in, _ = x.shape
        ix = ((grid[..., 0] + 1) * W_in - 1) / 2
        iy = ((grid[..., 1] + 1) * H_in - 1) / 2

        ix_nw = mx.floor(ix).astype(mx.int32)
        iy_nw = mx.floor(iy).astype(mx.int32)

        ix_ne = ix_nw + 1
        iy_ne = iy_nw

        ix_sw = ix_nw
        iy_sw = iy_nw + 1

        ix_se = ix_nw + 1
        iy_se = iy_nw + 1

        nw = (ix_se - ix)    * (iy_se - iy)
        ne = (ix    - ix_sw) * (iy_sw - iy)
        sw = (ix_ne - ix)    * (iy    - iy_ne)
        se = (ix    - ix_nw) * (iy    - iy_nw)

        I_nw = x[mx.arange(N)[:, None, None], iy_nw, ix_nw, :]
        I_ne = x[mx.arange(N)[:, None, None], iy_ne, ix_ne, :]
        I_sw = x[mx.arange(N)[:, None, None], iy_sw, ix_sw, :]
        I_se = x[mx.arange(N)[:, None, None], iy_se, ix_se, :]

        mask_nw = (iy_nw >= 0) & (iy_nw <= H_in - 1) & (ix_nw >= 0) & (ix_nw <= W_in - 1)
        mask_ne = (iy_ne >= 0) & (iy_ne <= H_in - 1) & (ix_ne >= 0) & (ix_ne <= W_in - 1)
        mask_sw = (iy_sw >= 0) & (iy_sw <= H_in - 1) & (ix_sw >= 0) & (ix_sw <= W_in - 1)
        mask_se = (iy_se >= 0) & (iy_se <= H_in - 1) & (ix_se >= 0) & (ix_se <= W_in - 1)

        I_nw *= mask_nw[..., None]
        I_ne *= mask_ne[..., None]
        I_sw *= mask_sw[..., None]
        I_se *= mask_se[..., None]

        output = nw[..., None] * I_nw + ne[..., None] * I_ne + sw[..., None] * I_sw + se[..., None] * I_se

        return output

</div>

</div>

Now let’s use <span class="pre">`mx.custom_function`</span> together
with <span class="pre">`mx.fast.metal_kernel`</span> to write a fast GPU
kernel for both the forward and backward passes.

First we’ll implement the forward pass as a fused kernel:

<div class="highlight-python notranslate">

<div class="highlight">

    @mx.custom_function
    def grid_sample(x, grid):

        assert x.ndim == 4, "`x` must be 4D."
        assert grid.ndim == 4, "`grid` must be 4D."

        B, _, _, C = x.shape
        _, gN, gM, D = grid.shape
        out_shape = (B, gN, gM, C)

        assert D == 2, "Last dim of `grid` must be size 2."

        source = """
            uint elem = thread_position_in_grid.x;
            int H = x_shape[1];
            int W = x_shape[2];
            int C = x_shape[3];
            int gH = grid_shape[1];
            int gW = grid_shape[2];

            int w_stride = C;
            int h_stride = W * w_stride;
            int b_stride = H * h_stride;

            uint grid_idx = elem / C * 2;
            float ix = ((grid[grid_idx] + 1) * W - 1) / 2;
            float iy = ((grid[grid_idx + 1] + 1) * H - 1) / 2;

            int ix_nw = floor(ix);
            int iy_nw = floor(iy);

            int ix_ne = ix_nw + 1;
            int iy_ne = iy_nw;

            int ix_sw = ix_nw;
            int iy_sw = iy_nw + 1;

            int ix_se = ix_nw + 1;
            int iy_se = iy_nw + 1;

            T nw = (ix_se - ix)    * (iy_se - iy);
            T ne = (ix    - ix_sw) * (iy_sw - iy);
            T sw = (ix_ne - ix)    * (iy    - iy_ne);
            T se = (ix    - ix_nw) * (iy    - iy_nw);

            int batch_idx = elem / C / gH / gW * b_stride;
            int channel_idx = elem % C;
            int base_idx = batch_idx + channel_idx;

            T I_nw = x[base_idx + iy_nw * h_stride + ix_nw * w_stride];
            T I_ne = x[base_idx + iy_ne * h_stride + ix_ne * w_stride];
            T I_sw = x[base_idx + iy_sw * h_stride + ix_sw * w_stride];
            T I_se = x[base_idx + iy_se * h_stride + ix_se * w_stride];

            I_nw = iy_nw >= 0 && iy_nw <= H - 1 && ix_nw >= 0 && ix_nw <= W - 1 ? I_nw : 0;
            I_ne = iy_ne >= 0 && iy_ne <= H - 1 && ix_ne >= 0 && ix_ne <= W - 1 ? I_ne : 0;
            I_sw = iy_sw >= 0 && iy_sw <= H - 1 && ix_sw >= 0 && ix_sw <= W - 1 ? I_sw : 0;
            I_se = iy_se >= 0 && iy_se <= H - 1 && ix_se >= 0 && ix_se <= W - 1 ? I_se : 0;

            out[elem] = nw * I_nw + ne * I_ne + sw * I_sw + se * I_se;
        """
        kernel = mx.fast.metal_kernel(
            name="grid_sample",
            input_names=["x", "grid"],
            output_names=["out"],
            source=source,
        )
        outputs = kernel(
            inputs=[x, grid],
            template=[("T", x.dtype)],
            output_shapes=[out_shape],
            output_dtypes=[x.dtype],
            grid=(np.prod(out_shape), 1, 1),
            threadgroup=(256, 1, 1),
        )
        return outputs[0]

</div>

</div>

For a reasonably sized input such as:

<div class="highlight-python notranslate">

<div class="highlight">

    x.shape = (8, 1024, 1024, 64)
    grid.shape = (8, 256, 256, 2)

</div>

</div>

On an M1 Max, we see a big performance improvement:

<span class="pre">`55.7ms`</span>` `<span class="pre">`->`</span>` `<span class="pre">`6.7ms`</span>` `<span class="pre">`=>`</span>` `<span class="pre">`8x`</span>` `<span class="pre">`speed`</span>` `<span class="pre">`up`</span>

</div>

<div id="grid-sample-vjp" class="section">

## Grid Sample VJP<a href="https://ml-explore.github.io/mlx/build/html/#grid-sample-vjp"
class="headerlink" title="Link to this heading">#</a>

Since we decorated <span class="pre">`grid_sample`</span> with
<span class="pre">`mx.custom_function`</span>, we can now define its
custom vjp transform so MLX can differentiate it.

The backwards pass requires atomically updating
<span class="pre">`x_grad`</span>/<span class="pre">`grid_grad`</span>
and so requires a few extra
<span class="pre">`mx.fast.metal_kernel`</span> features:

- <span class="pre">`init_value=0`</span>  
  Initialize all of the kernel’s outputs to this value before it runs.
  This allows us to update only part of the output arrays with the
  kernel.

- <span class="pre">`atomic_outputs=True`</span>  
  Designate all of the kernel outputs as
  <span class="pre">`atomic`</span> in the function signature. This
  means we can use Metal’s <span class="pre">`atomic`</span> features to
  simultaneously update the <span class="pre">`x_grad`</span> and
  <span class="pre">`grid_grad`</span> arrays from multiple
  threadgroups. See section 6.15 of the <a
  href="https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf"
  class="reference external">Metal Shading Language Specification</a>
  for more details.

We can then implement the backwards pass as follows:

<div class="highlight-python notranslate">

<div class="highlight">

    @grid_sample.vjp
    def grid_sample_vjp(primals, cotangent, _):
        x, grid = primals
        B, _, _, C = x.shape
        _, gN, gM, D = grid.shape

        assert D == 2, "Last dim of `grid` must be size 2."

        source = """
            uint elem = thread_position_in_grid.x;
            int H = x_shape[1];
            int W = x_shape[2];
            int C = x_shape[3];
            // Pad C to the nearest larger simdgroup size multiple
            int C_padded = ceildiv(C, threads_per_simdgroup) * threads_per_simdgroup;

            int gH = grid_shape[1];
            int gW = grid_shape[2];

            int w_stride = C;
            int h_stride = W * w_stride;
            int b_stride = H * h_stride;

            uint grid_idx = elem / C_padded * 2;
            float ix = ((grid[grid_idx] + 1) * W - 1) / 2;
            float iy = ((grid[grid_idx + 1] + 1) * H - 1) / 2;

            int ix_nw = floor(ix);
            int iy_nw = floor(iy);

            int ix_ne = ix_nw + 1;
            int iy_ne = iy_nw;

            int ix_sw = ix_nw;
            int iy_sw = iy_nw + 1;

            int ix_se = ix_nw + 1;
            int iy_se = iy_nw + 1;

            T nw = (ix_se - ix)    * (iy_se - iy);
            T ne = (ix    - ix_sw) * (iy_sw - iy);
            T sw = (ix_ne - ix)    * (iy    - iy_ne);
            T se = (ix    - ix_nw) * (iy    - iy_nw);

            int batch_idx = elem / C_padded / gH / gW * b_stride;
            int channel_idx = elem % C_padded;
            int base_idx = batch_idx + channel_idx;

            T gix = T(0);
            T giy = T(0);
            if (channel_idx < C) {
                int cot_index = elem / C_padded * C + channel_idx;
                T cot = cotangent[cot_index];
                if (iy_nw >= 0 && iy_nw <= H - 1 && ix_nw >= 0 && ix_nw <= W - 1) {
                    int offset = base_idx + iy_nw * h_stride + ix_nw * w_stride;
                    atomic_fetch_add_explicit(&x_grad[offset], nw * cot, memory_order_relaxed);

                    T I_nw = x[offset];
                    gix -= I_nw * (iy_se - iy) * cot;
                    giy -= I_nw * (ix_se - ix) * cot;
                }
                if (iy_ne >= 0 && iy_ne <= H - 1 && ix_ne >= 0 && ix_ne <= W - 1) {
                    int offset = base_idx + iy_ne * h_stride + ix_ne * w_stride;
                    atomic_fetch_add_explicit(&x_grad[offset], ne * cot, memory_order_relaxed);

                    T I_ne = x[offset];
                    gix += I_ne * (iy_sw - iy) * cot;
                    giy -= I_ne * (ix - ix_sw) * cot;
                }
                if (iy_sw >= 0 && iy_sw <= H - 1 && ix_sw >= 0 && ix_sw <= W - 1) {
                    int offset = base_idx + iy_sw * h_stride + ix_sw * w_stride;
                    atomic_fetch_add_explicit(&x_grad[offset], sw * cot, memory_order_relaxed);

                    T I_sw = x[offset];
                    gix -= I_sw * (iy - iy_ne) * cot;
                    giy += I_sw * (ix_ne - ix) * cot;
                }
                if (iy_se >= 0 && iy_se <= H - 1 && ix_se >= 0 && ix_se <= W - 1) {
                    int offset = base_idx + iy_se * h_stride + ix_se * w_stride;
                    atomic_fetch_add_explicit(&x_grad[offset], se * cot, memory_order_relaxed);

                    T I_se = x[offset];
                    gix += I_se * (iy - iy_nw) * cot;
                    giy += I_se * (ix - ix_nw) * cot;
                }
            }

            T gix_mult = W / 2;
            T giy_mult = H / 2;

            // Reduce across each simdgroup first.
            // This is much faster than relying purely on atomics.
            gix = simd_sum(gix);
            giy = simd_sum(giy);

            if (thread_index_in_simdgroup == 0) {
                atomic_fetch_add_explicit(&grid_grad[grid_idx], gix * gix_mult, memory_order_relaxed);
                atomic_fetch_add_explicit(&grid_grad[grid_idx + 1], giy * giy_mult, memory_order_relaxed);
            }
        """
        kernel = mx.fast.metal_kernel(
            name="grid_sample_grad",
            input_names=["x", "grid", "cotangent"],
            output_names=["x_grad", "grid_grad"],
            source=source,
            atomic_outputs=True,
        )
        # pad the output channels to simd group size
        # so that our `simd_sum`s don't overlap.
        simdgroup_size = 32
        C_padded = (C + simdgroup_size - 1) // simdgroup_size * simdgroup_size
        grid_size = B * gN * gM * C_padded
        outputs = kernel(
            inputs=[x, grid, cotangent],
            template=[("T", x.dtype)],
            output_shapes=[x.shape, grid.shape],
            output_dtypes=[x.dtype, x.dtype],
            grid=(grid_size, 1, 1),
            threadgroup=(256, 1, 1),
            init_value=0,
        )
        return outputs[0], outputs[1]

</div>

</div>

There’s an even larger speed up for the vjp:

<span class="pre">`676.4ms`</span>` `<span class="pre">`->`</span>` `<span class="pre">`16.7ms`</span>` `<span class="pre">`=>`</span>` `<span class="pre">`40x`</span>` `<span class="pre">`speed`</span>` `<span class="pre">`up`</span>

</div>

</div>

<div class="prev-next-area">

<a
href="https://ml-explore.github.io/mlx/build/html/dev/metal_debugger.html"
class="left-prev" title="previous page"><em></em></a>

<div class="prev-next-info">

previous

Metal Debugger

</div>

<a
href="https://ml-explore.github.io/mlx/build/html/dev/mlx_in_cpp.html"
class="right-next" title="next page"></a>

<div class="prev-next-info">

next

Using MLX in C++

</div>

</div>

</div>

<div class="bd-sidebar-secondary bd-toc">

<div class="sidebar-secondary-items sidebar-secondary__inner">

<div class="sidebar-secondary-item">

<div class="page-toc tocsection onthispage">

Contents

</div>

- <a href="https://ml-explore.github.io/mlx/build/html/#simple-example"
  class="reference internal nav-link">Simple Example</a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#using-shape-strides"
  class="reference internal nav-link">Using Shape/Strides</a>
- <a href="https://ml-explore.github.io/mlx/build/html/#complex-example"
  class="reference internal nav-link">Complex Example</a>
- <a href="https://ml-explore.github.io/mlx/build/html/#grid-sample-vjp"
  class="reference internal nav-link">Grid Sample VJP</a>

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
