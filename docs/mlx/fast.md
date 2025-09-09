Curator's note: Prefer the human-authored MLX guides for clarity.
- ../docs_curated/README.md
- ../docs_curated/PYTORCH_DISSONANCE.md
- ../docs_curated/NUMPY_USERS.md
- ../docs_curated/COMMON_PITFALLS.md

<!--
Per-file analysis (fast.md):
- Catalog of performance-focused kernels (norms, attention, RoPE, custom Metal JIT).
- Value: usage cautions for masks/scales in attention and dtype notes for norms.
-->

## Curated Notes

- Norms (`rms_norm`, `layer_norm`): keep weights/bias in the same dtype as activations to avoid hidden casts.
- Scaled dot-product attention: ensure mask broadcasting is correct; scale typically `1/sqrt(d_k)`.
- `fast.metal_kernel`: prototype on tiny shapes; validate correctness vs a pure-MLX version before scaling up.

### Examples

```python
import mlx.core as mx

# Scaled dot-product attention (single head)
Q = mx.random.normal((2, 8, 16))   # (batch, seq_q, d)
K = mx.random.normal((2, 8, 16))   # (batch, seq_k, d)
V = mx.random.normal((2, 8, 32))   # (batch, seq_k, dv)
scale = 1.0 / mx.sqrt(mx.array(16.0))

# Optional mask: True for keep, False for mask out
mask = mx.array([[True]*8 + [False]*0]*2)[:, :8]  # (2, 8)
mask = mx.reshape(mask, (2, 1, 8))                # broadcast to (B, 1, seq_k)

O = mx.fast.scaled_dot_product_attention(Q, K, V, scale=scale)
```

### Metal Kernel Quick Reference

- `mx.fast.metal_kernel(name, input_names, output_names, source, header="", ensure_row_contiguous=True, atomic_outputs=False)` returns a callable.
- Call the returned kernel with:
  - `inputs=[...]`, `template=[("T", dtype), ("K", 3), ...]`
  - `grid=(n,1,1)`, `threadgroup=(tg,1,1)`
  - `output_shapes=[...]`, `output_dtypes=[...]`
  - Optional: `ensure_row_contiguous=False` (use `elem_to_loc`), `init_value=0`, `atomic_outputs=True`, `verbose=True`

Header/verbose example:

```python
src = """
    // body uses helper declared in header
    uint elem = thread_position_in_grid.x;
    out[elem] = helper(inp[elem]);
"""
hdr = """
    inline T helper(T v) { return metal::exp(v); }
"""
k = mx.fast.metal_kernel(
    name="myexp_hdr", input_names=["inp"], output_names=["out"], source=src, header=hdr
)
out = k(inputs=[a], template=[("T", a.dtype)], grid=(a.size,1,1), threadgroup=(256,1,1),
        output_shapes=[a.shape], output_dtypes=[a.dtype], verbose=True)[0]
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
  href="https://ml-explore.github.io/mlx/build/html/_sources/python/fast.rst"
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

# Fast

<div id="print-main-content">

<div id="jb-print-toc">

</div>

</div>

</div>

<div id="searchbox">

</div>

<div id="fast" class="section">

<span id="id1"></span>

# Fast<a href="https://ml-explore.github.io/mlx/build/html/#fast"
class="headerlink" title="Link to this heading">#</a>

<div class="pst-scrollable-table-container">

|  |  |
|----|----|
| <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fast.rms_norm.html#mlx.core.fast.rms_norm"
class="reference internal" title="mlx.core.fast.rms_norm"><span
class="pre"><code class="sourceCode python">rms_norm</code></span></a>(x, weight, eps, \*\[, stream\]) | Root Mean Square normalization (RMS norm). |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fast.layer_norm.html#mlx.core.fast.layer_norm"
class="reference internal" title="mlx.core.fast.layer_norm"><span
class="pre"><code class="sourceCode python">layer_norm</code></span></a>(x, weight, bias, eps, \*\[, stream\]) | Layer normalization. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fast.rope.html#mlx.core.fast.rope"
class="reference internal" title="mlx.core.fast.rope"><span
class="pre"><code class="sourceCode python">rope</code></span></a>(a, dims, \*, traditional, base, scale, ...) | Apply rotary positional encoding to the input. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fast.scaled_dot_product_attention.html#mlx.core.fast.scaled_dot_product_attention"
class="reference internal"
title="mlx.core.fast.scaled_dot_product_attention"><span
class="pre"><code
class="sourceCode python">scaled_dot_product_attention</code></span></a>(q, k, v, \*, scale) | A fast implementation of multi-head attention: <span class="pre">`O`</span>` `<span class="pre">`=`</span>` `<span class="pre">`softmax(Q`</span>` `<span class="pre">`@`</span>` `<span class="pre">`K.T,`</span>` `<span class="pre">`dim=-1)`</span>` `<span class="pre">`@`</span>` `<span class="pre">`V`</span>. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fast.metal_kernel.html#mlx.core.fast.metal_kernel"
class="reference internal" title="mlx.core.fast.metal_kernel"><span
class="pre"><code
class="sourceCode python">metal_kernel</code></span></a>(name, input_names, ...\[, ...\]) | A jit-compiled custom Metal kernel defined from a source string. |

</div>

</div>

<div class="prev-next-area">

<a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.vmap.html"
class="left-prev" title="previous page"><em></em></a>

<div class="prev-next-info">

previous

mlx.core.vmap

</div>

<a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fast.rms_norm.html"
class="right-next" title="next page"></a>

<div class="prev-next-info">

next

mlx.core.fast.rms_norm

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
