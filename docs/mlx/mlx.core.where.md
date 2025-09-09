Curator's note: Prefer the human-authored MLX guides for clarity.
- ../docs_curated/README.md
- ../docs_curated/PYTORCH_DISSONANCE.md
- ../docs_curated/NUMPY_USERS.md
- ../docs_curated/COMMON_PITFALLS.md


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
  href="https://ml-explore.github.io/mlx/build/html/_sources/python/_autosummary/mlx.core.where.rst"
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

# mlx.core.where

<div id="print-main-content">

<div id="jb-print-toc">

<div>

## Contents

</div>

- <a href="https://ml-explore.github.io/mlx/build/html/#mlx.core.where"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">where()</code></span></a>

</div>

</div>

</div>

<div id="searchbox">

</div>

<div id="mlx-core-where" class="section">

# mlx.core.where<a href="https://ml-explore.github.io/mlx/build/html/#mlx-core-where"
class="headerlink" title="Link to this heading">#</a>

## Curated Notes

- Elementwise conditional: returns a new array selecting between `x` and `y` using `condition` with full NumPy‑style broadcasting rules.
- Lazy + fuseable: `mx.where` participates in the lazy graph; evaluation is deferred until needed and may be fused with adjacent ops.
- No `device=`: control placement via default device or per‑op `stream`.
- Grad‑friendly: works under `mx.value_and_grad`; condition is treated as data, not control flow.

### Performance Implications

- Vectorized execution: runs in parallel over all elements (CPU/GPU), far faster than Python `for` loops with `if/else`.
- Lazy evaluation: builds a graph node; MLX can fuse producers/consumers of `where` to reduce memory traffic.
- Branchless on GPU: typically computes both `x` and `y` expressions and then selects. This avoids warp divergence and is standard for SIMD/SIMT hardware.
- Allocation: produces a new array. MLX may reuse buffers due to laziness, but treat it as an allocation in your mental model.

When to prefer `mx.where`:
- Large arrays and batched math where parallelism dominates.
- Inside larger MLX graphs (especially compiled) where kernels can be fused.
- Differentiable conditionals where you need gradients to flow through `x`/`y`.

Python loop vs `mx.where` (summary):
- Execution: vectorized, hardware‑accelerated vs interpreted, single‑threaded.
- Computation: typically evaluates both `x` and `y` vs per‑element branch; the former wins on GPUs by avoiding divergence and leveraging throughput.
- Graph: `mx.where` is part of the MLX graph (optimizable, differentiable); loops are not.

Tips:
- Keep `x`/`y` expressions as simple as possible; heavy, redundant work will be computed for all elements.
- For chained conditionals, compose `where` calls or precompute masks; MLX can still fuse simple patterns.
- Wrap `where` usage inside a compiled function (if available) to maximize fusion and scheduling.

### Examples

```python
import mlx.core as mx

x = mx.arange(10)
mask = (x % 2) == 0
y = mx.where(mask, x, -x)          # even keep, odd negate

# Conditional arithmetic
z = mx.where(x > 3, x * 10, x + 1)

# Under grad
def loss(a):
    return mx.mean(mx.where(a > 0, a, 0.1 * a) ** 2)
val, grad = mx.value_and_grad(loss)(x.astype(mx.float32))

# Route to CPU if you need float64 internally
with mx.default_device(mx.cpu):
    y64 = mx.where(mask, x.astype(mx.float64), (-x).astype(mx.float64))
```

<span class="sig-name descname"><span class="pre">where</span></span><span class="sig-paren">(</span>*<span class="n"><span class="pre">condition</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><span class="pre">scalar</span><span class="w"> </span><span class="p"><span class="pre">\|</span></span><span class="w"> </span><a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.html#mlx.core.array"
class="reference internal" title="mlx.core.array"><span
class="pre">array</span></a></span>*, *<span class="n"><span class="pre">x</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><span class="pre">scalar</span><span class="w"> </span><span class="p"><span class="pre">\|</span></span><span class="w"> </span><a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.html#mlx.core.array"
class="reference internal" title="mlx.core.array"><span
class="pre">array</span></a></span>*, *<span class="n"><span class="pre">y</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><span class="pre">scalar</span><span class="w"> </span><span class="p"><span class="pre">\|</span></span><span class="w"> </span><a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.html#mlx.core.array"
class="reference internal" title="mlx.core.array"><span
class="pre">array</span></a></span>*, *<span class="o"><span class="pre">/</span></span>*, *<span class="o"><span class="pre">\*</span></span>*, *<span class="n"><span class="pre">stream</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><a href="https://docs.python.org/3/library/constants.html#None"
class="reference external" title="(in Python v3.13)"><span
class="pre">None</span></a><span class="w"> </span><span class="p"><span class="pre">\|</span></span><span class="w"> </span><a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/stream_class.html#mlx.core.Stream"
class="reference internal" title="mlx.core.Stream"><span
class="pre">Stream</span></a><span class="w"> </span><span class="p"><span class="pre">\|</span></span><span class="w"> </span><a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.Device.html#mlx.core.Device"
class="reference internal" title="mlx.core.Device"><span
class="pre">Device</span></a></span><span class="w"> </span><span class="o"><span class="pre">=</span></span><span class="w"> </span><span class="default_value"><span class="pre">None</span></span>*<span class="sig-paren">)</span> <span class="sig-return"><span class="sig-return-icon">→</span> <span class="sig-return-typehint"><a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.html#mlx.core.array"
class="reference internal" title="mlx.core.array"><span
class="pre">array</span></a></span></span><a href="https://ml-explore.github.io/mlx/build/html/#mlx.core.where"
class="headerlink" title="Link to this definition">#</a>  
Select from <span class="pre">`x`</span> or <span class="pre">`y`</span>
according to <span class="pre">`condition`</span>.

The condition and input arrays must be the same shape or broadcastable
with each another.

Parameters<span class="colon">:</span>  
- **condition** (<a
  href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.html#mlx.core.array"
  class="reference internal" title="mlx.core.array"><em>array</em></a>)
  – The condition array.

- **x** (<a
  href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.html#mlx.core.array"
  class="reference internal" title="mlx.core.array"><em>array</em></a>)
  – The input selected from where condition is
  <span class="pre">`True`</span>.

- **y** (<a
  href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.html#mlx.core.array"
  class="reference internal" title="mlx.core.array"><em>array</em></a>)
  – The input selected from where condition is
  <span class="pre">`False`</span>.

Returns<span class="colon">:</span>  
The output containing elements selected from
<span class="pre">`x`</span> and <span class="pre">`y`</span>.

Return type<span class="colon">:</span>  
<a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.html#mlx.core.array"
class="reference internal" title="mlx.core.array"><em>array</em></a>

</div>

<div class="prev-next-area">

<a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.view.html"
class="left-prev" title="previous page"><em></em></a>

<div class="prev-next-info">

previous

mlx.core.view

</div>

<a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.zeros.html"
class="right-next" title="next page"></a>

<div class="prev-next-info">

next

mlx.core.zeros

</div>

</div>

</div>

<div class="bd-sidebar-secondary bd-toc">

<div class="sidebar-secondary-items sidebar-secondary__inner">

<div class="sidebar-secondary-item">

<div class="page-toc tocsection onthispage">

Contents

</div>

- <a href="https://ml-explore.github.io/mlx/build/html/#mlx.core.where"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">where()</code></span></a>

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
