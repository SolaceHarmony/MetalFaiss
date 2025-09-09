Curator's note: Prefer the human-authored MLX guides for clarity.
- ../docs_curated/README.md
- ../docs_curated/PYTORCH_DISSONANCE.md
- ../docs_curated/NUMPY_USERS.md
- ../docs_curated/COMMON_PITFALLS.md

<!--
Per-file analysis (mlx.core.arctan2.md):
- Quadrant-aware angle from y and x; argument order matters.
- Add ordering, range, and zero-handling notes.
-->

## Curated Notes

- Signature is `arctan2(y, x)` (first arg is vertical component, second is horizontal).
- Output is in radians with range (-π, π]. Handles signs of both inputs to determine the correct quadrant.
- When both inputs are 0, the result is implementation‑defined; avoid passing (0, 0) if your logic depends on a specific angle.

### Examples

```python
import mlx.core as mx

# Basic angles
y = mx.array([0.0, 1.0, 0.0, -1.0])
x = mx.array([1.0, 0.0, -1.0, 0.0])
theta = mx.arctan2(y, x)
# -> [0,  π/2,  π, -π/2] (in radians)

# Quadrant awareness vs arctan(y/x)
yx = mx.array([[1.0, 1.0], [1.0, -1.0], [-1.0, -1.0], [-1.0, 1.0]])
ang2 = mx.arctan2(yx[:, 0], yx[:, 1])  # (y, x)
ang_naive = mx.arctan(yx[:, 0] / yx[:, 1])
print(ang2)
print(ang_naive)  # loses quadrant information when x<0

# Broadcasting over batches
Y = mx.random.normal((4, 3))
X = mx.random.normal((4, 3))
angles = mx.arctan2(Y, X)  # shape (4, 3)

# Avoid undefined (0, 0)
eps = 1e-8
angles_safe = mx.arctan2(Y, mx.where((Y == 0) & (X == 0), eps, X))
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
  href="https://ml-explore.github.io/mlx/build/html/_sources/python/_autosummary/mlx.core.arctan2.rst"
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

# mlx.core.arctan2

<div id="print-main-content">

<div id="jb-print-toc">

<div>

## Contents

</div>

- <a href="https://ml-explore.github.io/mlx/build/html/#mlx.core.arctan2"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">arctan2()</code></span></a>

</div>

</div>

</div>

<div id="searchbox">

</div>

<div id="mlx-core-arctan2" class="section">

# mlx.core.arctan2<a href="https://ml-explore.github.io/mlx/build/html/#mlx-core-arctan2"
class="headerlink" title="Link to this heading">#</a>

<span class="sig-name descname"><span class="pre">arctan2</span></span><span class="sig-paren">(</span>*<span class="n"><span class="pre">a</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.html#mlx.core.array"
class="reference internal" title="mlx.core.array"><span
class="pre">array</span></a></span>*, *<span class="n"><span class="pre">b</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><a
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
class="pre">array</span></a></span></span><a href="https://ml-explore.github.io/mlx/build/html/#mlx.core.arctan2"
class="headerlink" title="Link to this definition">#</a>  
Element-wise inverse tangent of the ratio of two arrays.

Parameters<span class="colon">:</span>  
- **a** (<a
  href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.html#mlx.core.array"
  class="reference internal" title="mlx.core.array"><em>array</em></a>)
  – Input array.

- **b** (<a
  href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.html#mlx.core.array"
  class="reference internal" title="mlx.core.array"><em>array</em></a>)
  – Input array.

Returns<span class="colon">:</span>  
The inverse tangent of the ratio of <span class="pre">`a`</span> and
<span class="pre">`b`</span>.

Return type<span class="colon">:</span>  
<a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.html#mlx.core.array"
class="reference internal" title="mlx.core.array"><em>array</em></a>

</div>

<div class="prev-next-area">

<a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.arctan.html"
class="left-prev" title="previous page"><em></em></a>

<div class="prev-next-info">

previous

mlx.core.arctan

</div>

<a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.arctanh.html"
class="right-next" title="next page"></a>

<div class="prev-next-info">

next

mlx.core.arctanh

</div>

</div>

</div>

<div class="bd-sidebar-secondary bd-toc">

<div class="sidebar-secondary-items sidebar-secondary__inner">

<div class="sidebar-secondary-item">

<div class="page-toc tocsection onthispage">

Contents

</div>

- <a href="https://ml-explore.github.io/mlx/build/html/#mlx.core.arctan2"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">arctan2()</code></span></a>

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
