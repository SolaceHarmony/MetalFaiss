Curator's note: Prefer the human-authored MLX guides for clarity.
- ../docs_curated/README.md
- ../docs_curated/PYTORCH_DISSONANCE.md
- ../docs_curated/NUMPY_USERS.md
- ../docs_curated/COMMON_PITFALLS.md

<!--
Per-file analysis (metal.md):
- Lists Metal introspection/capture helpers; ties to metal_debugger.
- Add practical notes on availability checks and capture hygiene.
-->

## Curated Notes

- Guard GPU‑specific code paths with `mx.metal.is_available()` to keep code portable to CPU‑only environments.
- Inspect `mx.metal.device_info()` once and cache results for logging/debug builds; it’s stable across a run.
- Pair `start_capture(path)`/`stop_capture()` within a tight scope; avoid leaving capture enabled during unrelated work.


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
  href="https://ml-explore.github.io/mlx/build/html/_sources/python/metal.rst"
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

# Metal

<div id="print-main-content">

<div id="jb-print-toc">

</div>

</div>

</div>

<div id="searchbox">

</div>

<div id="metal" class="section">

# Metal<a href="https://ml-explore.github.io/mlx/build/html/#metal"
class="headerlink" title="Link to this heading">#</a>

<div class="pst-scrollable-table-container">

|  |  |
|----|----|
| <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.metal.is_available.html#mlx.core.metal.is_available"
class="reference internal" title="mlx.core.metal.is_available"><span
class="pre"><code
class="sourceCode python">is_available</code></span></a>() | Check if the Metal back-end is available. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.metal.device_info.html#mlx.core.metal.device_info"
class="reference internal" title="mlx.core.metal.device_info"><span
class="pre"><code
class="sourceCode python">device_info</code></span></a>() | Get information about the GPU device and system settings. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.metal.start_capture.html#mlx.core.metal.start_capture"
class="reference internal" title="mlx.core.metal.start_capture"><span
class="pre"><code
class="sourceCode python">start_capture</code></span></a>(path) | Start a Metal capture. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.metal.stop_capture.html#mlx.core.metal.stop_capture"
class="reference internal" title="mlx.core.metal.stop_capture"><span
class="pre"><code
class="sourceCode python">stop_capture</code></span></a>() | Stop a Metal capture. |

</div>

</div>

<div class="prev-next-area">

<a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.linalg.solve_triangular.html"
class="left-prev" title="previous page"><em></em></a>

<div class="prev-next-info">

previous

mlx.core.linalg.solve_triangular

</div>

<a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.metal.is_available.html"
class="right-next" title="next page"></a>

<div class="prev-next-info">

next

mlx.core.metal.is_available

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
