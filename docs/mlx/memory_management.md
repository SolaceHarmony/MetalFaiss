Curator's note: Prefer the human-authored MLX guides for clarity.
- ../docs_curated/README.md
- ../docs_curated/PYTORCH_DISSONANCE.md
- ../docs_curated/NUMPY_USERS.md
- ../docs_curated/COMMON_PITFALLS.md

<!--
Per-file analysis (memory_management.md):
- API table for memory queries and limits.
- Add practical guidance for leaks, caches, and profiling.
-->

## Curated Notes

- Use `get_active_memory`/`get_peak_memory` to bracket critical sections; call `reset_peak_memory()` between runs to compare.
- `clear_cache()` frees cached buffers; avoid calling it in tight loops as it can harm performance.
- Consider `set_memory_limit` in constrained environments to catch oversubscription early during development.


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
  href="https://ml-explore.github.io/mlx/build/html/_sources/python/memory_management.rst"
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

# Memory Management

<div id="print-main-content">

<div id="jb-print-toc">

</div>

</div>

</div>

<div id="searchbox">

</div>

<div id="memory-management" class="section">

# Memory Management<a href="https://ml-explore.github.io/mlx/build/html/#memory-management"
class="headerlink" title="Link to this heading">#</a>

<div class="pst-scrollable-table-container">

|  |  |
|----|----|
| <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.get_active_memory.html#mlx.core.get_active_memory"
class="reference internal" title="mlx.core.get_active_memory"><span
class="pre"><code
class="sourceCode python">get_active_memory</code></span></a>() | Get the actively used memory in bytes. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.get_peak_memory.html#mlx.core.get_peak_memory"
class="reference internal" title="mlx.core.get_peak_memory"><span
class="pre"><code
class="sourceCode python">get_peak_memory</code></span></a>() | Get the peak amount of used memory in bytes. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.reset_peak_memory.html#mlx.core.reset_peak_memory"
class="reference internal" title="mlx.core.reset_peak_memory"><span
class="pre"><code
class="sourceCode python">reset_peak_memory</code></span></a>() | Reset the peak memory to zero. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.get_cache_memory.html#mlx.core.get_cache_memory"
class="reference internal" title="mlx.core.get_cache_memory"><span
class="pre"><code
class="sourceCode python">get_cache_memory</code></span></a>() | Get the cache size in bytes. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.set_memory_limit.html#mlx.core.set_memory_limit"
class="reference internal" title="mlx.core.set_memory_limit"><span
class="pre"><code
class="sourceCode python">set_memory_limit</code></span></a>(limit) | Set the memory limit. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.set_cache_limit.html#mlx.core.set_cache_limit"
class="reference internal" title="mlx.core.set_cache_limit"><span
class="pre"><code
class="sourceCode python">set_cache_limit</code></span></a>(limit) | Set the free cache limit. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.set_wired_limit.html#mlx.core.set_wired_limit"
class="reference internal" title="mlx.core.set_wired_limit"><span
class="pre"><code
class="sourceCode python">set_wired_limit</code></span></a>(limit) | Set the wired size limit. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.clear_cache.html#mlx.core.clear_cache"
class="reference internal" title="mlx.core.clear_cache"><span
class="pre"><code
class="sourceCode python">clear_cache</code></span></a>() | Clear the memory cache. |

</div>

</div>

<div class="prev-next-area">

<a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.metal.stop_capture.html"
class="left-prev" title="previous page"><em></em></a>

<div class="prev-next-info">

previous

mlx.core.metal.stop_capture

</div>

<a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.get_active_memory.html"
class="right-next" title="next page"></a>

<div class="prev-next-info">

next

mlx.core.get_active_memory

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
