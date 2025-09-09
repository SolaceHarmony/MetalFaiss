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
  href="https://ml-explore.github.io/mlx/build/html/_sources/python/_autosummary/mlx.core.set_cache_limit.rst"
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

# mlx.core.set_cache_limit

<div id="print-main-content">

<div id="jb-print-toc">

<div>

## Contents

</div>

- <a
  href="https://ml-explore.github.io/mlx/build/html/#mlx.core.set_cache_limit"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">set_cache_limit()</code></span></a>

</div>

</div>

</div>

<div id="searchbox">

</div>

<div id="mlx-core-set-cache-limit" class="section">

# mlx.core.set_cache_limit<a
href="https://ml-explore.github.io/mlx/build/html/#mlx-core-set-cache-limit"
class="headerlink" title="Link to this heading">#</a>

<span class="sig-name descname"><span class="pre">set_cache_limit</span></span><span class="sig-paren">(</span>*<span class="n"><span class="pre">limit</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><a href="https://docs.python.org/3/library/functions.html#int"
class="reference external" title="(in Python v3.13)"><span
class="pre">int</span></a></span>*<span class="sig-paren">)</span> <span class="sig-return"><span class="sig-return-icon">→</span> <span class="sig-return-typehint"><a href="https://docs.python.org/3/library/functions.html#int"
class="reference external" title="(in Python v3.13)"><span
class="pre">int</span></a></span></span><a
href="https://ml-explore.github.io/mlx/build/html/#mlx.core.set_cache_limit"
class="headerlink" title="Link to this definition">#</a>  
Set the free cache limit.

If using more than the given limit, free memory will be reclaimed from
the cache on the next allocation. To disable the cache, set the limit to
<span class="pre">`0`</span>.

The cache limit defaults to the memory limit. See <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.set_memory_limit.html#mlx.core.set_memory_limit"
class="reference internal" title="mlx.core.set_memory_limit"><span
class="pre"><code
class="sourceCode python">set_memory_limit()</code></span></a> for more
details.

Parameters<span class="colon">:</span>  
**limit**
(<a href="https://docs.python.org/3/library/functions.html#int"
class="reference external" title="(in Python v3.13)"><em>int</em></a>) –
The cache limit in bytes.

Returns<span class="colon">:</span>  
The previous cache limit in bytes.

Return type<span class="colon">:</span>  
<a href="https://docs.python.org/3/library/functions.html#int"
class="reference external" title="(in Python v3.13)"><em>int</em></a>

</div>

<div class="prev-next-area">

<a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.set_memory_limit.html"
class="left-prev" title="previous page"><em></em></a>

<div class="prev-next-info">

previous

mlx.core.set_memory_limit

</div>

<a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.set_wired_limit.html"
class="right-next" title="next page"></a>

<div class="prev-next-info">

next

mlx.core.set_wired_limit

</div>

</div>

</div>

<div class="bd-sidebar-secondary bd-toc">

<div class="sidebar-secondary-items sidebar-secondary__inner">

<div class="sidebar-secondary-item">

<div class="page-toc tocsection onthispage">

Contents

</div>

- <a
  href="https://ml-explore.github.io/mlx/build/html/#mlx.core.set_cache_limit"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">set_cache_limit()</code></span></a>

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
