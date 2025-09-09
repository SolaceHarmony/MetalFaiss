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
  href="https://ml-explore.github.io/mlx/build/html/_sources/python/_autosummary/mlx.core.vmap.rst"
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

# mlx.core.vmap

<div id="print-main-content">

<div id="jb-print-toc">

<div>

## Contents

</div>

- <a href="https://ml-explore.github.io/mlx/build/html/#mlx.core.vmap"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">vmap()</code></span></a>

</div>

</div>

</div>

<div id="searchbox">

</div>

<div id="mlx-core-vmap" class="section">

# mlx.core.vmap<a href="https://ml-explore.github.io/mlx/build/html/#mlx-core-vmap"
class="headerlink" title="Link to this heading">#</a>

<span class="sig-name descname"><span class="pre">vmap</span></span><span class="sig-paren">(</span>*<span class="n"><span class="pre">fun</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><span class="pre">Callable</span></span>*, *<span class="n"><span class="pre">in_axes</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><a href="https://docs.python.org/3/library/functions.html#object"
class="reference external" title="(in Python v3.13)"><span
class="pre">object</span></a></span><span class="w"> </span><span class="o"><span class="pre">=</span></span><span class="w"> </span><span class="default_value"><span class="pre">0</span></span>*, *<span class="n"><span class="pre">out_axes</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><a href="https://docs.python.org/3/library/functions.html#object"
class="reference external" title="(in Python v3.13)"><span
class="pre">object</span></a></span><span class="w"> </span><span class="o"><span class="pre">=</span></span><span class="w"> </span><span class="default_value"><span class="pre">0</span></span>*<span class="sig-paren">)</span> <span class="sig-return"><span class="sig-return-icon">→</span> <span class="sig-return-typehint"><span class="pre">Callable</span></span></span><a href="https://ml-explore.github.io/mlx/build/html/#mlx.core.vmap"
class="headerlink" title="Link to this definition">#</a>  
Returns a vectorized version of <span class="pre">`fun`</span>.

Parameters<span class="colon">:</span>  
- **fun** (*Callable*) – A function which takes a variable number of <a
  href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.html#mlx.core.array"
  class="reference internal" title="mlx.core.array"><span
  class="pre"><code class="sourceCode python">array</code></span></a> or
  a tree of <a
  href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.html#mlx.core.array"
  class="reference internal" title="mlx.core.array"><span
  class="pre"><code class="sourceCode python">array</code></span></a>
  and returns a variable number of <a
  href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.html#mlx.core.array"
  class="reference internal" title="mlx.core.array"><span
  class="pre"><code class="sourceCode python">array</code></span></a> or
  a tree of <a
  href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.html#mlx.core.array"
  class="reference internal" title="mlx.core.array"><span
  class="pre"><code class="sourceCode python">array</code></span></a>.

- **in_axes**
  (<a href="https://docs.python.org/3/library/functions.html#int"
  class="reference external" title="(in Python v3.13)"><em>int</em></a>*,*
  *optional*) – An integer or a valid prefix tree of the inputs to
  <span class="pre">`fun`</span> where each node specifies the vmapped
  axis. If the value is <span class="pre">`None`</span> then the
  corresponding input(s) are not vmapped. Defaults to
  <span class="pre">`0`</span>.

- **out_axes**
  (<a href="https://docs.python.org/3/library/functions.html#int"
  class="reference external" title="(in Python v3.13)"><em>int</em></a>*,*
  *optional*) – An integer or a valid prefix tree of the outputs of
  <span class="pre">`fun`</span> where each node specifies the vmapped
  axis. If the value is <span class="pre">`None`</span> then the
  corresponding outputs(s) are not vmapped. Defaults to
  <span class="pre">`0`</span>.

Returns<span class="colon">:</span>  
The vectorized function.

Return type<span class="colon">:</span>  
*Callable*

</div>

<div class="prev-next-area">

<a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.vjp.html"
class="left-prev" title="previous page"><em></em></a>

<div class="prev-next-info">

previous

mlx.core.vjp

</div>

<a href="https://ml-explore.github.io/mlx/build/html/python/fast.html"
class="right-next" title="next page"></a>

<div class="prev-next-info">

next

Fast

</div>

</div>

</div>

<div class="bd-sidebar-secondary bd-toc">

<div class="sidebar-secondary-items sidebar-secondary__inner">

<div class="sidebar-secondary-item">

<div class="page-toc tocsection onthispage">

Contents

</div>

- <a href="https://ml-explore.github.io/mlx/build/html/#mlx.core.vmap"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">vmap()</code></span></a>

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
