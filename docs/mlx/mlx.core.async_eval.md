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
  href="https://ml-explore.github.io/mlx/build/html/_sources/python/_autosummary/mlx.core.async_eval.rst"
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

# mlx.core.async_eval

<div id="print-main-content">

<div id="jb-print-toc">

<div>

## Contents

</div>

- <a
  href="https://ml-explore.github.io/mlx/build/html/#mlx.core.async_eval"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">async_eval()</code></span></a>

</div>

</div>

</div>

<div id="searchbox">

</div>

<div id="mlx-core-async-eval" class="section">

# mlx.core.async_eval<a
href="https://ml-explore.github.io/mlx/build/html/#mlx-core-async-eval"
class="headerlink" title="Link to this heading">#</a>

<span class="sig-name descname"><span class="pre">async_eval</span></span><span class="sig-paren">(</span>*<span class="o"><span class="pre">\*</span></span><span class="n"><span class="pre">args</span></span>*<span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#mlx.core.async_eval"
class="headerlink" title="Link to this definition">#</a>  
Asynchronously evaluate an <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.html#mlx.core.array"
class="reference internal" title="mlx.core.array"><span
class="pre"><code class="sourceCode python">array</code></span></a> or
tree of <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.html#mlx.core.array"
class="reference internal" title="mlx.core.array"><span
class="pre"><code class="sourceCode python">array</code></span></a>.

<div class="admonition note">

Note

This is an experimental API and may change in future versions.

</div>

Parameters<span class="colon">:</span>  
**\*args** (*arrays* *or* *trees* *of* *arrays*) – Each argument can be
a single array or a tree of arrays. If a tree is given the nodes can be
a Python <a href="https://docs.python.org/3/library/stdtypes.html#list"
class="reference external" title="(in Python v3.13)"><span
class="pre"><code
class="sourceCode python"><span class="bu">list</span></code></span></a>,
<a href="https://docs.python.org/3/library/stdtypes.html#tuple"
class="reference external" title="(in Python v3.13)"><span
class="pre"><code
class="sourceCode python"><span class="bu">tuple</span></code></span></a>
or <a href="https://docs.python.org/3/library/stdtypes.html#dict"
class="reference external" title="(in Python v3.13)"><span
class="pre"><code
class="sourceCode python"><span class="bu">dict</span></code></span></a>.
Leaves which are not arrays are ignored.

Example

<div class="doctest highlight-default notranslate">

<div class="highlight">

    >>> x = mx.array(1.0)
    >>> y = mx.exp(x)
    >>> mx.async_eval(y)
    >>> print(y)
    >>>
    >>> y = mx.exp(x)
    >>> mx.async_eval(y)
    >>> z = y + 3
    >>> mx.async_eval(z)
    >>> print(z)

</div>

</div>

</div>

<div class="prev-next-area">

<a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.eval.html"
class="left-prev" title="previous page"><em></em></a>

<div class="prev-next-info">

previous

mlx.core.eval

</div>

<a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.compile.html"
class="right-next" title="next page"></a>

<div class="prev-next-info">

next

mlx.core.compile

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
  href="https://ml-explore.github.io/mlx/build/html/#mlx.core.async_eval"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">async_eval()</code></span></a>

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
