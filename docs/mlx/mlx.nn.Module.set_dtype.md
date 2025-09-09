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
  href="https://ml-explore.github.io/mlx/build/html/_sources/python/nn/_autosummary/mlx.nn.Module.set_dtype.rst"
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

# mlx.nn.Module.set_dtype

<div id="print-main-content">

<div id="jb-print-toc">

<div>

## Contents

</div>

- <a
  href="https://ml-explore.github.io/mlx/build/html/#mlx.nn.Module.set_dtype"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">Module.set_dtype()</code></span></a>

</div>

</div>

</div>

<div id="searchbox">

</div>

<div id="mlx-nn-module-set-dtype" class="section">

# mlx.nn.Module.set_dtype<a
href="https://ml-explore.github.io/mlx/build/html/#mlx-nn-module-set-dtype"
class="headerlink" title="Link to this heading">#</a>

<span class="sig-prename descclassname"><span class="pre">Module.</span></span><span class="sig-name descname"><span class="pre">set_dtype</span></span><span class="sig-paren">(</span>*<span class="pre">dtype:</span> <span class="pre">~mlx.core.Dtype,</span> <span class="pre">predicate:</span> <span class="pre">~typing.Callable\[\[~mlx.core.Dtype\],</span> <span class="pre">bool\]</span> <span class="pre">\|</span> <span class="pre">None</span> <span class="pre">=</span> <span class="pre">\<function</span> <span class="pre">Module.\<lambda\>\></span>*<span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#mlx.nn.Module.set_dtype"
class="headerlink" title="Link to this definition">#</a>  
Set the dtype of the module’s parameters.

Parameters<span class="colon">:</span>  
- **dtype** (<a
  href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.Dtype.html#mlx.core.Dtype"
  class="reference internal" title="mlx.core.Dtype"><em>Dtype</em></a>)
  – The new dtype.

- **predicate**
  (<a href="https://docs.python.org/3/library/typing.html#typing.Callable"
  class="reference external"
  title="(in Python v3.13)"><em>Callable</em></a>*,* *optional*) – A
  predicate to select parameters to cast. By default, only parameters of
  type <span class="pre">`floating`</span> will be updated to avoid
  casting integer parameters to the new dtype.

</div>

<div class="prev-next-area">

<a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Module.save_weights.html"
class="left-prev" title="previous page"><em></em></a>

<div class="prev-next-info">

previous

mlx.nn.Module.save_weights

</div>

<a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Module.train.html"
class="right-next" title="next page"></a>

<div class="prev-next-info">

next

mlx.nn.Module.train

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
  href="https://ml-explore.github.io/mlx/build/html/#mlx.nn.Module.set_dtype"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">Module.set_dtype()</code></span></a>

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
