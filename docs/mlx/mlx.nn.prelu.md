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
  href="https://ml-explore.github.io/mlx/build/html/_sources/python/nn/_autosummary/mlx.nn.PReLU.rst"
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

# mlx.nn.PReLU

<div id="print-main-content">

<div id="jb-print-toc">

<div>

## Contents

</div>

- <a href="https://ml-explore.github.io/mlx/build/html/#mlx.nn.PReLU"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">PReLU</code></span></a>

</div>

</div>

</div>

<div id="searchbox">

</div>

<div id="mlx-nn-prelu" class="section">

# mlx.nn.PReLU<a href="https://ml-explore.github.io/mlx/build/html/#mlx-nn-prelu"
class="headerlink" title="Link to this heading">#</a>

*<span class="pre">class</span><span class="w"> </span>*<span class="sig-name descname"><span class="pre">PReLU</span></span><span class="sig-paren">(</span>*<span class="n"><span class="pre">num_parameters</span></span><span class="o"><span class="pre">=</span></span><span class="default_value"><span class="pre">1</span></span>*, *<span class="n"><span class="pre">init</span></span><span class="o"><span class="pre">=</span></span><span class="default_value"><span class="pre">0.25</span></span>*<span class="sig-paren">)</span><a href="https://ml-explore.github.io/mlx/build/html/#mlx.nn.PReLU"
class="headerlink" title="Link to this definition">#</a>  
Applies the element-wise parametric ReLU.  
Applies <span class="math notranslate nohighlight">\\\max(0, x) + a \*
\min(0, x)\\</span> element wise, where
<span class="math notranslate nohighlight">\\a\\</span> is an array.

See <a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary_functions/mlx.nn.prelu.html#mlx.nn.prelu"
class="reference internal" title="mlx.nn.prelu"><span class="pre"><code
class="sourceCode python">prelu()</code></span></a> for the functional
equivalent.

Parameters<span class="colon">:</span>  
- **num_parameters** – number of
  <span class="math notranslate nohighlight">\\a\\</span> to learn.
  Default: <span class="pre">`1`</span>

- **init** – the initial value of
  <span class="math notranslate nohighlight">\\a\\</span>. Default:
  <span class="pre">`0.25`</span>

Methods

<div class="pst-scrollable-table-container">

</div>

</div>

<div class="prev-next-area">

<a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.MultiHeadAttention.html"
class="left-prev" title="previous page"><em></em></a>

<div class="prev-next-info">

previous

mlx.nn.MultiHeadAttention

</div>

<a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.QuantizedEmbedding.html"
class="right-next" title="next page"></a>

<div class="prev-next-info">

next

mlx.nn.QuantizedEmbedding

</div>

</div>

</div>

<div class="bd-sidebar-secondary bd-toc">

<div class="sidebar-secondary-items sidebar-secondary__inner">

<div class="sidebar-secondary-item">

<div class="page-toc tocsection onthispage">

Contents

</div>

- <a href="https://ml-explore.github.io/mlx/build/html/#mlx.nn.PReLU"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">PReLU</code></span></a>

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
