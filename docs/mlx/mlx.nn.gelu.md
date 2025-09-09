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
  href="https://ml-explore.github.io/mlx/build/html/_sources/python/nn/_autosummary/mlx.nn.GELU.rst"
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

# mlx.nn.GELU

<div id="print-main-content">

<div id="jb-print-toc">

<div>

## Contents

</div>

- <a href="https://ml-explore.github.io/mlx/build/html/#mlx.nn.GELU"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">GELU</code></span></a>

</div>

</div>

</div>

<div id="searchbox">

</div>

<div id="mlx-nn-gelu" class="section">

# mlx.nn.GELU<a href="https://ml-explore.github.io/mlx/build/html/#mlx-nn-gelu"
class="headerlink" title="Link to this heading">#</a>

*<span class="pre">class</span><span class="w"> </span>*<span class="sig-name descname"><span class="pre">GELU</span></span><span class="sig-paren">(</span>*<span class="n"><span class="pre">approx</span></span><span class="o"><span class="pre">=</span></span><span class="default_value"><span class="pre">'none'</span></span>*<span class="sig-paren">)</span><a href="https://ml-explore.github.io/mlx/build/html/#mlx.nn.GELU"
class="headerlink" title="Link to this definition">#</a>  
Applies the Gaussian Error Linear Units.

<div class="math notranslate nohighlight">

\\\textrm{GELU}(x) = x \* \Phi(x)\\

</div>

where <span class="math notranslate nohighlight">\\\Phi(x)\\</span> is
the Gaussian CDF.

However, if <span class="pre">`approx`</span> is set to ‘precise’ or
‘fast’ it applies

<div class="math notranslate nohighlight">

\\\begin{split}\textrm{GELUApprox}(x) &= 0.5 \* x \* \left(1 +
\text{Tanh}\left((\sqrt{2 / \pi} \* \left(x + 0.044715 \*
x^3\right)\right)\right) \\ \textrm{GELUFast}(x) &= x \*
\sigma\left(1.702 \* x\right)\end{split}\\

</div>

respectively.

<div class="admonition note">

Note

For compatibility with the PyTorch API, ‘tanh’ can be used as an alias
for ‘precise’.

</div>

See <a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary_functions/mlx.nn.gelu.html#mlx.nn.gelu"
class="reference internal" title="mlx.nn.gelu"><span class="pre"><code
class="sourceCode python">gelu()</code></span></a>, <a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary_functions/mlx.nn.gelu_approx.html#mlx.nn.gelu_approx"
class="reference internal" title="mlx.nn.gelu_approx"><span
class="pre"><code
class="sourceCode python">gelu_approx()</code></span></a> and <a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary_functions/mlx.nn.gelu_fast_approx.html#mlx.nn.gelu_fast_approx"
class="reference internal" title="mlx.nn.gelu_fast_approx"><span
class="pre"><code
class="sourceCode python">gelu_fast_approx()</code></span></a> for the
functional equivalents and information regarding error bounds.

Parameters<span class="colon">:</span>  
**approx** (*'none'* *\|* *'precise'* *\|* *'fast'*) – Which
approximation to gelu to use if any.

Methods

<div class="pst-scrollable-table-container">

</div>

</div>

<div class="prev-next-area">

<a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.ELU.html"
class="left-prev" title="previous page"><em></em></a>

<div class="prev-next-info">

previous

mlx.nn.ELU

</div>

<a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.GLU.html"
class="right-next" title="next page"></a>

<div class="prev-next-info">

next

mlx.nn.GLU

</div>

</div>

</div>

<div class="bd-sidebar-secondary bd-toc">

<div class="sidebar-secondary-items sidebar-secondary__inner">

<div class="sidebar-secondary-item">

<div class="page-toc tocsection onthispage">

Contents

</div>

- <a href="https://ml-explore.github.io/mlx/build/html/#mlx.nn.GELU"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">GELU</code></span></a>

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
