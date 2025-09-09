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
  href="https://ml-explore.github.io/mlx/build/html/_sources/python/nn/_autosummary_functions/mlx.nn.losses.smooth_l1_loss.rst"
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

# mlx.nn.losses.smooth_l1_loss

<div id="print-main-content">

<div id="jb-print-toc">

<div>

## Contents

</div>

- <a
  href="https://ml-explore.github.io/mlx/build/html/#mlx.nn.losses.smooth_l1_loss"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">smooth_l1_loss</code></span></a>

</div>

</div>

</div>

<div id="searchbox">

</div>

<div id="mlx-nn-losses-smooth-l1-loss" class="section">

# mlx.nn.losses.smooth_l1_loss<a
href="https://ml-explore.github.io/mlx/build/html/#mlx-nn-losses-smooth-l1-loss"
class="headerlink" title="Link to this heading">#</a>

*<span class="pre">class</span><span class="w"> </span>*<span class="sig-name descname"><span class="pre">smooth_l1_loss</span></span><span class="sig-paren">(</span>*<span class="n"><span class="pre">predictions</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.html#mlx.core.array"
class="reference internal" title="mlx.core.array"><span
class="pre">array</span></a></span>*, *<span class="n"><span class="pre">targets</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.html#mlx.core.array"
class="reference internal" title="mlx.core.array"><span
class="pre">array</span></a></span>*, *<span class="n"><span class="pre">beta</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><a href="https://docs.python.org/3/library/functions.html#float"
class="reference external" title="(in Python v3.13)"><span
class="pre">float</span></a></span><span class="w"> </span><span class="o"><span class="pre">=</span></span><span class="w"> </span><span class="default_value"><span class="pre">1.0</span></span>*, *<span class="n"><span class="pre">reduction</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><a href="https://docs.python.org/3/library/typing.html#typing.Literal"
class="reference external" title="(in Python v3.13)"><span
class="pre">Literal</span></a><span class="p"><span class="pre">\[</span></span><span class="s"><span class="pre">'none'</span></span><span class="p"><span class="pre">,</span></span><span class="w"> </span><span class="s"><span class="pre">'mean'</span></span><span class="p"><span class="pre">,</span></span><span class="w"> </span><span class="s"><span class="pre">'sum'</span></span><span class="p"><span class="pre">\]</span></span></span><span class="w"> </span><span class="o"><span class="pre">=</span></span><span class="w"> </span><span class="default_value"><span class="pre">'mean'</span></span>*<span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#mlx.nn.losses.smooth_l1_loss"
class="headerlink" title="Link to this definition">#</a>  
Computes the smooth L1 loss.

The smooth L1 loss is a variant of the L1 loss which replaces the
absolute difference with a squared difference when the absolute
difference is less than <span class="pre">`beta`</span>.

The formula for the smooth L1 Loss is:

<div class="math notranslate nohighlight">

\\\begin{split}l = \begin{cases} 0.5 (x - y)^2 / \beta, & \text{if }
\|x - y\| \< \beta \\ \|x - y\| - 0.5 \beta, & \text{otherwise}
\end{cases}\end{split}\\

</div>

Parameters<span class="colon">:</span>  
- **predictions** (<a
  href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.html#mlx.core.array"
  class="reference internal" title="mlx.core.array"><em>array</em></a>)
  – Predicted values.

- **targets** (<a
  href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.html#mlx.core.array"
  class="reference internal" title="mlx.core.array"><em>array</em></a>)
  – Ground truth values.

- **beta**
  (<a href="https://docs.python.org/3/library/functions.html#float"
  class="reference external" title="(in Python v3.13)"><em>float</em></a>*,*
  *optional*) – The threshold after which the loss changes from the
  squared to the absolute difference. Default:
  <span class="pre">`1.0`</span>.

- **reduction**
  (<a href="https://docs.python.org/3/library/stdtypes.html#str"
  class="reference external" title="(in Python v3.13)"><em>str</em></a>*,*
  *optional*) – Specifies the reduction to apply to the output:
  <span class="pre">`'none'`</span> \| <span class="pre">`'mean'`</span>
  \| <span class="pre">`'sum'`</span>. Default:
  <span class="pre">`'mean'`</span>.

Returns<span class="colon">:</span>  
The computed smooth L1 loss.

Return type<span class="colon">:</span>  
<a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.html#mlx.core.array"
class="reference internal" title="mlx.core.array"><em>array</em></a>

</div>

<div class="prev-next-area">

<a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary_functions/mlx.nn.losses.nll_loss.html"
class="left-prev" title="previous page"><em></em></a>

<div class="prev-next-info">

previous

mlx.nn.losses.nll_loss

</div>

<a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary_functions/mlx.nn.losses.triplet_loss.html"
class="right-next" title="next page"></a>

<div class="prev-next-info">

next

mlx.nn.losses.triplet_loss

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
  href="https://ml-explore.github.io/mlx/build/html/#mlx.nn.losses.smooth_l1_loss"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">smooth_l1_loss</code></span></a>

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
