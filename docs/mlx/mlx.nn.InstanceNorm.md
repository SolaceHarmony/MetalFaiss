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
  href="https://ml-explore.github.io/mlx/build/html/_sources/python/nn/_autosummary/mlx.nn.InstanceNorm.rst"
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

# mlx.nn.InstanceNorm

<div id="print-main-content">

<div id="jb-print-toc">

<div>

## Contents

</div>

- <a
  href="https://ml-explore.github.io/mlx/build/html/#mlx.nn.InstanceNorm"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">InstanceNorm</code></span></a>

</div>

</div>

</div>

<div id="searchbox">

</div>

<div id="mlx-nn-instancenorm" class="section">

# mlx.nn.InstanceNorm<a
href="https://ml-explore.github.io/mlx/build/html/#mlx-nn-instancenorm"
class="headerlink" title="Link to this heading">#</a>

*<span class="pre">class</span><span class="w"> </span>*<span class="sig-name descname"><span class="pre">InstanceNorm</span></span><span class="sig-paren">(</span>*<span class="n"><span class="pre">dims</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><a href="https://docs.python.org/3/library/functions.html#int"
class="reference external" title="(in Python v3.13)"><span
class="pre">int</span></a></span>*, *<span class="n"><span class="pre">eps</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><a href="https://docs.python.org/3/library/functions.html#float"
class="reference external" title="(in Python v3.13)"><span
class="pre">float</span></a></span><span class="w"> </span><span class="o"><span class="pre">=</span></span><span class="w"> </span><span class="default_value"><span class="pre">1e-05</span></span>*, *<span class="n"><span class="pre">affine</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><a href="https://docs.python.org/3/library/functions.html#bool"
class="reference external" title="(in Python v3.13)"><span
class="pre">bool</span></a></span><span class="w"> </span><span class="o"><span class="pre">=</span></span><span class="w"> </span><span class="default_value"><span class="pre">False</span></span>*<span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#mlx.nn.InstanceNorm"
class="headerlink" title="Link to this definition">#</a>  
Applies instance normalization \[1\] on the inputs.

Computes

<div class="math notranslate nohighlight">

\\y = \frac{x - \mathrm{E}\[x\]}{ \sqrt{\mathrm{Var}\[x\] + \epsilon}}
\* \gamma + \beta,\\

</div>

where <span class="math notranslate nohighlight">\\\gamma\\</span> and
<span class="math notranslate nohighlight">\\\beta\\</span> are learned
per feature dimension parameters initialized at 1 and 0 respectively.
Both are of size <span class="pre">`dims`</span>, if
<span class="pre">`affine`</span> is <span class="pre">`True`</span>.

Parameters<span class="colon">:</span>  
- **dims**
  (<a href="https://docs.python.org/3/library/functions.html#int"
  class="reference external" title="(in Python v3.13)"><em>int</em></a>)
  – The number of features of the input.

- **eps**
  (<a href="https://docs.python.org/3/library/functions.html#float"
  class="reference external" title="(in Python v3.13)"><em>float</em></a>)
  – A value added to the denominator for numerical stability. Default:
  <span class="pre">`1e-5`</span>.

- **affine**
  (<a href="https://docs.python.org/3/library/functions.html#bool"
  class="reference external" title="(in Python v3.13)"><em>bool</em></a>)
  – Default: <span class="pre">`False`</span>.

Shape:  
- Input: <span class="math notranslate nohighlight">\\(..., C)\\</span>
  where <span class="math notranslate nohighlight">\\C\\</span> is equal
  to <span class="pre">`dims`</span>.

- Output: Same shape as the input.

Examples

<div class="doctest highlight-default notranslate">

<div class="highlight">

    >>> import mlx.core as mx
    >>> import mlx.nn as nn
    >>> x = mx.random.normal((8, 4, 4, 16))
    >>> inorm = nn.InstanceNorm(dims=16)
    >>> output = inorm(x)

</div>

</div>

References

\[1\]: <a href="https://arxiv.org/abs/1607.08022"
class="reference external">https://arxiv.org/abs/1607.08022</a>

Methods

<div class="pst-scrollable-table-container">

</div>

</div>

<div class="prev-next-area">

<a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Hardswish.html"
class="left-prev" title="previous page"><em></em></a>

<div class="prev-next-info">

previous

mlx.nn.Hardswish

</div>

<a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.LayerNorm.html"
class="right-next" title="next page"></a>

<div class="prev-next-info">

next

mlx.nn.LayerNorm

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
  href="https://ml-explore.github.io/mlx/build/html/#mlx.nn.InstanceNorm"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">InstanceNorm</code></span></a>

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
