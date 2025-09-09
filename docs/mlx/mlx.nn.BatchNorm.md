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
  href="https://ml-explore.github.io/mlx/build/html/_sources/python/nn/_autosummary/mlx.nn.BatchNorm.rst"
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

# mlx.nn.BatchNorm

<div id="print-main-content">

<div id="jb-print-toc">

<div>

## Contents

</div>

- <a href="https://ml-explore.github.io/mlx/build/html/#mlx.nn.BatchNorm"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">BatchNorm</code></span></a>

</div>

</div>

</div>

<div id="searchbox">

</div>

<div id="mlx-nn-batchnorm" class="section">

# mlx.nn.BatchNorm<a href="https://ml-explore.github.io/mlx/build/html/#mlx-nn-batchnorm"
class="headerlink" title="Link to this heading">#</a>

*<span class="pre">class</span><span class="w"> </span>*<span class="sig-name descname"><span class="pre">BatchNorm</span></span><span class="sig-paren">(</span>*<span class="n"><span class="pre">num_features</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><a href="https://docs.python.org/3/library/functions.html#int"
class="reference external" title="(in Python v3.13)"><span
class="pre">int</span></a></span>*, *<span class="n"><span class="pre">eps</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><a href="https://docs.python.org/3/library/functions.html#float"
class="reference external" title="(in Python v3.13)"><span
class="pre">float</span></a></span><span class="w"> </span><span class="o"><span class="pre">=</span></span><span class="w"> </span><span class="default_value"><span class="pre">1e-05</span></span>*, *<span class="n"><span class="pre">momentum</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><a href="https://docs.python.org/3/library/functions.html#float"
class="reference external" title="(in Python v3.13)"><span
class="pre">float</span></a></span><span class="w"> </span><span class="o"><span class="pre">=</span></span><span class="w"> </span><span class="default_value"><span class="pre">0.1</span></span>*, *<span class="n"><span class="pre">affine</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><a href="https://docs.python.org/3/library/functions.html#bool"
class="reference external" title="(in Python v3.13)"><span
class="pre">bool</span></a></span><span class="w"> </span><span class="o"><span class="pre">=</span></span><span class="w"> </span><span class="default_value"><span class="pre">True</span></span>*, *<span class="n"><span class="pre">track_running_stats</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><a href="https://docs.python.org/3/library/functions.html#bool"
class="reference external" title="(in Python v3.13)"><span
class="pre">bool</span></a></span><span class="w"> </span><span class="o"><span class="pre">=</span></span><span class="w"> </span><span class="default_value"><span class="pre">True</span></span>*<span class="sig-paren">)</span><a href="https://ml-explore.github.io/mlx/build/html/#mlx.nn.BatchNorm"
class="headerlink" title="Link to this definition">#</a>  
Applies Batch Normalization over a 2D or 3D input.

Computes

<div class="math notranslate nohighlight">

\\y = \frac{x - E\[x\]}{\sqrt{Var\[x\]} + \epsilon} \gamma + \beta,\\

</div>

where <span class="math notranslate nohighlight">\\\gamma\\</span> and
<span class="math notranslate nohighlight">\\\beta\\</span> are learned
per feature dimension parameters initialized at 1 and 0 respectively.

The input shape is specified as <span class="pre">`NC`</span> or
<span class="pre">`NLC`</span>, where <span class="pre">`N`</span> is
the batch, <span class="pre">`C`</span> is the number of features or
channels, and <span class="pre">`L`</span> is the sequence length. The
output has the same shape as the input. For four-dimensional arrays, the
shape is <span class="pre">`NHWC`</span>, where
<span class="pre">`H`</span> and <span class="pre">`W`</span> are the
height and width respectively.

For more information on Batch Normalization, see the original paper
<a href="https://arxiv.org/abs/1502.03167"
class="reference external">Batch Normalization: Accelerating Deep
Network Training by Reducing Internal Covariate Shift</a>.

Parameters<span class="colon">:</span>  
- **num_features**
  (<a href="https://docs.python.org/3/library/functions.html#int"
  class="reference external" title="(in Python v3.13)"><em>int</em></a>)
  – The feature dimension to normalize over.

- **eps**
  (<a href="https://docs.python.org/3/library/functions.html#float"
  class="reference external" title="(in Python v3.13)"><em>float</em></a>*,*
  *optional*) – A small additive constant for numerical stability.
  Default: <span class="pre">`1e-5`</span>.

- **momentum**
  (<a href="https://docs.python.org/3/library/functions.html#float"
  class="reference external" title="(in Python v3.13)"><em>float</em></a>*,*
  *optional*) – The momentum for updating the running mean and variance.
  Default: <span class="pre">`0.1`</span>.

- **affine**
  (<a href="https://docs.python.org/3/library/functions.html#bool"
  class="reference external" title="(in Python v3.13)"><em>bool</em></a>*,*
  *optional*) – If <span class="pre">`True`</span>, apply a learned
  affine transformation after the normalization. Default:
  <span class="pre">`True`</span>.

- **track_running_stats**
  (<a href="https://docs.python.org/3/library/functions.html#bool"
  class="reference external" title="(in Python v3.13)"><em>bool</em></a>*,*
  *optional*) – If <span class="pre">`True`</span>, track the running
  mean and variance. Default: <span class="pre">`True`</span>.

Examples

<div class="doctest highlight-default notranslate">

<div class="highlight">

    >>> import mlx.core as mx
    >>> import mlx.nn as nn
    >>> x = mx.random.normal((5, 4))
    >>> bn = nn.BatchNorm(num_features=4, affine=True)
    >>> output = bn(x)

</div>

</div>

Methods

<div class="pst-scrollable-table-container">

|  |  |
|----|----|
| <span class="pre">`unfreeze`</span>(\*args, \*\*kwargs) | Wrap unfreeze to make sure that running_mean and var are always frozen parameters. |

</div>

</div>

<div class="prev-next-area">

<a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.AvgPool3d.html"
class="left-prev" title="previous page"><em></em></a>

<div class="prev-next-info">

previous

mlx.nn.AvgPool3d

</div>

<a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.CELU.html"
class="right-next" title="next page"></a>

<div class="prev-next-info">

next

mlx.nn.CELU

</div>

</div>

</div>

<div class="bd-sidebar-secondary bd-toc">

<div class="sidebar-secondary-items sidebar-secondary__inner">

<div class="sidebar-secondary-item">

<div class="page-toc tocsection onthispage">

Contents

</div>

- <a href="https://ml-explore.github.io/mlx/build/html/#mlx.nn.BatchNorm"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">BatchNorm</code></span></a>

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
