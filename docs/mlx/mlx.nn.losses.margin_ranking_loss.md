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
  href="https://ml-explore.github.io/mlx/build/html/_sources/python/nn/_autosummary_functions/mlx.nn.losses.margin_ranking_loss.rst"
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

# mlx.nn.losses.margin_ranking_loss

<div id="print-main-content">

<div id="jb-print-toc">

<div>

## Contents

</div>

- <a
  href="https://ml-explore.github.io/mlx/build/html/#mlx.nn.losses.margin_ranking_loss"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">margin_ranking_loss</code></span></a>

</div>

</div>

</div>

<div id="searchbox">

</div>

<div id="mlx-nn-losses-margin-ranking-loss" class="section">

# mlx.nn.losses.margin_ranking_loss<a
href="https://ml-explore.github.io/mlx/build/html/#mlx-nn-losses-margin-ranking-loss"
class="headerlink" title="Link to this heading">#</a>

*<span class="pre">class</span><span class="w"> </span>*<span class="sig-name descname"><span class="pre">margin_ranking_loss</span></span><span class="sig-paren">(</span>*<span class="n"><span class="pre">inputs1</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.html#mlx.core.array"
class="reference internal" title="mlx.core.array"><span
class="pre">array</span></a></span>*, *<span class="n"><span class="pre">inputs2</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.html#mlx.core.array"
class="reference internal" title="mlx.core.array"><span
class="pre">array</span></a></span>*, *<span class="n"><span class="pre">targets</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.html#mlx.core.array"
class="reference internal" title="mlx.core.array"><span
class="pre">array</span></a></span>*, *<span class="n"><span class="pre">margin</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><a href="https://docs.python.org/3/library/functions.html#float"
class="reference external" title="(in Python v3.13)"><span
class="pre">float</span></a></span><span class="w"> </span><span class="o"><span class="pre">=</span></span><span class="w"> </span><span class="default_value"><span class="pre">0.0</span></span>*, *<span class="n"><span class="pre">reduction</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><a href="https://docs.python.org/3/library/typing.html#typing.Literal"
class="reference external" title="(in Python v3.13)"><span
class="pre">Literal</span></a><span class="p"><span class="pre">\[</span></span><span class="s"><span class="pre">'none'</span></span><span class="p"><span class="pre">,</span></span><span class="w"> </span><span class="s"><span class="pre">'mean'</span></span><span class="p"><span class="pre">,</span></span><span class="w"> </span><span class="s"><span class="pre">'sum'</span></span><span class="p"><span class="pre">\]</span></span></span><span class="w"> </span><span class="o"><span class="pre">=</span></span><span class="w"> </span><span class="default_value"><span class="pre">'none'</span></span>*<span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#mlx.nn.losses.margin_ranking_loss"
class="headerlink" title="Link to this definition">#</a>  
Calculate the margin ranking loss that loss given inputs
<span class="math notranslate nohighlight">\\x_1\\</span>,
<span class="math notranslate nohighlight">\\x_2\\</span> and a label
<span class="math notranslate nohighlight">\\y\\</span> (containing 1 or
-1).

The loss is given by:

<div class="math notranslate nohighlight">

\\\text{loss} = \max (0, -y \* (x_1 - x_2) + \text{margin})\\

</div>

Where <span class="math notranslate nohighlight">\\y\\</span> represents
<span class="pre">`targets`</span>,
<span class="math notranslate nohighlight">\\x_1\\</span> represents
<span class="pre">`inputs1`</span> and
<span class="math notranslate nohighlight">\\x_2\\</span> represents
<span class="pre">`inputs2`</span>.

Parameters<span class="colon">:</span>  
- **inputs1** (<a
  href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.html#mlx.core.array"
  class="reference internal" title="mlx.core.array"><em>array</em></a>)
  – Scores for the first input.

- **inputs2** (<a
  href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.html#mlx.core.array"
  class="reference internal" title="mlx.core.array"><em>array</em></a>)
  – Scores for the second input.

- **targets** (<a
  href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.html#mlx.core.array"
  class="reference internal" title="mlx.core.array"><em>array</em></a>)
  – Labels indicating whether samples in
  <span class="pre">`inputs1`</span> should be ranked higher than
  samples in <span class="pre">`inputs2`</span>. Values should be 1 or
  -1.

- **margin**
  (<a href="https://docs.python.org/3/library/functions.html#float"
  class="reference external" title="(in Python v3.13)"><em>float</em></a>*,*
  *optional*) – The margin by which the scores should be separated.
  Default: <span class="pre">`0.0`</span>.

- **reduction**
  (<a href="https://docs.python.org/3/library/stdtypes.html#str"
  class="reference external" title="(in Python v3.13)"><em>str</em></a>*,*
  *optional*) – Specifies the reduction to apply to the output:
  <span class="pre">`'none'`</span> \| <span class="pre">`'mean'`</span>
  \| <span class="pre">`'sum'`</span>. Default:
  <span class="pre">`'none'`</span>.

Returns<span class="colon">:</span>  
The computed margin ranking loss.

Return type<span class="colon">:</span>  
<a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.html#mlx.core.array"
class="reference internal" title="mlx.core.array"><em>array</em></a>

Examples

<div class="doctest highlight-default notranslate">

<div class="highlight">

    >>> import mlx.core as mx
    >>> import mlx.nn as nn
    >>> targets = mx.array([1, 1, -1])
    >>> inputs1 = mx.array([-0.573409, -0.765166, -0.0638])
    >>> inputs2 = mx.array([0.75596, 0.225763, 0.256995])
    >>> loss = nn.losses.margin_ranking_loss(inputs1, inputs2, targets)
    >>> loss
    array(0.773433, dtype=float32)

</div>

</div>

</div>

<div class="prev-next-area">

<a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary_functions/mlx.nn.losses.log_cosh_loss.html"
class="left-prev" title="previous page"><em></em></a>

<div class="prev-next-info">

previous

mlx.nn.losses.log_cosh_loss

</div>

<a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary_functions/mlx.nn.losses.mse_loss.html"
class="right-next" title="next page"></a>

<div class="prev-next-info">

next

mlx.nn.losses.mse_loss

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
  href="https://ml-explore.github.io/mlx/build/html/#mlx.nn.losses.margin_ranking_loss"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">margin_ranking_loss</code></span></a>

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
