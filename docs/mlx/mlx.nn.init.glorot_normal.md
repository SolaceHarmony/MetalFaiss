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
  href="https://ml-explore.github.io/mlx/build/html/_sources/python/nn/_autosummary/mlx.nn.init.glorot_normal.rst"
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

# mlx.nn.init.glorot_normal

<div id="print-main-content">

<div id="jb-print-toc">

<div>

## Contents

</div>

- <a
  href="https://ml-explore.github.io/mlx/build/html/#mlx.nn.init.glorot_normal"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">glorot_normal()</code></span></a>

</div>

</div>

</div>

<div id="searchbox">

</div>

<div id="mlx-nn-init-glorot-normal" class="section">

# mlx.nn.init.glorot_normal<a
href="https://ml-explore.github.io/mlx/build/html/#mlx-nn-init-glorot-normal"
class="headerlink" title="Link to this heading">#</a>

<span class="sig-name descname"><span class="pre">glorot_normal</span></span><span class="sig-paren">(</span>*<span class="n"><span class="pre">dtype</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.Dtype.html#mlx.core.Dtype"
class="reference internal" title="mlx.core.Dtype"><span
class="pre">Dtype</span></a></span><span class="w"> </span><span class="o"><span class="pre">=</span></span><span class="w"> </span><span class="default_value"><span class="pre">mlx.core.float32</span></span>*<span class="sig-paren">)</span> <span class="sig-return"><span class="sig-return-icon">→</span> <span class="sig-return-typehint"><a href="https://docs.python.org/3/library/typing.html#typing.Callable"
class="reference external" title="(in Python v3.13)"><span
class="pre">Callable</span></a><span class="p"><span class="pre">\[</span></span><span class="p"><span class="pre">\[</span></span><a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.html#mlx.core.array"
class="reference internal" title="mlx.core.array"><span
class="pre">array</span></a><span class="p"><span class="pre">,</span></span><span class="w"> </span><a href="https://docs.python.org/3/library/functions.html#float"
class="reference external" title="(in Python v3.13)"><span
class="pre">float</span></a><span class="p"><span class="pre">\]</span></span><span class="p"><span class="pre">,</span></span><span class="w"> </span><a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.html#mlx.core.array"
class="reference internal" title="mlx.core.array"><span
class="pre">array</span></a><span class="p"><span class="pre">\]</span></span></span></span><a
href="https://ml-explore.github.io/mlx/build/html/#mlx.nn.init.glorot_normal"
class="headerlink" title="Link to this definition">#</a>  
A Glorot normal initializer.

This initializer samples from a normal distribution with a standard
deviation computed from the number of input
(<span class="pre">`fan_in`</span>) and output
(<span class="pre">`fan_out`</span>) units according to:

<div class="math notranslate nohighlight">

\\\sigma = \gamma \sqrt{\frac{2.0}{\text{fan\\in} + \text{fan\\out}}}\\

</div>

For more details see the original reference:
<a href="https://proceedings.mlr.press/v9/glorot10a.html"
class="reference external">Understanding the difficulty of training deep
feedforward neural networks</a>

Parameters<span class="colon">:</span>  
**dtype** (<a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.Dtype.html#mlx.core.Dtype"
class="reference internal" title="mlx.core.Dtype"><em>Dtype</em></a>*,*
*optional*) – The data type of the array. Default:
<span class="pre">`float32`</span>.

Returns<span class="colon">:</span>  
An initializer that returns an array with the same shape as the input,
filled with samples from the Glorot normal distribution.

Return type<span class="colon">:</span>  
*Callable*\[\[<a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.html#mlx.core.array"
class="reference internal" title="mlx.core.array"><em>array</em></a>,
<a href="https://docs.python.org/3/library/functions.html#float"
class="reference external" title="(in Python v3.13)"><em>float</em></a>\],
<a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.html#mlx.core.array"
class="reference internal" title="mlx.core.array"><em>array</em></a>\]

Example

<div class="doctest highlight-default notranslate">

<div class="highlight">

    >>> init_fn = nn.init.glorot_normal()
    >>> init_fn(mx.zeros((2, 2)))
    array([[0.191107, 1.61278],
           [-0.150594, -0.363207]], dtype=float32)
    >>> init_fn(mx.zeros((2, 2)), gain=4.0)
    array([[1.89613, -4.53947],
           [4.48095, 0.995016]], dtype=float32)

</div>

</div>

</div>

<div class="prev-next-area">

<a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.init.identity.html"
class="left-prev" title="previous page"><em></em></a>

<div class="prev-next-info">

previous

mlx.nn.init.identity

</div>

<a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.init.glorot_uniform.html"
class="right-next" title="next page"></a>

<div class="prev-next-info">

next

mlx.nn.init.glorot_uniform

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
  href="https://ml-explore.github.io/mlx/build/html/#mlx.nn.init.glorot_normal"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">glorot_normal()</code></span></a>

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
