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
  href="https://ml-explore.github.io/mlx/build/html/_sources/python/nn/_autosummary/mlx.nn.MaxPool1d.rst"
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

# mlx.nn.MaxPool1d

<div id="print-main-content">

<div id="jb-print-toc">

<div>

## Contents

</div>

- <a href="https://ml-explore.github.io/mlx/build/html/#mlx.nn.MaxPool1d"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">MaxPool1d</code></span></a>

</div>

</div>

</div>

<div id="searchbox">

</div>

<div id="mlx-nn-maxpool1d" class="section">

# mlx.nn.MaxPool1d<a href="https://ml-explore.github.io/mlx/build/html/#mlx-nn-maxpool1d"
class="headerlink" title="Link to this heading">#</a>

*<span class="pre">class</span><span class="w"> </span>*<span class="sig-name descname"><span class="pre">MaxPool1d</span></span><span class="sig-paren">(</span>*<span class="n"><span class="pre">kernel_size</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><a href="https://docs.python.org/3/library/functions.html#int"
class="reference external" title="(in Python v3.13)"><span
class="pre">int</span></a><span class="w"> </span><span class="p"><span class="pre">\|</span></span><span class="w"> </span><a href="https://docs.python.org/3/library/typing.html#typing.Tuple"
class="reference external" title="(in Python v3.13)"><span
class="pre">Tuple</span></a><span class="p"><span class="pre">\[</span></span><a href="https://docs.python.org/3/library/functions.html#int"
class="reference external" title="(in Python v3.13)"><span
class="pre">int</span></a><span class="p"><span class="pre">\]</span></span></span>*, *<span class="n"><span class="pre">stride</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><a href="https://docs.python.org/3/library/functions.html#int"
class="reference external" title="(in Python v3.13)"><span
class="pre">int</span></a><span class="w"> </span><span class="p"><span class="pre">\|</span></span><span class="w"> </span><a href="https://docs.python.org/3/library/typing.html#typing.Tuple"
class="reference external" title="(in Python v3.13)"><span
class="pre">Tuple</span></a><span class="p"><span class="pre">\[</span></span><a href="https://docs.python.org/3/library/functions.html#int"
class="reference external" title="(in Python v3.13)"><span
class="pre">int</span></a><span class="p"><span class="pre">\]</span></span><span class="w"> </span><span class="p"><span class="pre">\|</span></span><span class="w"> </span><a href="https://docs.python.org/3/library/constants.html#None"
class="reference external" title="(in Python v3.13)"><span
class="pre">None</span></a></span><span class="w"> </span><span class="o"><span class="pre">=</span></span><span class="w"> </span><span class="default_value"><span class="pre">None</span></span>*, *<span class="n"><span class="pre">padding</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><a href="https://docs.python.org/3/library/functions.html#int"
class="reference external" title="(in Python v3.13)"><span
class="pre">int</span></a><span class="w"> </span><span class="p"><span class="pre">\|</span></span><span class="w"> </span><a href="https://docs.python.org/3/library/typing.html#typing.Tuple"
class="reference external" title="(in Python v3.13)"><span
class="pre">Tuple</span></a><span class="p"><span class="pre">\[</span></span><a href="https://docs.python.org/3/library/functions.html#int"
class="reference external" title="(in Python v3.13)"><span
class="pre">int</span></a><span class="p"><span class="pre">\]</span></span></span><span class="w"> </span><span class="o"><span class="pre">=</span></span><span class="w"> </span><span class="default_value"><span class="pre">0</span></span>*<span class="sig-paren">)</span><a href="https://ml-explore.github.io/mlx/build/html/#mlx.nn.MaxPool1d"
class="headerlink" title="Link to this definition">#</a>  
Applies 1-dimensional max pooling.

Spatially downsamples the input by taking the maximum of a sliding
window of size <span class="pre">`kernel_size`</span> and sliding stride
<span class="pre">`stride`</span>.

Parameters<span class="colon">:</span>  
- **kernel_size**
  (<a href="https://docs.python.org/3/library/functions.html#int"
  class="reference external" title="(in Python v3.13)"><em>int</em></a>
  *or* <a href="https://docs.python.org/3/library/stdtypes.html#tuple"
  class="reference external" title="(in Python v3.13)"><em>tuple</em></a>*(*<a href="https://docs.python.org/3/library/functions.html#int"
  class="reference external" title="(in Python v3.13)"><em>int</em></a>*)*)
  – The size of the pooling window kernel.

- **stride**
  (<a href="https://docs.python.org/3/library/functions.html#int"
  class="reference external" title="(in Python v3.13)"><em>int</em></a>
  *or* <a href="https://docs.python.org/3/library/stdtypes.html#tuple"
  class="reference external" title="(in Python v3.13)"><em>tuple</em></a>*(*<a href="https://docs.python.org/3/library/functions.html#int"
  class="reference external" title="(in Python v3.13)"><em>int</em></a>*),*
  *optional*) – The stride of the pooling window. Default:
  <span class="pre">`kernel_size`</span>.

- **padding**
  (<a href="https://docs.python.org/3/library/functions.html#int"
  class="reference external" title="(in Python v3.13)"><em>int</em></a>
  *or* <a href="https://docs.python.org/3/library/stdtypes.html#tuple"
  class="reference external" title="(in Python v3.13)"><em>tuple</em></a>*(*<a href="https://docs.python.org/3/library/functions.html#int"
  class="reference external" title="(in Python v3.13)"><em>int</em></a>*),*
  *optional*) – How much negative infinity padding to apply to the
  input. The padding amount is applied to both sides of the spatial
  axis. Default: <span class="pre">`0`</span>.

Examples

<div class="doctest highlight-default notranslate">

<div class="highlight">

    >>> import mlx.core as mx
    >>> import mlx.nn.layers as nn
    >>> x = mx.random.normal(shape=(4, 16, 5))
    >>> pool = nn.MaxPool1d(kernel_size=2, stride=2)
    >>> pool(x)

</div>

</div>

Methods

<div class="pst-scrollable-table-container">

</div>

</div>

<div class="prev-next-area">

<a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.LSTM.html"
class="left-prev" title="previous page"><em></em></a>

<div class="prev-next-info">

previous

mlx.nn.LSTM

</div>

<a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.MaxPool2d.html"
class="right-next" title="next page"></a>

<div class="prev-next-info">

next

mlx.nn.MaxPool2d

</div>

</div>

</div>

<div class="bd-sidebar-secondary bd-toc">

<div class="sidebar-secondary-items sidebar-secondary__inner">

<div class="sidebar-secondary-item">

<div class="page-toc tocsection onthispage">

Contents

</div>

- <a href="https://ml-explore.github.io/mlx/build/html/#mlx.nn.MaxPool1d"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">MaxPool1d</code></span></a>

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
