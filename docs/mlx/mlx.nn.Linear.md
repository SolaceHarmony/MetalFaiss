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
  href="https://ml-explore.github.io/mlx/build/html/_sources/python/nn/_autosummary/mlx.nn.Linear.rst"
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

# mlx.nn.Linear

<div id="print-main-content">

<div id="jb-print-toc">

<div>

## Contents

</div>

- <a href="https://ml-explore.github.io/mlx/build/html/#mlx.nn.Linear"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">Linear</code></span></a>

</div>

</div>

</div>

<div id="searchbox">

</div>

<div id="mlx-nn-linear" class="section">

# mlx.nn.Linear<a href="https://ml-explore.github.io/mlx/build/html/#mlx-nn-linear"
class="headerlink" title="Link to this heading">#</a>

*<span class="pre">class</span><span class="w"> </span>*<span class="sig-name descname"><span class="pre">Linear</span></span><span class="sig-paren">(</span>*<span class="n"><span class="pre">input_dims</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><a href="https://docs.python.org/3/library/functions.html#int"
class="reference external" title="(in Python v3.13)"><span
class="pre">int</span></a></span>*, *<span class="n"><span class="pre">output_dims</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><a href="https://docs.python.org/3/library/functions.html#int"
class="reference external" title="(in Python v3.13)"><span
class="pre">int</span></a></span>*, *<span class="n"><span class="pre">bias</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><a href="https://docs.python.org/3/library/functions.html#bool"
class="reference external" title="(in Python v3.13)"><span
class="pre">bool</span></a></span><span class="w"> </span><span class="o"><span class="pre">=</span></span><span class="w"> </span><span class="default_value"><span class="pre">True</span></span>*<span class="sig-paren">)</span><a href="https://ml-explore.github.io/mlx/build/html/#mlx.nn.Linear"
class="headerlink" title="Link to this definition">#</a>  
Applies an affine transformation to the input.

Concretely:

<div class="math notranslate nohighlight">

\\y = x W^\top + b\\

</div>

where: where <span class="math notranslate nohighlight">\\W\\</span> has
shape
<span class="pre">`[output_dims,`</span>` `<span class="pre">`input_dims]`</span>
and <span class="math notranslate nohighlight">\\b\\</span> has shape
<span class="pre">`[output_dims]`</span>.

The values are initialized from the uniform distribution
<span class="math notranslate nohighlight">\\\mathcal{U}(-{k},
{k})\\</span>, where <span class="math notranslate nohighlight">\\k =
\frac{1}{\sqrt{D_i}}\\</span> and
<span class="math notranslate nohighlight">\\D_i\\</span> is equal to
<span class="pre">`input_dims`</span>.

Parameters<span class="colon">:</span>  
- **input_dims**
  (<a href="https://docs.python.org/3/library/functions.html#int"
  class="reference external" title="(in Python v3.13)"><em>int</em></a>)
  – The dimensionality of the input features

- **output_dims**
  (<a href="https://docs.python.org/3/library/functions.html#int"
  class="reference external" title="(in Python v3.13)"><em>int</em></a>)
  – The dimensionality of the output features

- **bias**
  (<a href="https://docs.python.org/3/library/functions.html#bool"
  class="reference external" title="(in Python v3.13)"><em>bool</em></a>*,*
  *optional*) – If set to <span class="pre">`False`</span> then the
  layer will not use a bias. Default is <span class="pre">`True`</span>.

Methods

<div class="pst-scrollable-table-container">

|  |  |
|----|----|
| <span class="pre">`to_quantized`</span>(\[group_size, bits\]) | Return a <a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.QuantizedLinear.html#mlx.nn.QuantizedLinear"
class="reference internal" title="mlx.nn.QuantizedLinear"><span
class="pre"><code
class="sourceCode python">QuantizedLinear</code></span></a> layer that approximates this layer. |

</div>

</div>

<div class="prev-next-area">

<a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.LeakyReLU.html"
class="left-prev" title="previous page"><em></em></a>

<div class="prev-next-info">

previous

mlx.nn.LeakyReLU

</div>

<a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.LogSigmoid.html"
class="right-next" title="next page"></a>

<div class="prev-next-info">

next

mlx.nn.LogSigmoid

</div>

</div>

</div>

<div class="bd-sidebar-secondary bd-toc">

<div class="sidebar-secondary-items sidebar-secondary__inner">

<div class="sidebar-secondary-item">

<div class="page-toc tocsection onthispage">

Contents

</div>

- <a href="https://ml-explore.github.io/mlx/build/html/#mlx.nn.Linear"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">Linear</code></span></a>

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
