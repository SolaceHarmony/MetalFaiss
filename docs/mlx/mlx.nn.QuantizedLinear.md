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
  href="https://ml-explore.github.io/mlx/build/html/_sources/python/nn/_autosummary/mlx.nn.QuantizedLinear.rst"
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

# mlx.nn.QuantizedLinear

<div id="print-main-content">

<div id="jb-print-toc">

<div>

## Contents

</div>

- <a
  href="https://ml-explore.github.io/mlx/build/html/#mlx.nn.QuantizedLinear"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">QuantizedLinear</code></span></a>

</div>

</div>

</div>

<div id="searchbox">

</div>

<div id="mlx-nn-quantizedlinear" class="section">

# mlx.nn.QuantizedLinear<a
href="https://ml-explore.github.io/mlx/build/html/#mlx-nn-quantizedlinear"
class="headerlink" title="Link to this heading">#</a>

*<span class="pre">class</span><span class="w"> </span>*<span class="sig-name descname"><span class="pre">QuantizedLinear</span></span><span class="sig-paren">(</span>*<span class="n"><span class="pre">input_dims</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><a href="https://docs.python.org/3/library/functions.html#int"
class="reference external" title="(in Python v3.13)"><span
class="pre">int</span></a></span>*, *<span class="n"><span class="pre">output_dims</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><a href="https://docs.python.org/3/library/functions.html#int"
class="reference external" title="(in Python v3.13)"><span
class="pre">int</span></a></span>*, *<span class="n"><span class="pre">bias</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><a href="https://docs.python.org/3/library/functions.html#bool"
class="reference external" title="(in Python v3.13)"><span
class="pre">bool</span></a></span><span class="w"> </span><span class="o"><span class="pre">=</span></span><span class="w"> </span><span class="default_value"><span class="pre">True</span></span>*, *<span class="n"><span class="pre">group_size</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><a href="https://docs.python.org/3/library/functions.html#int"
class="reference external" title="(in Python v3.13)"><span
class="pre">int</span></a></span><span class="w"> </span><span class="o"><span class="pre">=</span></span><span class="w"> </span><span class="default_value"><span class="pre">64</span></span>*, *<span class="n"><span class="pre">bits</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><a href="https://docs.python.org/3/library/functions.html#int"
class="reference external" title="(in Python v3.13)"><span
class="pre">int</span></a></span><span class="w"> </span><span class="o"><span class="pre">=</span></span><span class="w"> </span><span class="default_value"><span class="pre">4</span></span>*<span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#mlx.nn.QuantizedLinear"
class="headerlink" title="Link to this definition">#</a>  
Applies an affine transformation to the input using a quantized weight
matrix.

It is the quantized equivalent of <a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Linear.html#mlx.nn.Linear"
class="reference internal" title="mlx.nn.Linear"><span class="pre"><code
class="sourceCode python">mlx.nn.Linear</code></span></a>. For now its
parameters are frozen and will not be included in any gradient
computation but this will probably change in the future.

<a
href="https://ml-explore.github.io/mlx/build/html/#mlx.nn.QuantizedLinear"
class="reference internal" title="mlx.nn.QuantizedLinear"><span
class="pre"><code
class="sourceCode python">QuantizedLinear</code></span></a> also
provides a classmethod <span class="pre">`from_linear()`</span> to
convert linear layers to <a
href="https://ml-explore.github.io/mlx/build/html/#mlx.nn.QuantizedLinear"
class="reference internal" title="mlx.nn.QuantizedLinear"><span
class="pre"><code
class="sourceCode python">QuantizedLinear</code></span></a> layers.

Parameters<span class="colon">:</span>  
- **input_dims**
  (<a href="https://docs.python.org/3/library/functions.html#int"
  class="reference external" title="(in Python v3.13)"><em>int</em></a>)
  – The dimensionality of the input features.

- **output_dims**
  (<a href="https://docs.python.org/3/library/functions.html#int"
  class="reference external" title="(in Python v3.13)"><em>int</em></a>)
  – The dimensionality of the output features.

- **bias**
  (<a href="https://docs.python.org/3/library/functions.html#bool"
  class="reference external" title="(in Python v3.13)"><em>bool</em></a>*,*
  *optional*) – If set to <span class="pre">`False`</span> then the
  layer will not use a bias. Default: <span class="pre">`True`</span>.

- **group_size**
  (<a href="https://docs.python.org/3/library/functions.html#int"
  class="reference external" title="(in Python v3.13)"><em>int</em></a>*,*
  *optional*) – The group size to use for the quantized weight. See <a
  href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.quantize.html#mlx.core.quantize"
  class="reference internal" title="mlx.core.quantize"><span
  class="pre"><code class="sourceCode python">quantize()</code></span></a>.
  Default: <span class="pre">`64`</span>.

- **bits**
  (<a href="https://docs.python.org/3/library/functions.html#int"
  class="reference external" title="(in Python v3.13)"><em>int</em></a>*,*
  *optional*) – The bit width to use for the quantized weight. See <a
  href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.quantize.html#mlx.core.quantize"
  class="reference internal" title="mlx.core.quantize"><span
  class="pre"><code class="sourceCode python">quantize()</code></span></a>.
  Default: <span class="pre">`4`</span>.

Methods

<div class="pst-scrollable-table-container">

|  |  |
|----|----|
| <span class="pre">`from_linear`</span>(linear_layer\[, group_size, bits\]) | Create a <a
href="https://ml-explore.github.io/mlx/build/html/#mlx.nn.QuantizedLinear"
class="reference internal" title="mlx.nn.QuantizedLinear"><span
class="pre"><code
class="sourceCode python">QuantizedLinear</code></span></a> layer from a <a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Linear.html#mlx.nn.Linear"
class="reference internal" title="mlx.nn.Linear"><span class="pre"><code
class="sourceCode python">Linear</code></span></a> layer. |
| <span class="pre">`unfreeze`</span>(\*args, \*\*kwargs) | Wrap unfreeze so that we unfreeze any layers we might contain but our parameters will remain frozen. |

</div>

</div>

<div class="prev-next-area">

<a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.QuantizedEmbedding.html"
class="left-prev" title="previous page"><em></em></a>

<div class="prev-next-info">

previous

mlx.nn.QuantizedEmbedding

</div>

<a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.RMSNorm.html"
class="right-next" title="next page"></a>

<div class="prev-next-info">

next

mlx.nn.RMSNorm

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
  href="https://ml-explore.github.io/mlx/build/html/#mlx.nn.QuantizedLinear"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">QuantizedLinear</code></span></a>

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
