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
  href="https://ml-explore.github.io/mlx/build/html/_sources/python/nn/_autosummary/mlx.nn.QuantizedEmbedding.rst"
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

# mlx.nn.QuantizedEmbedding

<div id="print-main-content">

<div id="jb-print-toc">

<div>

## Contents

</div>

- <a
  href="https://ml-explore.github.io/mlx/build/html/#mlx.nn.QuantizedEmbedding"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">QuantizedEmbedding</code></span></a>

</div>

</div>

</div>

<div id="searchbox">

</div>

<div id="mlx-nn-quantizedembedding" class="section">

# mlx.nn.QuantizedEmbedding<a
href="https://ml-explore.github.io/mlx/build/html/#mlx-nn-quantizedembedding"
class="headerlink" title="Link to this heading">#</a>

*<span class="pre">class</span><span class="w"> </span>*<span class="sig-name descname"><span class="pre">QuantizedEmbedding</span></span><span class="sig-paren">(</span>*<span class="n"><span class="pre">num_embeddings</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><a href="https://docs.python.org/3/library/functions.html#int"
class="reference external" title="(in Python v3.13)"><span
class="pre">int</span></a></span>*, *<span class="n"><span class="pre">dims</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><a href="https://docs.python.org/3/library/functions.html#int"
class="reference external" title="(in Python v3.13)"><span
class="pre">int</span></a></span>*, *<span class="n"><span class="pre">group_size</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><a href="https://docs.python.org/3/library/functions.html#int"
class="reference external" title="(in Python v3.13)"><span
class="pre">int</span></a></span><span class="w"> </span><span class="o"><span class="pre">=</span></span><span class="w"> </span><span class="default_value"><span class="pre">64</span></span>*, *<span class="n"><span class="pre">bits</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><a href="https://docs.python.org/3/library/functions.html#int"
class="reference external" title="(in Python v3.13)"><span
class="pre">int</span></a></span><span class="w"> </span><span class="o"><span class="pre">=</span></span><span class="w"> </span><span class="default_value"><span class="pre">4</span></span>*<span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#mlx.nn.QuantizedEmbedding"
class="headerlink" title="Link to this definition">#</a>  
The same as <a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Embedding.html#mlx.nn.Embedding"
class="reference internal" title="mlx.nn.Embedding"><span
class="pre"><code class="sourceCode python">Embedding</code></span></a>
but with a quantized weight matrix.

<a
href="https://ml-explore.github.io/mlx/build/html/#mlx.nn.QuantizedEmbedding"
class="reference internal" title="mlx.nn.QuantizedEmbedding"><span
class="pre"><code
class="sourceCode python">QuantizedEmbedding</code></span></a> also
provides a <span class="pre">`from_embedding()`</span> classmethod to
convert embedding layers to <a
href="https://ml-explore.github.io/mlx/build/html/#mlx.nn.QuantizedEmbedding"
class="reference internal" title="mlx.nn.QuantizedEmbedding"><span
class="pre"><code
class="sourceCode python">QuantizedEmbedding</code></span></a> layers.

Parameters<span class="colon">:</span>  
- **num_embeddings**
  (<a href="https://docs.python.org/3/library/functions.html#int"
  class="reference external" title="(in Python v3.13)"><em>int</em></a>)
  – How many possible discrete tokens can we embed. Usually called the
  vocabulary size.

- **dims**
  (<a href="https://docs.python.org/3/library/functions.html#int"
  class="reference external" title="(in Python v3.13)"><em>int</em></a>)
  – The dimensionality of the embeddings.

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
| <span class="pre">`as_linear`</span>(x) | Call the quantized embedding layer as a quantized linear layer. |
| <span class="pre">`from_embedding`</span>(embedding_layer\[, ...\]) | Create a <a
href="https://ml-explore.github.io/mlx/build/html/#mlx.nn.QuantizedEmbedding"
class="reference internal" title="mlx.nn.QuantizedEmbedding"><span
class="pre"><code
class="sourceCode python">QuantizedEmbedding</code></span></a> layer from an <a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Embedding.html#mlx.nn.Embedding"
class="reference internal" title="mlx.nn.Embedding"><span
class="pre"><code class="sourceCode python">Embedding</code></span></a> layer. |

</div>

</div>

<div class="prev-next-area">

<a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.PReLU.html"
class="left-prev" title="previous page"><em></em></a>

<div class="prev-next-info">

previous

mlx.nn.PReLU

</div>

<a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.QuantizedLinear.html"
class="right-next" title="next page"></a>

<div class="prev-next-info">

next

mlx.nn.QuantizedLinear

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
  href="https://ml-explore.github.io/mlx/build/html/#mlx.nn.QuantizedEmbedding"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">QuantizedEmbedding</code></span></a>

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
