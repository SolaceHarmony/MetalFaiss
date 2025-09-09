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
  href="https://ml-explore.github.io/mlx/build/html/_sources/python/nn/_autosummary/mlx.nn.Transformer.rst"
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

# mlx.nn.Transformer

<div id="print-main-content">

<div id="jb-print-toc">

<div>

## Contents

</div>

- <a
  href="https://ml-explore.github.io/mlx/build/html/#mlx.nn.Transformer"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">Transformer</code></span></a>

</div>

</div>

</div>

<div id="searchbox">

</div>

<div id="mlx-nn-transformer" class="section">

# mlx.nn.Transformer<a
href="https://ml-explore.github.io/mlx/build/html/#mlx-nn-transformer"
class="headerlink" title="Link to this heading">#</a>

*<span class="pre">class</span><span class="w"> </span>*<span class="sig-name descname"><span class="pre">Transformer</span></span><span class="sig-paren">(</span>*<span class="pre">dims:</span> <span class="pre">int</span> <span class="pre">=</span> <span class="pre">512,</span> <span class="pre">num_heads:</span> <span class="pre">int</span> <span class="pre">=</span> <span class="pre">8,</span> <span class="pre">num_encoder_layers:</span> <span class="pre">int</span> <span class="pre">=</span> <span class="pre">6,</span> <span class="pre">num_decoder_layers:</span> <span class="pre">int</span> <span class="pre">=</span> <span class="pre">6,</span> <span class="pre">mlp_dims:</span> <span class="pre">int</span> <span class="pre">\|</span> <span class="pre">None</span> <span class="pre">=</span> <span class="pre">None,</span> <span class="pre">dropout:</span> <span class="pre">float</span> <span class="pre">=</span> <span class="pre">0.0,</span> <span class="pre">activation:</span> <span class="pre">~typing.Callable\[\[~typing.Any\],</span> <span class="pre">~typing.Any\]</span> <span class="pre">=</span> <span class="pre">\<mlx.gc_func</span> <span class="pre">object\>,</span> <span class="pre">custom_encoder:</span> <span class="pre">~typing.Any</span> <span class="pre">\|</span> <span class="pre">None</span> <span class="pre">=</span> <span class="pre">None,</span> <span class="pre">custom_decoder:</span> <span class="pre">~typing.Any</span> <span class="pre">\|</span> <span class="pre">None</span> <span class="pre">=</span> <span class="pre">None,</span> <span class="pre">norm_first:</span> <span class="pre">bool</span> <span class="pre">=</span> <span class="pre">True,</span> <span class="pre">checkpoint:</span> <span class="pre">bool</span> <span class="pre">=</span> <span class="pre">False</span>*<span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#mlx.nn.Transformer"
class="headerlink" title="Link to this definition">#</a>  
Implements a standard Transformer model.

The implementation is based on
<a href="https://arxiv.org/abs/1706.03762"
class="reference external">Attention Is All You Need</a>.

The Transformer model contains an encoder and a decoder. The encoder
processes the input sequence and the decoder generates the output
sequence. The interaction between encoder and decoder happens through
the attention mechanism.

Parameters<span class="colon">:</span>  
- **dims**
  (<a href="https://docs.python.org/3/library/functions.html#int"
  class="reference external" title="(in Python v3.13)"><em>int</em></a>*,*
  *optional*) – The number of expected features in the encoder/decoder
  inputs. Default: <span class="pre">`512`</span>.

- **num_heads**
  (<a href="https://docs.python.org/3/library/functions.html#int"
  class="reference external" title="(in Python v3.13)"><em>int</em></a>*,*
  *optional*) – The number of attention heads. Default:
  <span class="pre">`8`</span>.

- **num_encoder_layers**
  (<a href="https://docs.python.org/3/library/functions.html#int"
  class="reference external" title="(in Python v3.13)"><em>int</em></a>*,*
  *optional*) – The number of encoder layers in the Transformer encoder.
  Default: <span class="pre">`6`</span>.

- **num_decoder_layers**
  (<a href="https://docs.python.org/3/library/functions.html#int"
  class="reference external" title="(in Python v3.13)"><em>int</em></a>*,*
  *optional*) – The number of decoder layers in the Transformer decoder.
  Default: <span class="pre">`6`</span>.

- **mlp_dims**
  (<a href="https://docs.python.org/3/library/functions.html#int"
  class="reference external" title="(in Python v3.13)"><em>int</em></a>*,*
  *optional*) – The hidden dimension of the MLP block in each
  Transformer layer. Defaults to <span class="pre">`4*dims`</span> if
  not provided. Default: <span class="pre">`None`</span>.

- **dropout**
  (<a href="https://docs.python.org/3/library/functions.html#float"
  class="reference external" title="(in Python v3.13)"><em>float</em></a>*,*
  *optional*) – The dropout value for the Transformer encoder and
  decoder. Dropout is used after each attention layer and the activation
  in the MLP layer. Default: <span class="pre">`0.0`</span>.

- **activation** (*function,* *optional*) – the activation function for
  the MLP hidden layer. Default: <a
  href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary_functions/mlx.nn.relu.html#mlx.nn.relu"
  class="reference internal" title="mlx.nn.relu"><span class="pre"><code
  class="sourceCode python">mlx.nn.relu()</code></span></a>.

- **custom_encoder** (<a
  href="https://ml-explore.github.io/mlx/build/html/python/nn/module.html#mlx.nn.Module"
  class="reference internal" title="mlx.nn.Module"><em>Module</em></a>*,*
  *optional*) – A custom encoder to replace the standard Transformer
  encoder. Default: <span class="pre">`None`</span>.

- **custom_decoder** (<a
  href="https://ml-explore.github.io/mlx/build/html/python/nn/module.html#mlx.nn.Module"
  class="reference internal" title="mlx.nn.Module"><em>Module</em></a>*,*
  *optional*) – A custom decoder to replace the standard Transformer
  decoder. Default: <span class="pre">`None`</span>.

- **norm_first**
  (<a href="https://docs.python.org/3/library/functions.html#bool"
  class="reference external" title="(in Python v3.13)"><em>bool</em></a>*,*
  *optional*) – if <span class="pre">`True`</span>, encoder and decoder
  layers will perform layer normalization before attention and MLP
  operations, otherwise after. Default: <span class="pre">`True`</span>.

- **checkpoint**
  (<a href="https://docs.python.org/3/library/functions.html#bool"
  class="reference external" title="(in Python v3.13)"><em>bool</em></a>*,*
  *optional*) – if <span class="pre">`True`</span> perform gradient
  checkpointing to reduce the memory usage at the expense of more
  computation. Default: <span class="pre">`False`</span>.

Methods

<div class="pst-scrollable-table-container">

</div>

</div>

<div class="prev-next-area">

<a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Tanh.html"
class="left-prev" title="previous page"><em></em></a>

<div class="prev-next-info">

previous

mlx.nn.Tanh

</div>

<a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Upsample.html"
class="right-next" title="next page"></a>

<div class="prev-next-info">

next

mlx.nn.Upsample

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
  href="https://ml-explore.github.io/mlx/build/html/#mlx.nn.Transformer"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">Transformer</code></span></a>

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
