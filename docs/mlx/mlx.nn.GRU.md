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
  href="https://ml-explore.github.io/mlx/build/html/_sources/python/nn/_autosummary/mlx.nn.GRU.rst"
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

# mlx.nn.GRU

<div id="print-main-content">

<div id="jb-print-toc">

<div>

## Contents

</div>

- <a href="https://ml-explore.github.io/mlx/build/html/#mlx.nn.GRU"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">GRU</code></span></a>

</div>

</div>

</div>

<div id="searchbox">

</div>

<div id="mlx-nn-gru" class="section">

# mlx.nn.GRU<a href="https://ml-explore.github.io/mlx/build/html/#mlx-nn-gru"
class="headerlink" title="Link to this heading">#</a>

*<span class="pre">class</span><span class="w"> </span>*<span class="sig-name descname"><span class="pre">GRU</span></span><span class="sig-paren">(</span>*<span class="n"><span class="pre">input_size</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><a href="https://docs.python.org/3/library/functions.html#int"
class="reference external" title="(in Python v3.13)"><span
class="pre">int</span></a></span>*, *<span class="n"><span class="pre">hidden_size</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><a href="https://docs.python.org/3/library/functions.html#int"
class="reference external" title="(in Python v3.13)"><span
class="pre">int</span></a></span>*, *<span class="n"><span class="pre">bias</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><a href="https://docs.python.org/3/library/functions.html#bool"
class="reference external" title="(in Python v3.13)"><span
class="pre">bool</span></a></span><span class="w"> </span><span class="o"><span class="pre">=</span></span><span class="w"> </span><span class="default_value"><span class="pre">True</span></span>*<span class="sig-paren">)</span><a href="https://ml-explore.github.io/mlx/build/html/#mlx.nn.GRU"
class="headerlink" title="Link to this definition">#</a>  
A gated recurrent unit (GRU) RNN layer.

The input has shape <span class="pre">`NLD`</span> or
<span class="pre">`LD`</span> where:

- <span class="pre">`N`</span> is the optional batch dimension

- <span class="pre">`L`</span> is the sequence length

- <span class="pre">`D`</span> is the input’s feature dimension

Concretely, for each element of the sequence, this layer computes:

<div class="math notranslate nohighlight">

\\\begin{split}\begin{aligned} r_t &= \sigma (W\_{xr}x_t + W\_{hr}h_t +
b\_{r}) \\ z_t &= \sigma (W\_{xz}x_t + W\_{hz}h_t + b\_{z}) \\ n_t &=
\text{tanh}(W\_{xn}x_t + b\_{n} + r_t \odot (W\_{hn}h_t + b\_{hn})) \\
h\_{t + 1} &= (1 - z_t) \odot n_t + z_t \odot h_t
\end{aligned}\end{split}\\

</div>

The hidden state <span class="math notranslate nohighlight">\\h\\</span>
has shape <span class="pre">`NH`</span> or <span class="pre">`H`</span>
depending on whether the input is batched or not. Returns the hidden
state at each time step of shape <span class="pre">`NLH`</span> or
<span class="pre">`LH`</span>.

Parameters<span class="colon">:</span>  
- **input_size**
  (<a href="https://docs.python.org/3/library/functions.html#int"
  class="reference external" title="(in Python v3.13)"><em>int</em></a>)
  – Dimension of the input, <span class="pre">`D`</span>.

- **hidden_size**
  (<a href="https://docs.python.org/3/library/functions.html#int"
  class="reference external" title="(in Python v3.13)"><em>int</em></a>)
  – Dimension of the hidden state, <span class="pre">`H`</span>.

- **bias**
  (<a href="https://docs.python.org/3/library/functions.html#bool"
  class="reference external" title="(in Python v3.13)"><em>bool</em></a>)
  – Whether to use biases or not. Default:
  <span class="pre">`True`</span>.

Methods

<div class="pst-scrollable-table-container">

</div>

</div>

<div class="prev-next-area">

<a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.GroupNorm.html"
class="left-prev" title="previous page"><em></em></a>

<div class="prev-next-info">

previous

mlx.nn.GroupNorm

</div>

<a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.HardShrink.html"
class="right-next" title="next page"></a>

<div class="prev-next-info">

next

mlx.nn.HardShrink

</div>

</div>

</div>

<div class="bd-sidebar-secondary bd-toc">

<div class="sidebar-secondary-items sidebar-secondary__inner">

<div class="sidebar-secondary-item">

<div class="page-toc tocsection onthispage">

Contents

</div>

- <a href="https://ml-explore.github.io/mlx/build/html/#mlx.nn.GRU"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">GRU</code></span></a>

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
