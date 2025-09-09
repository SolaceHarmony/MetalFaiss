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
  href="https://ml-explore.github.io/mlx/build/html/_sources/python/_autosummary/mlx.nn.value_and_grad.rst"
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

# mlx.nn.value_and_grad

<div id="print-main-content">

<div id="jb-print-toc">

<div>

## Contents

</div>

- <a
  href="https://ml-explore.github.io/mlx/build/html/#mlx.nn.value_and_grad"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">value_and_grad()</code></span></a>

</div>

</div>

</div>

<div id="searchbox">

</div>

<div id="mlx-nn-value-and-grad" class="section">

# mlx.nn.value_and_grad<a
href="https://ml-explore.github.io/mlx/build/html/#mlx-nn-value-and-grad"
class="headerlink" title="Link to this heading">#</a>

<span class="sig-name descname"><span class="pre">value_and_grad</span></span><span class="sig-paren">(</span>*<span class="n"><span class="pre">model</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><a
href="https://ml-explore.github.io/mlx/build/html/python/nn/module.html#mlx.nn.Module"
class="reference internal" title="mlx.nn.layers.base.Module"><span
class="pre">Module</span></a></span>*, *<span class="n"><span class="pre">fn</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><a href="https://docs.python.org/3/library/typing.html#typing.Callable"
class="reference external" title="(in Python v3.13)"><span
class="pre">Callable</span></a></span>*<span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#mlx.nn.value_and_grad"
class="headerlink" title="Link to this definition">#</a>  
Transform the passed function <span class="pre">`fn`</span> to a
function that computes the gradients of <span class="pre">`fn`</span>
wrt the model’s trainable parameters and also its value.

Parameters<span class="colon">:</span>  
- **model** (<a
  href="https://ml-explore.github.io/mlx/build/html/python/nn/module.html#mlx.nn.Module"
  class="reference internal" title="mlx.nn.Module"><em>Module</em></a>)
  – The model whose trainable parameters to compute gradients for

- **fn** (*Callable*) – The scalar function to compute gradients for

Returns<span class="colon">:</span>  
A callable that returns the value of <span class="pre">`fn`</span> and
the gradients wrt the trainable parameters of
<span class="pre">`model`</span>

</div>

<div class="prev-next-area">

<a href="https://ml-explore.github.io/mlx/build/html/python/nn.html"
class="left-prev" title="previous page"><em></em></a>

<div class="prev-next-info">

previous

Neural Networks

</div>

<a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.nn.quantize.html"
class="right-next" title="next page"></a>

<div class="prev-next-info">

next

mlx.nn.quantize

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
  href="https://ml-explore.github.io/mlx/build/html/#mlx.nn.value_and_grad"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">value_and_grad()</code></span></a>

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
