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
  href="https://ml-explore.github.io/mlx/build/html/_sources/python/optimizers/_autosummary/mlx.optimizers.Optimizer.update.rst"
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

# mlx.optimizers.Optimizer.update

<div id="print-main-content">

<div id="jb-print-toc">

<div>

## Contents

</div>

- <a
  href="https://ml-explore.github.io/mlx/build/html/#mlx.optimizers.Optimizer.update"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">Optimizer.update()</code></span></a>

</div>

</div>

</div>

<div id="searchbox">

</div>

<div id="mlx-optimizers-optimizer-update" class="section">

# mlx.optimizers.Optimizer.update<a
href="https://ml-explore.github.io/mlx/build/html/#mlx-optimizers-optimizer-update"
class="headerlink" title="Link to this heading">#</a>

<span class="sig-prename descclassname"><span class="pre">Optimizer.</span></span><span class="sig-name descname"><span class="pre">update</span></span><span class="sig-paren">(</span>*<span class="n"><span class="pre">model</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><a
href="https://ml-explore.github.io/mlx/build/html/python/nn/module.html#mlx.nn.Module"
class="reference internal" title="mlx.nn.layers.base.Module"><span
class="pre">Module</span></a></span>*, *<span class="n"><span class="pre">gradients</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><a href="https://docs.python.org/3/library/stdtypes.html#dict"
class="reference external" title="(in Python v3.13)"><span
class="pre">dict</span></a></span>*<span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#mlx.optimizers.Optimizer.update"
class="headerlink" title="Link to this definition">#</a>  
Apply the gradients to the parameters of the model and update the model
with the new parameters.

Parameters<span class="colon">:</span>  
- **model** (<a
  href="https://ml-explore.github.io/mlx/build/html/python/nn/module.html#mlx.nn.Module"
  class="reference internal" title="mlx.nn.Module"><em>Module</em></a>)
  – An mlx module to be updated.

- **gradients**
  (<a href="https://docs.python.org/3/library/stdtypes.html#dict"
  class="reference external" title="(in Python v3.13)"><em>dict</em></a>)
  – A Python tree of gradients, most likely computed via <a
  href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.nn.value_and_grad.html#mlx.nn.value_and_grad"
  class="reference internal" title="mlx.nn.value_and_grad"><span
  class="pre"><code
  class="sourceCode python">mlx.nn.value_and_grad()</code></span></a>.

</div>

<div class="prev-next-area">

<a
href="https://ml-explore.github.io/mlx/build/html/python/optimizers/_autosummary/mlx.optimizers.Optimizer.init.html"
class="left-prev" title="previous page"><em></em></a>

<div class="prev-next-info">

previous

mlx.optimizers.Optimizer.init

</div>

<a
href="https://ml-explore.github.io/mlx/build/html/python/optimizers/common_optimizers.html"
class="right-next" title="next page"></a>

<div class="prev-next-info">

next

Common Optimizers

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
  href="https://ml-explore.github.io/mlx/build/html/#mlx.optimizers.Optimizer.update"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">Optimizer.update()</code></span></a>

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
