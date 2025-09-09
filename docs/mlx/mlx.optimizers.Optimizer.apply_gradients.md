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
  href="https://ml-explore.github.io/mlx/build/html/_sources/python/optimizers/_autosummary/mlx.optimizers.Optimizer.apply_gradients.rst"
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

# mlx.optimizers.Optimizer.apply_gradients

<div id="print-main-content">

<div id="jb-print-toc">

<div>

## Contents

</div>

- <a
  href="https://ml-explore.github.io/mlx/build/html/#mlx.optimizers.Optimizer.apply_gradients"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">Optimizer.apply_gradients()</code></span></a>

</div>

</div>

</div>

<div id="searchbox">

</div>

<div id="mlx-optimizers-optimizer-apply-gradients" class="section">

# mlx.optimizers.Optimizer.apply_gradients<a
href="https://ml-explore.github.io/mlx/build/html/#mlx-optimizers-optimizer-apply-gradients"
class="headerlink" title="Link to this heading">#</a>

<span class="sig-prename descclassname"><span class="pre">Optimizer.</span></span><span class="sig-name descname"><span class="pre">apply_gradients</span></span><span class="sig-paren">(</span>*<span class="n"><span class="pre">gradients</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><a href="https://docs.python.org/3/library/stdtypes.html#dict"
class="reference external" title="(in Python v3.13)"><span
class="pre">dict</span></a></span>*, *<span class="n"><span class="pre">parameters</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><a href="https://docs.python.org/3/library/stdtypes.html#dict"
class="reference external" title="(in Python v3.13)"><span
class="pre">dict</span></a></span>*<span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#mlx.optimizers.Optimizer.apply_gradients"
class="headerlink" title="Link to this definition">#</a>  
Apply the gradients to the parameters and return the updated parameters.

Can be used to update a model via
<span class="pre">`model.update(opt.apply_gradients(grads,`</span>` `<span class="pre">`model))`</span>
which is precisely how <a
href="https://ml-explore.github.io/mlx/build/html/python/optimizers/_autosummary/mlx.optimizers.Optimizer.update.html#mlx.optimizers.Optimizer.update"
class="reference internal" title="mlx.optimizers.Optimizer.update"><span
class="pre"><code
class="sourceCode python">Optimizer.update()</code></span></a> is
implemented.

Parameters<span class="colon">:</span>  
- **gradients**
  (<a href="https://docs.python.org/3/library/stdtypes.html#dict"
  class="reference external" title="(in Python v3.13)"><em>dict</em></a>)
  – A Python tree of gradients.

- **parameters**
  (<a href="https://docs.python.org/3/library/stdtypes.html#dict"
  class="reference external" title="(in Python v3.13)"><em>dict</em></a>)
  – A Python tree of parameters. It can be a superset of the gradients.
  In that case the returned python tree will be of the same structure as
  the gradients.

</div>

<div class="prev-next-area">

<a
href="https://ml-explore.github.io/mlx/build/html/python/optimizers/_autosummary/mlx.optimizers.Optimizer.state.html"
class="left-prev" title="previous page"><em></em></a>

<div class="prev-next-info">

previous

mlx.optimizers.Optimizer.state

</div>

<a
href="https://ml-explore.github.io/mlx/build/html/python/optimizers/_autosummary/mlx.optimizers.Optimizer.init.html"
class="right-next" title="next page"></a>

<div class="prev-next-info">

next

mlx.optimizers.Optimizer.init

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
  href="https://ml-explore.github.io/mlx/build/html/#mlx.optimizers.Optimizer.apply_gradients"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">Optimizer.apply_gradients()</code></span></a>

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
