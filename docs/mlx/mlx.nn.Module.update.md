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
  href="https://ml-explore.github.io/mlx/build/html/_sources/python/nn/_autosummary/mlx.nn.Module.update.rst"
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

# mlx.nn.Module.update

<div id="print-main-content">

<div id="jb-print-toc">

<div>

## Contents

</div>

- <a
  href="https://ml-explore.github.io/mlx/build/html/#mlx.nn.Module.update"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">Module.update()</code></span></a>

</div>

</div>

</div>

<div id="searchbox">

</div>

<div id="mlx-nn-module-update" class="section">

# mlx.nn.Module.update<a
href="https://ml-explore.github.io/mlx/build/html/#mlx-nn-module-update"
class="headerlink" title="Link to this heading">#</a>

<span class="sig-prename descclassname"><span class="pre">Module.</span></span><span class="sig-name descname"><span class="pre">update</span></span><span class="sig-paren">(</span>*<span class="n"><span class="pre">parameters</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><a href="https://docs.python.org/3/library/stdtypes.html#dict"
class="reference external" title="(in Python v3.13)"><span
class="pre">dict</span></a></span>*<span class="sig-paren">)</span> <span class="sig-return"><span class="sig-return-icon">→</span> <span class="sig-return-typehint"><a
href="https://ml-explore.github.io/mlx/build/html/python/nn/module.html#mlx.nn.Module"
class="reference internal" title="mlx.nn.layers.base.Module"><span
class="pre">Module</span></a></span></span><a
href="https://ml-explore.github.io/mlx/build/html/#mlx.nn.Module.update"
class="headerlink" title="Link to this definition">#</a>  
Replace the parameters of this Module with the provided ones in the dict
of dicts and lists.

Commonly used by the optimizer to change the model to the updated
(optimized) parameters. Also used by the <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.nn.value_and_grad.html#mlx.nn.value_and_grad"
class="reference internal" title="mlx.nn.value_and_grad"><span
class="pre"><code
class="sourceCode python">mlx.nn.value_and_grad()</code></span></a> to
set the tracers in the model in order to compute gradients.

The passed in parameters dictionary need not be a full dictionary
similar to <a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Module.parameters.html#mlx.nn.Module.parameters"
class="reference internal" title="mlx.nn.Module.parameters"><span
class="pre"><code
class="sourceCode python">parameters()</code></span></a>. Only the
provided locations will be updated.

Parameters<span class="colon">:</span>  
**parameters**
(<a href="https://docs.python.org/3/library/stdtypes.html#dict"
class="reference external" title="(in Python v3.13)"><em>dict</em></a>)
– A complete or partial dictionary of the modules parameters.

Returns<span class="colon">:</span>  
The module instance after updating the parameters.

</div>

<div class="prev-next-area">

<a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Module.unfreeze.html"
class="left-prev" title="previous page"><em></em></a>

<div class="prev-next-info">

previous

mlx.nn.Module.unfreeze

</div>

<a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Module.update_modules.html"
class="right-next" title="next page"></a>

<div class="prev-next-info">

next

mlx.nn.Module.update_modules

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
  href="https://ml-explore.github.io/mlx/build/html/#mlx.nn.Module.update"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">Module.update()</code></span></a>

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
