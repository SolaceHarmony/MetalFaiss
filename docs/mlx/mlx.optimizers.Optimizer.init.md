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
  href="https://ml-explore.github.io/mlx/build/html/_sources/python/optimizers/_autosummary/mlx.optimizers.Optimizer.init.rst"
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

# mlx.optimizers.Optimizer.init

<div id="print-main-content">

<div id="jb-print-toc">

<div>

## Contents

</div>

- <a
  href="https://ml-explore.github.io/mlx/build/html/#mlx.optimizers.Optimizer.init"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">Optimizer.init()</code></span></a>

</div>

</div>

</div>

<div id="searchbox">

</div>

<div id="mlx-optimizers-optimizer-init" class="section">

# mlx.optimizers.Optimizer.init<a
href="https://ml-explore.github.io/mlx/build/html/#mlx-optimizers-optimizer-init"
class="headerlink" title="Link to this heading">#</a>

<span class="sig-prename descclassname"><span class="pre">Optimizer.</span></span><span class="sig-name descname"><span class="pre">init</span></span><span class="sig-paren">(</span>*<span class="n"><span class="pre">parameters</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><a href="https://docs.python.org/3/library/stdtypes.html#dict"
class="reference external" title="(in Python v3.13)"><span
class="pre">dict</span></a></span>*<span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#mlx.optimizers.Optimizer.init"
class="headerlink" title="Link to this definition">#</a>  
Initialize the optimizer’s state

This function can be used to initialize optimizers which have state
(like momentum in <a
href="https://ml-explore.github.io/mlx/build/html/python/optimizers/_autosummary/mlx.optimizers.SGD.html#mlx.optimizers.SGD"
class="reference internal" title="mlx.optimizers.SGD"><span
class="pre"><code class="sourceCode python">SGD</code></span></a>).
Using this method is optional as the optimizer will initialize itself if
the state is not yet set. However, there are some cases where explicit
initialization is useful in order to have access to the <a
href="https://ml-explore.github.io/mlx/build/html/python/optimizers/_autosummary/mlx.optimizers.Optimizer.state.html#mlx.optimizers.Optimizer.state"
class="reference internal" title="mlx.optimizers.Optimizer.state"><span
class="pre"><code
class="sourceCode python">Optimizer.state</code></span></a> before the
first call to <a
href="https://ml-explore.github.io/mlx/build/html/python/optimizers/_autosummary/mlx.optimizers.Optimizer.update.html#mlx.optimizers.Optimizer.update"
class="reference internal" title="mlx.optimizers.Optimizer.update"><span
class="pre"><code
class="sourceCode python">Optimizer.update()</code></span></a>.

Parameters<span class="colon">:</span>  
**model**
(<a href="https://docs.python.org/3/library/stdtypes.html#dict"
class="reference external" title="(in Python v3.13)"><em>dict</em></a>)
– A Python tree of parameters.

Example

<div class="doctest highlight-default notranslate">

<div class="highlight">

    >>> optimizer = optim.SGD(learning_rate=1e-1, momentum=0.9)
    >>> model = nn.Linear(2, 2)
    >>> optimizer.init(model.trainable_parameters())
    >>> optimizer.state.keys()
    dict_keys(['step', 'learning_rate', 'weight', 'bias'])

</div>

</div>

</div>

<div class="prev-next-area">

<a
href="https://ml-explore.github.io/mlx/build/html/python/optimizers/_autosummary/mlx.optimizers.Optimizer.apply_gradients.html"
class="left-prev" title="previous page"><em></em></a>

<div class="prev-next-info">

previous

mlx.optimizers.Optimizer.apply_gradients

</div>

<a
href="https://ml-explore.github.io/mlx/build/html/python/optimizers/_autosummary/mlx.optimizers.Optimizer.update.html"
class="right-next" title="next page"></a>

<div class="prev-next-info">

next

mlx.optimizers.Optimizer.update

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
  href="https://ml-explore.github.io/mlx/build/html/#mlx.optimizers.Optimizer.init"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">Optimizer.init()</code></span></a>

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
