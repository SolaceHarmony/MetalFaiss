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
  href="https://ml-explore.github.io/mlx/build/html/_sources/python/optimizers/_autosummary/mlx.optimizers.MultiOptimizer.rst"
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

# mlx.optimizers.MultiOptimizer

<div id="print-main-content">

<div id="jb-print-toc">

<div>

## Contents

</div>

- <a
  href="https://ml-explore.github.io/mlx/build/html/#mlx.optimizers.MultiOptimizer"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">MultiOptimizer</code></span></a>

</div>

</div>

</div>

<div id="searchbox">

</div>

<div id="mlx-optimizers-multioptimizer" class="section">

# mlx.optimizers.MultiOptimizer<a
href="https://ml-explore.github.io/mlx/build/html/#mlx-optimizers-multioptimizer"
class="headerlink" title="Link to this heading">#</a>

*<span class="pre">class</span><span class="w"> </span>*<span class="sig-name descname"><span class="pre">MultiOptimizer</span></span><span class="sig-paren">(</span>*<span class="n"><span class="pre">optimizers</span></span>*, *<span class="n"><span class="pre">filters</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><a href="https://docs.python.org/3/library/stdtypes.html#list"
class="reference external" title="(in Python v3.13)"><span
class="pre">list</span></a></span><span class="w"> </span><span class="o"><span class="pre">=</span></span><span class="w"> </span><span class="default_value"><span class="pre">\[\]</span></span>*<span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#mlx.optimizers.MultiOptimizer"
class="headerlink" title="Link to this definition">#</a>  
Wraps a list of optimizers with corresponding weight predicates/filters
to make it easy to use different optimizers for different weights.

The predicates take the full “path” of the weight and the weight itself
and return True if it should be considered for this optimizer. The last
optimizer in the list is a fallback optimizer and no predicate should be
given for it.

Parameters<span class="colon">:</span>  
- **optimizers**
  (<a href="https://docs.python.org/3/library/stdtypes.html#list"
  class="reference external" title="(in Python v3.13)"><em>list</em></a>*\[*<a
  href="https://ml-explore.github.io/mlx/build/html/python/optimizers/optimizer.html#mlx.optimizers.Optimizer"
  class="reference internal"
  title="mlx.optimizers.Optimizer"><em>Optimizer</em></a>*\]*) – A list
  of optimizers to delegate to

- **filters**
  (<a href="https://docs.python.org/3/library/stdtypes.html#list"
  class="reference external" title="(in Python v3.13)"><em>list</em></a>*\[Callable\[\[*<a href="https://docs.python.org/3/library/stdtypes.html#str"
  class="reference external" title="(in Python v3.13)"><em>str</em></a>*,*
  <a
  href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.html#mlx.core.array"
  class="reference internal" title="mlx.core.array"><em>array</em></a>*\],*
  <a href="https://docs.python.org/3/library/functions.html#bool"
  class="reference external" title="(in Python v3.13)"><em>bool</em></a>*\]*)
  – A list of predicates that should be one less than the provided
  optimizers.

Methods

<div class="pst-scrollable-table-container">

|  |  |
|----|----|
| <span class="pre">`__init__`</span>(optimizers\[, filters\]) |  |
| <span class="pre">`apply_gradients`</span>(gradients, parameters) | Apply the gradients to the parameters and return the updated parameters. |
| <span class="pre">`init`</span>(parameters) | Initialize the optimizer's state |

</div>

</div>

<div class="prev-next-area">

<a
href="https://ml-explore.github.io/mlx/build/html/python/optimizers/_autosummary/mlx.optimizers.Lion.html"
class="left-prev" title="previous page"><em></em></a>

<div class="prev-next-info">

previous

mlx.optimizers.Lion

</div>

<a
href="https://ml-explore.github.io/mlx/build/html/python/optimizers/schedulers.html"
class="right-next" title="next page"></a>

<div class="prev-next-info">

next

Schedulers

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
  href="https://ml-explore.github.io/mlx/build/html/#mlx.optimizers.MultiOptimizer"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">MultiOptimizer</code></span></a>

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
