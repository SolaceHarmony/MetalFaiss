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
  href="https://ml-explore.github.io/mlx/build/html/_sources/python/_autosummary/mlx.optimizers.clip_grad_norm.rst"
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

# mlx.optimizers.clip_grad_norm

<div id="print-main-content">

<div id="jb-print-toc">

<div>

## Contents

</div>

- <a
  href="https://ml-explore.github.io/mlx/build/html/#mlx.optimizers.clip_grad_norm"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">clip_grad_norm()</code></span></a>

</div>

</div>

</div>

<div id="searchbox">

</div>

<div id="mlx-optimizers-clip-grad-norm" class="section">

# mlx.optimizers.clip_grad_norm<a
href="https://ml-explore.github.io/mlx/build/html/#mlx-optimizers-clip-grad-norm"
class="headerlink" title="Link to this heading">#</a>

<span class="sig-name descname"><span class="pre">clip_grad_norm</span></span><span class="sig-paren">(</span>*<span class="n"><span class="pre">grads</span></span>*, *<span class="n"><span class="pre">max_norm</span></span>*<span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#mlx.optimizers.clip_grad_norm"
class="headerlink" title="Link to this definition">#</a>  
Clips the global norm of the gradients.

This function ensures that the global norm of the gradients does not
exceed <span class="pre">`max_norm`</span>. It scales down the gradients
proportionally if their norm is greater than
<span class="pre">`max_norm`</span>.

Example

<div class="doctest highlight-default notranslate">

<div class="highlight">

    >>> grads = {"w1": mx.array([2, 3]), "w2": mx.array([1])}
    >>> clipped_grads, total_norm = clip_grad_norm(grads, max_norm=2.0)
    >>> print(clipped_grads)
    {"w1": mx.array([...]), "w2": mx.array([...])}

</div>

</div>

Parameters<span class="colon">:</span>  
- **grads**
  (<a href="https://docs.python.org/3/library/stdtypes.html#dict"
  class="reference external" title="(in Python v3.13)"><em>dict</em></a>)
  – A dictionary containing the gradient arrays.

- **max_norm**
  (<a href="https://docs.python.org/3/library/functions.html#float"
  class="reference external" title="(in Python v3.13)"><em>float</em></a>)
  – The maximum allowed global norm of the gradients.

Returns<span class="colon">:</span>  
The possibly rescaled gradients and the original gradient norm.

Return type<span class="colon">:</span>  
(<a href="https://docs.python.org/3/library/stdtypes.html#dict"
class="reference external" title="(in Python v3.13)"><em>dict</em></a>,
<a href="https://docs.python.org/3/library/functions.html#float"
class="reference external" title="(in Python v3.13)"><em>float</em></a>)

</div>

<div class="prev-next-area">

<a
href="https://ml-explore.github.io/mlx/build/html/python/optimizers/_autosummary/mlx.optimizers.step_decay.html"
class="left-prev" title="previous page"><em></em></a>

<div class="prev-next-info">

previous

mlx.optimizers.step_decay

</div>

<a
href="https://ml-explore.github.io/mlx/build/html/python/distributed.html"
class="right-next" title="next page"></a>

<div class="prev-next-info">

next

Distributed Communication

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
  href="https://ml-explore.github.io/mlx/build/html/#mlx.optimizers.clip_grad_norm"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">clip_grad_norm()</code></span></a>

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
