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
  href="https://ml-explore.github.io/mlx/build/html/_sources/python/optimizers/_autosummary/mlx.optimizers.AdaDelta.rst"
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

# mlx.optimizers.AdaDelta

<div id="print-main-content">

<div id="jb-print-toc">

<div>

## Contents

</div>

- <a
  href="https://ml-explore.github.io/mlx/build/html/#mlx.optimizers.AdaDelta"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">AdaDelta</code></span></a>

</div>

</div>

</div>

<div id="searchbox">

</div>

<div id="mlx-optimizers-adadelta" class="section">

# mlx.optimizers.AdaDelta<a
href="https://ml-explore.github.io/mlx/build/html/#mlx-optimizers-adadelta"
class="headerlink" title="Link to this heading">#</a>

*<span class="pre">class</span><span class="w"> </span>*<span class="sig-name descname"><span class="pre">AdaDelta</span></span><span class="sig-paren">(</span>*<span class="n"><span class="pre">learning_rate</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><a href="https://docs.python.org/3/library/functions.html#float"
class="reference external" title="(in Python v3.13)"><span
class="pre">float</span></a><span class="w"> </span><span class="p"><span class="pre">\|</span></span><span class="w"> </span><a href="https://docs.python.org/3/library/typing.html#typing.Callable"
class="reference external" title="(in Python v3.13)"><span
class="pre">Callable</span></a><span class="p"><span class="pre">\[</span></span><span class="p"><span class="pre">\[</span></span><a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.html#mlx.core.array"
class="reference internal" title="mlx.core.array"><span
class="pre">array</span></a><span class="p"><span class="pre">\]</span></span><span class="p"><span class="pre">,</span></span><span class="w"> </span><a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.html#mlx.core.array"
class="reference internal" title="mlx.core.array"><span
class="pre">array</span></a><span class="p"><span class="pre">\]</span></span></span>*, *<span class="n"><span class="pre">rho</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><a href="https://docs.python.org/3/library/functions.html#float"
class="reference external" title="(in Python v3.13)"><span
class="pre">float</span></a></span><span class="w"> </span><span class="o"><span class="pre">=</span></span><span class="w"> </span><span class="default_value"><span class="pre">0.9</span></span>*, *<span class="n"><span class="pre">eps</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><a href="https://docs.python.org/3/library/functions.html#float"
class="reference external" title="(in Python v3.13)"><span
class="pre">float</span></a></span><span class="w"> </span><span class="o"><span class="pre">=</span></span><span class="w"> </span><span class="default_value"><span class="pre">1e-06</span></span>*<span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#mlx.optimizers.AdaDelta"
class="headerlink" title="Link to this definition">#</a>  
The AdaDelta optimizer with a learning rate \[1\].

Our AdaDelta implementation follows the original paper. In detail,

\[1\]: Zeiler, M.D., 2012. ADADELTA: an adaptive learning rate method.
arXiv preprint arXiv:1212.5701.

<div class="math notranslate nohighlight">

\\\begin{split}v\_{t+1} &= \rho v_t + (1 - \rho) g_t^2 \\ \Delta
w\_{t+1} &= \frac{\sqrt{u_t + \epsilon}}{\sqrt{v\_{t+1} + \epsilon}} g_t
\\ u\_{t+1} &= \rho u_t + (1 - \rho) \Delta w\_{t+1}^2 \\ w\_{t+1} &=
w_t - \lambda \Delta w\_{t+1}\end{split}\\

</div>

Parameters<span class="colon">:</span>  
- **learning_rate**
  (<a href="https://docs.python.org/3/library/functions.html#float"
  class="reference external" title="(in Python v3.13)"><em>float</em></a>
  *or* *callable*) – The learning rate
  <span class="math notranslate nohighlight">\\\lambda\\</span>.

- **rho**
  (<a href="https://docs.python.org/3/library/functions.html#float"
  class="reference external" title="(in Python v3.13)"><em>float</em></a>*,*
  *optional*) – The coefficient
  <span class="math notranslate nohighlight">\\\rho\\</span> used for
  computing a running average of squared gradients. Default:
  <span class="pre">`0.9`</span>

- **eps**
  (<a href="https://docs.python.org/3/library/functions.html#float"
  class="reference external" title="(in Python v3.13)"><em>float</em></a>*,*
  *optional*) – The term
  <span class="math notranslate nohighlight">\\\epsilon\\</span> added
  to the denominator to improve numerical stability. Default: 1e-8

Methods

<div class="pst-scrollable-table-container">

|  |  |
|----|----|
| <span class="pre">`__init__`</span>(learning_rate\[, rho, eps\]) |  |
| <span class="pre">`apply_single`</span>(gradient, parameter, state) | Performs the AdaDelta parameter update and stores <span class="math notranslate nohighlight">\\v\\</span> and <span class="math notranslate nohighlight">\\u\\</span> in the optimizer state. |
| <span class="pre">`init_single`</span>(parameter, state) | Initialize optimizer state |

</div>

</div>

<div class="prev-next-area">

<a
href="https://ml-explore.github.io/mlx/build/html/python/optimizers/_autosummary/mlx.optimizers.Adafactor.html"
class="left-prev" title="previous page"><em></em></a>

<div class="prev-next-info">

previous

mlx.optimizers.Adafactor

</div>

<a
href="https://ml-explore.github.io/mlx/build/html/python/optimizers/_autosummary/mlx.optimizers.Adam.html"
class="right-next" title="next page"></a>

<div class="prev-next-info">

next

mlx.optimizers.Adam

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
  href="https://ml-explore.github.io/mlx/build/html/#mlx.optimizers.AdaDelta"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">AdaDelta</code></span></a>

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
