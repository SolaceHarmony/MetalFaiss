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
  href="https://ml-explore.github.io/mlx/build/html/_sources/python/optimizers/_autosummary/mlx.optimizers.Adam.rst"
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

# mlx.optimizers.Adam

<div id="print-main-content">

<div id="jb-print-toc">

<div>

## Contents

</div>

- <a
  href="https://ml-explore.github.io/mlx/build/html/#mlx.optimizers.Adam"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">Adam</code></span></a>

</div>

</div>

</div>

<div id="searchbox">

</div>

<div id="mlx-optimizers-adam" class="section">

# mlx.optimizers.Adam<a
href="https://ml-explore.github.io/mlx/build/html/#mlx-optimizers-adam"
class="headerlink" title="Link to this heading">#</a>

*<span class="pre">class</span><span class="w"> </span>*<span class="sig-name descname"><span class="pre">Adam</span></span><span class="sig-paren">(</span>*<span class="n"><span class="pre">learning_rate</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><a href="https://docs.python.org/3/library/functions.html#float"
class="reference external" title="(in Python v3.13)"><span
class="pre">float</span></a><span class="w"> </span><span class="p"><span class="pre">\|</span></span><span class="w"> </span><a href="https://docs.python.org/3/library/typing.html#typing.Callable"
class="reference external" title="(in Python v3.13)"><span
class="pre">Callable</span></a><span class="p"><span class="pre">\[</span></span><span class="p"><span class="pre">\[</span></span><a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.html#mlx.core.array"
class="reference internal" title="mlx.core.array"><span
class="pre">array</span></a><span class="p"><span class="pre">\]</span></span><span class="p"><span class="pre">,</span></span><span class="w"> </span><a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.html#mlx.core.array"
class="reference internal" title="mlx.core.array"><span
class="pre">array</span></a><span class="p"><span class="pre">\]</span></span></span>*, *<span class="n"><span class="pre">betas</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><a href="https://docs.python.org/3/library/typing.html#typing.List"
class="reference external" title="(in Python v3.13)"><span
class="pre">List</span></a><span class="p"><span class="pre">\[</span></span><a href="https://docs.python.org/3/library/functions.html#float"
class="reference external" title="(in Python v3.13)"><span
class="pre">float</span></a><span class="p"><span class="pre">\]</span></span></span><span class="w"> </span><span class="o"><span class="pre">=</span></span><span class="w"> </span><span class="default_value"><span class="pre">\[0.9,</span> <span class="pre">0.999\]</span></span>*, *<span class="n"><span class="pre">eps</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><a href="https://docs.python.org/3/library/functions.html#float"
class="reference external" title="(in Python v3.13)"><span
class="pre">float</span></a></span><span class="w"> </span><span class="o"><span class="pre">=</span></span><span class="w"> </span><span class="default_value"><span class="pre">1e-08</span></span>*, *<span class="n"><span class="pre">bias_correction</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><a href="https://docs.python.org/3/library/functions.html#bool"
class="reference external" title="(in Python v3.13)"><span
class="pre">bool</span></a></span><span class="w"> </span><span class="o"><span class="pre">=</span></span><span class="w"> </span><span class="default_value"><span class="pre">False</span></span>*<span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#mlx.optimizers.Adam"
class="headerlink" title="Link to this definition">#</a>  
The Adam optimizer \[1\]. In detail,

\[1\]: Kingma, D.P. and Ba, J., 2015. Adam: A method for stochastic
optimization. ICLR 2015.

<div class="math notranslate nohighlight">

\\\begin{split}m\_{t+1} &= \beta_1 m_t + (1 - \beta_1) g_t \\ v\_{t+1}
&= \beta_2 v_t + (1 - \beta_2) g_t^2 \\ w\_{t+1} &= w_t - \lambda
\frac{m\_{t+1}}{\sqrt{v\_{t+1} + \epsilon}}\end{split}\\

</div>

Parameters<span class="colon">:</span>  
- **learning_rate**
  (<a href="https://docs.python.org/3/library/functions.html#float"
  class="reference external" title="(in Python v3.13)"><em>float</em></a>
  *or* *callable*) – The learning rate
  <span class="math notranslate nohighlight">\\\lambda\\</span>.

- **betas**
  (*Tuple\[*<a href="https://docs.python.org/3/library/functions.html#float"
  class="reference external" title="(in Python v3.13)"><em>float</em></a>*,*
  <a href="https://docs.python.org/3/library/functions.html#float"
  class="reference external" title="(in Python v3.13)"><em>float</em></a>*\],*
  *optional*) – The coefficients
  <span class="math notranslate nohighlight">\\(\beta_1,
  \beta_2)\\</span> used for computing running averages of the gradient
  and its square. Default:
  <span class="pre">`(0.9,`</span>` `<span class="pre">`0.999)`</span>

- **eps**
  (<a href="https://docs.python.org/3/library/functions.html#float"
  class="reference external" title="(in Python v3.13)"><em>float</em></a>*,*
  *optional*) – The term
  <span class="math notranslate nohighlight">\\\epsilon\\</span> added
  to the denominator to improve numerical stability. Default:
  <span class="pre">`1e-8`</span>

- **bias_correction**
  (<a href="https://docs.python.org/3/library/functions.html#bool"
  class="reference external" title="(in Python v3.13)"><em>bool</em></a>*,*
  *optional*) – If set to <span class="pre">`True`</span>, bias
  correction is applied. Default: <span class="pre">`False`</span>

Methods

<div class="pst-scrollable-table-container">

|  |  |
|----|----|
| <span class="pre">`__init__`</span>(learning_rate\[, betas, eps, ...\]) |  |
| <span class="pre">`apply_single`</span>(gradient, parameter, state) | Performs the Adam parameter update and stores <span class="math notranslate nohighlight">\\v\\</span> and <span class="math notranslate nohighlight">\\m\\</span> in the optimizer state. |
| <span class="pre">`init_single`</span>(parameter, state) | Initialize optimizer state |

</div>

</div>

<div class="prev-next-area">

<a
href="https://ml-explore.github.io/mlx/build/html/python/optimizers/_autosummary/mlx.optimizers.AdaDelta.html"
class="left-prev" title="previous page"><em></em></a>

<div class="prev-next-info">

previous

mlx.optimizers.AdaDelta

</div>

<a
href="https://ml-explore.github.io/mlx/build/html/python/optimizers/_autosummary/mlx.optimizers.AdamW.html"
class="right-next" title="next page"></a>

<div class="prev-next-info">

next

mlx.optimizers.AdamW

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
  href="https://ml-explore.github.io/mlx/build/html/#mlx.optimizers.Adam"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">Adam</code></span></a>

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
