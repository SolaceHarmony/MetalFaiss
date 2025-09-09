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
  href="https://ml-explore.github.io/mlx/build/html/_sources/python/optimizers/_autosummary/mlx.optimizers.Adafactor.rst"
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

# mlx.optimizers.Adafactor

<div id="print-main-content">

<div id="jb-print-toc">

<div>

## Contents

</div>

- <a
  href="https://ml-explore.github.io/mlx/build/html/#mlx.optimizers.Adafactor"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">Adafactor</code></span></a>

</div>

</div>

</div>

<div id="searchbox">

</div>

<div id="mlx-optimizers-adafactor" class="section">

# mlx.optimizers.Adafactor<a
href="https://ml-explore.github.io/mlx/build/html/#mlx-optimizers-adafactor"
class="headerlink" title="Link to this heading">#</a>

*<span class="pre">class</span><span class="w"> </span>*<span class="sig-name descname"><span class="pre">Adafactor</span></span><span class="sig-paren">(</span>*<span class="n"><span class="pre">learning_rate</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><a href="https://docs.python.org/3/library/functions.html#float"
class="reference external" title="(in Python v3.13)"><span
class="pre">float</span></a><span class="w"> </span><span class="p"><span class="pre">\|</span></span><span class="w"> </span><a href="https://docs.python.org/3/library/typing.html#typing.Callable"
class="reference external" title="(in Python v3.13)"><span
class="pre">Callable</span></a><span class="p"><span class="pre">\[</span></span><span class="p"><span class="pre">\[</span></span><a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.html#mlx.core.array"
class="reference internal" title="mlx.core.array"><span
class="pre">array</span></a><span class="p"><span class="pre">\]</span></span><span class="p"><span class="pre">,</span></span><span class="w"> </span><a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.html#mlx.core.array"
class="reference internal" title="mlx.core.array"><span
class="pre">array</span></a><span class="p"><span class="pre">\]</span></span><span class="w"> </span><span class="p"><span class="pre">\|</span></span><span class="w"> </span><a href="https://docs.python.org/3/library/constants.html#None"
class="reference external" title="(in Python v3.13)"><span
class="pre">None</span></a></span><span class="w"> </span><span class="o"><span class="pre">=</span></span><span class="w"> </span><span class="default_value"><span class="pre">None</span></span>*, *<span class="n"><span class="pre">eps</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><a href="https://docs.python.org/3/library/typing.html#typing.Tuple"
class="reference external" title="(in Python v3.13)"><span
class="pre">Tuple</span></a><span class="p"><span class="pre">\[</span></span><a href="https://docs.python.org/3/library/functions.html#float"
class="reference external" title="(in Python v3.13)"><span
class="pre">float</span></a><span class="p"><span class="pre">,</span></span><span class="w"> </span><a href="https://docs.python.org/3/library/functions.html#float"
class="reference external" title="(in Python v3.13)"><span
class="pre">float</span></a><span class="p"><span class="pre">\]</span></span></span><span class="w"> </span><span class="o"><span class="pre">=</span></span><span class="w"> </span><span class="default_value"><span class="pre">(1e-30,</span> <span class="pre">0.001)</span></span>*, *<span class="n"><span class="pre">clip_threshold</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><a href="https://docs.python.org/3/library/functions.html#float"
class="reference external" title="(in Python v3.13)"><span
class="pre">float</span></a></span><span class="w"> </span><span class="o"><span class="pre">=</span></span><span class="w"> </span><span class="default_value"><span class="pre">1.0</span></span>*, *<span class="n"><span class="pre">decay_rate</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><a href="https://docs.python.org/3/library/functions.html#float"
class="reference external" title="(in Python v3.13)"><span
class="pre">float</span></a></span><span class="w"> </span><span class="o"><span class="pre">=</span></span><span class="w"> </span><span class="default_value"><span class="pre">-0.8</span></span>*, *<span class="n"><span class="pre">beta_1</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><a href="https://docs.python.org/3/library/functions.html#float"
class="reference external" title="(in Python v3.13)"><span
class="pre">float</span></a><span class="w"> </span><span class="p"><span class="pre">\|</span></span><span class="w"> </span><a href="https://docs.python.org/3/library/constants.html#None"
class="reference external" title="(in Python v3.13)"><span
class="pre">None</span></a></span><span class="w"> </span><span class="o"><span class="pre">=</span></span><span class="w"> </span><span class="default_value"><span class="pre">None</span></span>*, *<span class="n"><span class="pre">weight_decay</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><a href="https://docs.python.org/3/library/functions.html#float"
class="reference external" title="(in Python v3.13)"><span
class="pre">float</span></a></span><span class="w"> </span><span class="o"><span class="pre">=</span></span><span class="w"> </span><span class="default_value"><span class="pre">0.0</span></span>*, *<span class="n"><span class="pre">scale_parameter</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><a href="https://docs.python.org/3/library/functions.html#bool"
class="reference external" title="(in Python v3.13)"><span
class="pre">bool</span></a></span><span class="w"> </span><span class="o"><span class="pre">=</span></span><span class="w"> </span><span class="default_value"><span class="pre">True</span></span>*, *<span class="n"><span class="pre">relative_step</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><a href="https://docs.python.org/3/library/functions.html#bool"
class="reference external" title="(in Python v3.13)"><span
class="pre">bool</span></a></span><span class="w"> </span><span class="o"><span class="pre">=</span></span><span class="w"> </span><span class="default_value"><span class="pre">True</span></span>*, *<span class="n"><span class="pre">warmup_init</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><a href="https://docs.python.org/3/library/functions.html#bool"
class="reference external" title="(in Python v3.13)"><span
class="pre">bool</span></a></span><span class="w"> </span><span class="o"><span class="pre">=</span></span><span class="w"> </span><span class="default_value"><span class="pre">False</span></span>*<span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#mlx.optimizers.Adafactor"
class="headerlink" title="Link to this definition">#</a>  
The Adafactor optimizer.

Our Adafactor implementation follows the original paper:
<a href="https://arxiv.org/abs/1804.04235"
class="reference external">Adafactor: Adaptive Learning Rates with
Sublinear Memory Cost</a>

Parameters<span class="colon">:</span>  
- **learning_rate**
  (<a href="https://docs.python.org/3/library/functions.html#float"
  class="reference external" title="(in Python v3.13)"><em>float</em></a>
  *or* *callable,* *optional*) – The learning rate. Default:
  <span class="pre">`None`</span>.

- **eps**
  (<a href="https://docs.python.org/3/library/stdtypes.html#tuple"
  class="reference external" title="(in Python v3.13)"><em>tuple</em></a>*(*<a href="https://docs.python.org/3/library/functions.html#float"
  class="reference external" title="(in Python v3.13)"><em>float</em></a>*,*
  <a href="https://docs.python.org/3/library/functions.html#float"
  class="reference external" title="(in Python v3.13)"><em>float</em></a>*),*
  *optional*) – The first term
  <span class="math notranslate nohighlight">\\\epsilon_1\\</span> added
  to the square of the gradients to improve numerical stability and the
  second term
  <span class="math notranslate nohighlight">\\\epsilon_2\\</span> is
  used for parameter scaling if
  <span class="pre">`parameter_scale`</span> is set to
  <span class="pre">`True`</span>. Default:
  <span class="pre">`(1e-30,`</span>` `<span class="pre">`1e-3)`</span>.

- **clip_threshold**
  (<a href="https://docs.python.org/3/library/functions.html#float"
  class="reference external" title="(in Python v3.13)"><em>float</em></a>*,*
  *optional*) – Clips the unscaled update at
  <span class="pre">`clip_threshold`</span>. Default:
  <span class="pre">`1.0`</span>.

- **decay_rate**
  (<a href="https://docs.python.org/3/library/functions.html#float"
  class="reference external" title="(in Python v3.13)"><em>float</em></a>*,*
  *optional*) – Coefficient for the running average of the squared
  gradient. Default: <span class="pre">`-0.8`</span>.

- **beta_1**
  (<a href="https://docs.python.org/3/library/functions.html#float"
  class="reference external" title="(in Python v3.13)"><em>float</em></a>*,*
  *optional*) – If set to a value bigger than zero then first moment
  will be used. Default: <span class="pre">`None`</span>.

- **weight_decay**
  (<a href="https://docs.python.org/3/library/functions.html#float"
  class="reference external" title="(in Python v3.13)"><em>float</em></a>*,*
  *optional*) – The weight decay
  <span class="math notranslate nohighlight">\\\lambda\\</span>.
  Default: <span class="pre">`0.0`</span>.

- **scale_parameter**
  (<a href="https://docs.python.org/3/library/functions.html#bool"
  class="reference external" title="(in Python v3.13)"><em>bool</em></a>*,*
  *optional*) – If set to <span class="pre">`True`</span> the learning
  rate will be scaled by
  <span class="math notranslate nohighlight">\\\max(\epsilon_1,
  \text{RMS}(w\_{t-1}))\\</span>. Default:
  <span class="pre">`True`</span>.

- **relative_step**
  (<a href="https://docs.python.org/3/library/functions.html#bool"
  class="reference external" title="(in Python v3.13)"><em>bool</em></a>*,*
  *optional*) – If set to <span class="pre">`True`</span> the
  <span class="pre">`learning_rate`</span> will be ignored and relative
  step size will be computed. Default: <span class="pre">`True`</span>.

- **warmup_init**
  (<a href="https://docs.python.org/3/library/functions.html#bool"
  class="reference external" title="(in Python v3.13)"><em>bool</em></a>*,*
  *optional*) – If set to <span class="pre">`True`</span> then the
  relative step size will be calculated by the current step. Default:
  <span class="pre">`False`</span>.

Methods

<div class="pst-scrollable-table-container">

|  |  |
|----|----|
| <span class="pre">`__init__`</span>(\[learning_rate, eps, ...\]) |  |
| <span class="pre">`apply_single`</span>(gradient, parameter, state) | Performs the Adafactor parameter and state update. |
| <span class="pre">`init_single`</span>(parameter, state) | Initialize optimizer state |

</div>

</div>

<div class="prev-next-area">

<a
href="https://ml-explore.github.io/mlx/build/html/python/optimizers/_autosummary/mlx.optimizers.Adagrad.html"
class="left-prev" title="previous page"><em></em></a>

<div class="prev-next-info">

previous

mlx.optimizers.Adagrad

</div>

<a
href="https://ml-explore.github.io/mlx/build/html/python/optimizers/_autosummary/mlx.optimizers.AdaDelta.html"
class="right-next" title="next page"></a>

<div class="prev-next-info">

next

mlx.optimizers.AdaDelta

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
  href="https://ml-explore.github.io/mlx/build/html/#mlx.optimizers.Adafactor"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">Adafactor</code></span></a>

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
