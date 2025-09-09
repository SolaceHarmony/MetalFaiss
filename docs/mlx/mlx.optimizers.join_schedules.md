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
  href="https://ml-explore.github.io/mlx/build/html/_sources/python/optimizers/_autosummary/mlx.optimizers.join_schedules.rst"
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

# mlx.optimizers.join_schedules

<div id="print-main-content">

<div id="jb-print-toc">

<div>

## Contents

</div>

- <a
  href="https://ml-explore.github.io/mlx/build/html/#mlx.optimizers.join_schedules"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">join_schedules()</code></span></a>

</div>

</div>

</div>

<div id="searchbox">

</div>

<div id="mlx-optimizers-join-schedules" class="section">

# mlx.optimizers.join_schedules<a
href="https://ml-explore.github.io/mlx/build/html/#mlx-optimizers-join-schedules"
class="headerlink" title="Link to this heading">#</a>

<span class="sig-name descname"><span class="pre">join_schedules</span></span><span class="sig-paren">(</span>*<span class="n"><span class="pre">schedules</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><a href="https://docs.python.org/3/library/typing.html#typing.List"
class="reference external" title="(in Python v3.13)"><span
class="pre">List</span></a><span class="p"><span class="pre">\[</span></span><a href="https://docs.python.org/3/library/typing.html#typing.Callable"
class="reference external" title="(in Python v3.13)"><span
class="pre">Callable</span></a><span class="p"><span class="pre">\]</span></span></span>*, *<span class="n"><span class="pre">boundaries</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><a href="https://docs.python.org/3/library/typing.html#typing.List"
class="reference external" title="(in Python v3.13)"><span
class="pre">List</span></a><span class="p"><span class="pre">\[</span></span><a href="https://docs.python.org/3/library/functions.html#int"
class="reference external" title="(in Python v3.13)"><span
class="pre">int</span></a><span class="p"><span class="pre">\]</span></span></span>*<span class="sig-paren">)</span> <span class="sig-return"><span class="sig-return-icon">→</span> <span class="sig-return-typehint"><a href="https://docs.python.org/3/library/typing.html#typing.Callable"
class="reference external" title="(in Python v3.13)"><span
class="pre">Callable</span></a></span></span><a
href="https://ml-explore.github.io/mlx/build/html/#mlx.optimizers.join_schedules"
class="headerlink" title="Link to this definition">#</a>  
Join multiple schedules to create a new schedule.

Parameters<span class="colon">:</span>  
- **schedules**
  (<a href="https://docs.python.org/3/library/stdtypes.html#list"
  class="reference external" title="(in Python v3.13)"><em>list</em></a>*(Callable)*)
  – A list of schedules. Schedule
  <span class="math notranslate nohighlight">\\i+1\\</span> receives a
  step count indicating the number of steps since the
  <span class="math notranslate nohighlight">\\i\\</span>-th boundary.

- **boundaries**
  (<a href="https://docs.python.org/3/library/stdtypes.html#list"
  class="reference external" title="(in Python v3.13)"><em>list</em></a>*(*<a href="https://docs.python.org/3/library/functions.html#int"
  class="reference external" title="(in Python v3.13)"><em>int</em></a>*)*)
  – A list of integers of length
  <span class="pre">`len(schedules)`</span>` `<span class="pre">`-`</span>` `<span class="pre">`1`</span>
  that indicates when to transition between schedules.

Example

<div class="doctest highlight-default notranslate">

<div class="highlight">

    >>> linear = optim.linear_schedule(0, 1e-1, steps=10)
    >>> cosine = optim.cosine_decay(1e-1, 200)
    >>> lr_schedule = optim.join_schedules([linear, cosine], [10])
    >>> optimizer = optim.Adam(learning_rate=lr_schedule)
    >>> optimizer.learning_rate
    array(0.0, dtype=float32)
    >>> for _ in range(12): optimizer.update({}, {})
    ...
    >>> optimizer.learning_rate
    array(0.0999938, dtype=float32)

</div>

</div>

</div>

<div class="prev-next-area">

<a
href="https://ml-explore.github.io/mlx/build/html/python/optimizers/_autosummary/mlx.optimizers.exponential_decay.html"
class="left-prev" title="previous page"><em></em></a>

<div class="prev-next-info">

previous

mlx.optimizers.exponential_decay

</div>

<a
href="https://ml-explore.github.io/mlx/build/html/python/optimizers/_autosummary/mlx.optimizers.linear_schedule.html"
class="right-next" title="next page"></a>

<div class="prev-next-info">

next

mlx.optimizers.linear_schedule

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
  href="https://ml-explore.github.io/mlx/build/html/#mlx.optimizers.join_schedules"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">join_schedules()</code></span></a>

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
