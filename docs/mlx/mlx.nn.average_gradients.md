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
  href="https://ml-explore.github.io/mlx/build/html/_sources/python/_autosummary/mlx.nn.average_gradients.rst"
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

# mlx.nn.average_gradients

<div id="print-main-content">

<div id="jb-print-toc">

<div>

## Contents

</div>

- <a
  href="https://ml-explore.github.io/mlx/build/html/#mlx.nn.average_gradients"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">average_gradients()</code></span></a>

</div>

</div>

</div>

<div id="searchbox">

</div>

<div id="mlx-nn-average-gradients" class="section">

# mlx.nn.average_gradients<a
href="https://ml-explore.github.io/mlx/build/html/#mlx-nn-average-gradients"
class="headerlink" title="Link to this heading">#</a>

<span class="sig-name descname"><span class="pre">average_gradients</span></span><span class="sig-paren">(</span>*<span class="n"><span class="pre">gradients</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><a href="https://docs.python.org/3/library/typing.html#typing.Any"
class="reference external" title="(in Python v3.13)"><span
class="pre">Any</span></a></span>*, *<span class="n"><span class="pre">group</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.distributed.Group.html#mlx.core.distributed.Group"
class="reference internal" title="mlx.core.distributed.Group"><span
class="pre">Group</span></a><span class="w"> </span><span class="p"><span class="pre">\|</span></span><span class="w"> </span><a href="https://docs.python.org/3/library/constants.html#None"
class="reference external" title="(in Python v3.13)"><span
class="pre">None</span></a></span><span class="w"> </span><span class="o"><span class="pre">=</span></span><span class="w"> </span><span class="default_value"><span class="pre">None</span></span>*, *<span class="n"><span class="pre">all_reduce_size</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><a href="https://docs.python.org/3/library/functions.html#int"
class="reference external" title="(in Python v3.13)"><span
class="pre">int</span></a></span><span class="w"> </span><span class="o"><span class="pre">=</span></span><span class="w"> </span><span class="default_value"><span class="pre">33554432</span></span>*, *<span class="n"><span class="pre">communication_type</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.Dtype.html#mlx.core.Dtype"
class="reference internal" title="mlx.core.Dtype"><span
class="pre">Dtype</span></a><span class="w"> </span><span class="p"><span class="pre">\|</span></span><span class="w"> </span><a href="https://docs.python.org/3/library/constants.html#None"
class="reference external" title="(in Python v3.13)"><span
class="pre">None</span></a></span><span class="w"> </span><span class="o"><span class="pre">=</span></span><span class="w"> </span><span class="default_value"><span class="pre">None</span></span>*<span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#mlx.nn.average_gradients"
class="headerlink" title="Link to this definition">#</a>  
Average the gradients across the distributed processes in the passed
group.

This helper enables concatenating several gradients of small arrays to
one big all reduce call for better networking performance.

Parameters<span class="colon">:</span>  
- **gradients** (*Any*) – The Python tree containing the gradients (it
  should have the same structure across processes)

- **group** (*Optional\[*<a
  href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.distributed.Group.html#mlx.core.distributed.Group"
  class="reference internal"
  title="mlx.core.distributed.Group"><em>Group</em></a>*\]*) – The group
  of processes to average the gradients. If set to
  <span class="pre">`None`</span> the global group is used. Default:
  <span class="pre">`None`</span>.

- **all_reduce_size**
  (<a href="https://docs.python.org/3/library/functions.html#int"
  class="reference external" title="(in Python v3.13)"><em>int</em></a>)
  – Group arrays until their size in bytes exceeds this number. Perform
  one communication step per group of arrays. If less or equal to 0
  array grouping is disabled. Default: <span class="pre">`32MiB`</span>.

- **communication_type** (*Optional\[*<a
  href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.Dtype.html#mlx.core.Dtype"
  class="reference internal" title="mlx.core.Dtype"><em>Dtype</em></a>*\]*)
  – If provided cast to this type before performing the communication.
  Typically cast to a smaller float to reduce the communication size.
  Default: <span class="pre">`None`</span>.

</div>

<div class="prev-next-area">

<a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.nn.quantize.html"
class="left-prev" title="previous page"><em></em></a>

<div class="prev-next-info">

previous

mlx.nn.quantize

</div>

<a
href="https://ml-explore.github.io/mlx/build/html/python/nn/module.html"
class="right-next" title="next page"></a>

<div class="prev-next-info">

next

Module

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
  href="https://ml-explore.github.io/mlx/build/html/#mlx.nn.average_gradients"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">average_gradients()</code></span></a>

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
