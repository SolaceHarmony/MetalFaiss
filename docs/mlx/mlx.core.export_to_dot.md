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
  href="https://ml-explore.github.io/mlx/build/html/_sources/python/_autosummary/mlx.core.export_to_dot.rst"
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

# mlx.core.export_to_dot

<div id="print-main-content">

<div id="jb-print-toc">

<div>

## Contents

</div>

- <a
  href="https://ml-explore.github.io/mlx/build/html/#mlx.core.export_to_dot"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">export_to_dot()</code></span></a>

</div>

</div>

</div>

<div id="searchbox">

</div>

<div id="mlx-core-export-to-dot" class="section">

# mlx.core.export_to_dot<a
href="https://ml-explore.github.io/mlx/build/html/#mlx-core-export-to-dot"
class="headerlink" title="Link to this heading">#</a>

<span class="sig-name descname"><span class="pre">export_to_dot</span></span><span class="sig-paren">(</span>*<span class="n"><span class="pre">file</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><a href="https://docs.python.org/3/library/functions.html#object"
class="reference external" title="(in Python v3.13)"><span
class="pre">object</span></a></span>*, *<span class="o"><span class="pre">\*</span></span><span class="n"><span class="pre">args</span></span>*, *<span class="o"><span class="pre">\*\*</span></span><span class="n"><span class="pre">kwargs</span></span>*<span class="sig-paren">)</span> <span class="sig-return"><span class="sig-return-icon">→</span> <span class="sig-return-typehint"><a href="https://docs.python.org/3/library/constants.html#None"
class="reference external" title="(in Python v3.13)"><span
class="pre">None</span></a></span></span><a
href="https://ml-explore.github.io/mlx/build/html/#mlx.core.export_to_dot"
class="headerlink" title="Link to this definition">#</a>  
Export a graph to DOT format for visualization.

A variable number of output arrays can be provided for exporting The
graph exported will recursively include all unevaluated inputs of the
provided outputs.

Parameters<span class="colon">:</span>  
- **file**
  (<a href="https://docs.python.org/3/library/stdtypes.html#str"
  class="reference external" title="(in Python v3.13)"><em>str</em></a>)
  – The file path to export to.

- **\*args** (<a
  href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.html#mlx.core.array"
  class="reference internal" title="mlx.core.array"><em>array</em></a>)
  – The output arrays.

- **\*\*kwargs**
  (<a href="https://docs.python.org/3/library/stdtypes.html#dict"
  class="reference external" title="(in Python v3.13)"><em>dict</em></a>*\[*<a href="https://docs.python.org/3/library/stdtypes.html#str"
  class="reference external" title="(in Python v3.13)"><em>str</em></a>*,*
  <a
  href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.html#mlx.core.array"
  class="reference internal" title="mlx.core.array"><em>array</em></a>*\]*)
  – Provide some names for arrays in the graph to make the result easier
  to parse.

Example

<div class="doctest highlight-default notranslate">

<div class="highlight">

    >>> a = mx.array(1) + mx.array(2)
    >>> mx.export_to_dot("graph.dot", a)
    >>> x = mx.array(1)
    >>> y = mx.array(2)
    >>> mx.export_to_dot("graph.dot", x + y, x=x, y=y)

</div>

</div>

</div>

<div class="prev-next-area">

<a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.exporter.html"
class="left-prev" title="previous page"><em></em></a>

<div class="prev-next-info">

previous

mlx.core.exporter

</div>

<a href="https://ml-explore.github.io/mlx/build/html/python/ops.html"
class="right-next" title="next page"></a>

<div class="prev-next-info">

next

Operations

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
  href="https://ml-explore.github.io/mlx/build/html/#mlx.core.export_to_dot"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">export_to_dot()</code></span></a>

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
