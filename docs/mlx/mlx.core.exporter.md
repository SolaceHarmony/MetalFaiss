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
  href="https://ml-explore.github.io/mlx/build/html/_sources/python/_autosummary/mlx.core.exporter.rst"
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

# mlx.core.exporter

<div id="print-main-content">

<div id="jb-print-toc">

<div>

## Contents

</div>

- <a href="https://ml-explore.github.io/mlx/build/html/#mlx.core.exporter"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">exporter()</code></span></a>

</div>

</div>

</div>

<div id="searchbox">

</div>

<div id="mlx-core-exporter" class="section">

# mlx.core.exporter<a href="https://ml-explore.github.io/mlx/build/html/#mlx-core-exporter"
class="headerlink" title="Link to this heading">#</a>

<span class="sig-name descname"><span class="pre">exporter</span></span><span class="sig-paren">(</span>*<span class="n"><span class="pre">file</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><a href="https://docs.python.org/3/library/stdtypes.html#str"
class="reference external" title="(in Python v3.13)"><span
class="pre">str</span></a></span>*, *<span class="n"><span class="pre">fun</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><a
href="https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable"
class="reference external" title="(in Python v3.13)"><span
class="pre">Callable</span></a></span>*, *<span class="o"><span class="pre">\*</span></span>*, *<span class="n"><span class="pre">shapeless</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><a href="https://docs.python.org/3/library/functions.html#bool"
class="reference external" title="(in Python v3.13)"><span
class="pre">bool</span></a></span><span class="w"> </span><span class="o"><span class="pre">=</span></span><span class="w"> </span><span class="default_value"><span class="pre">False</span></span>*<span class="sig-paren">)</span> <span class="sig-return"><span class="sig-return-icon">→</span> <span class="sig-return-typehint"><span class="pre">mlx.core.FunctionExporter</span></span></span><a href="https://ml-explore.github.io/mlx/build/html/#mlx.core.exporter"
class="headerlink" title="Link to this definition">#</a>  
Make a callable object to export multiple traces of a function to a
file.

<div class="admonition warning">

Warning

This is part of an experimental API which is likely to change in future
versions of MLX. Functions exported with older versions of MLX may not
be compatible with future versions.

</div>

Parameters<span class="colon">:</span>  
- **file**
  (<a href="https://docs.python.org/3/library/stdtypes.html#str"
  class="reference external" title="(in Python v3.13)"><em>str</em></a>)
  – File path to export the function to.

- **shapeless**
  (<a href="https://docs.python.org/3/library/functions.html#bool"
  class="reference external" title="(in Python v3.13)"><em>bool</em></a>*,*
  *optional*) – Whether or not the function allows inputs with variable
  shapes. Default: <span class="pre">`False`</span>.

Example

<div class="highlight-python notranslate">

<div class="highlight">

    def fun(*args):
        return sum(args)

    with mx.exporter("fun.mlxfn", fun) as exporter:
        exporter(mx.array(1))
        exporter(mx.array(1), mx.array(2))
        exporter(mx.array(1), mx.array(2), mx.array(3))

</div>

</div>

</div>

<div class="prev-next-area">

<a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.import_function.html"
class="left-prev" title="previous page"><em></em></a>

<div class="prev-next-info">

previous

mlx.core.import_function

</div>

<a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.export_to_dot.html"
class="right-next" title="next page"></a>

<div class="prev-next-info">

next

mlx.core.export_to_dot

</div>

</div>

</div>

<div class="bd-sidebar-secondary bd-toc">

<div class="sidebar-secondary-items sidebar-secondary__inner">

<div class="sidebar-secondary-item">

<div class="page-toc tocsection onthispage">

Contents

</div>

- <a href="https://ml-explore.github.io/mlx/build/html/#mlx.core.exporter"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">exporter()</code></span></a>

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
