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
  href="https://ml-explore.github.io/mlx/build/html/_sources/python/_autosummary/mlx.core.savez.rst"
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

# mlx.core.savez

<div id="print-main-content">

<div id="jb-print-toc">

<div>

## Contents

</div>

- <a href="https://ml-explore.github.io/mlx/build/html/#mlx.core.savez"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">savez()</code></span></a>

</div>

</div>

</div>

<div id="searchbox">

</div>

<div id="mlx-core-savez" class="section">

# mlx.core.savez<a href="https://ml-explore.github.io/mlx/build/html/#mlx-core-savez"
class="headerlink" title="Link to this heading">#</a>

<span class="sig-name descname"><span class="pre">savez</span></span><span class="sig-paren">(</span>*<span class="n"><span class="pre">file</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><a href="https://docs.python.org/3/library/functions.html#object"
class="reference external" title="(in Python v3.13)"><span
class="pre">object</span></a></span>*, *<span class="o"><span class="pre">\*</span></span><span class="n"><span class="pre">args</span></span>*, *<span class="o"><span class="pre">\*\*</span></span><span class="n"><span class="pre">kwargs</span></span>*<span class="sig-paren">)</span> <span class="sig-return"><span class="sig-return-icon">→</span> <span class="sig-return-typehint"><a href="https://docs.python.org/3/library/constants.html#None"
class="reference external" title="(in Python v3.13)"><span
class="pre">None</span></a></span></span><a href="https://ml-explore.github.io/mlx/build/html/#mlx.core.savez"
class="headerlink" title="Link to this definition">#</a>  
Save several arrays to a binary file in uncompressed
<span class="pre">`.npz`</span> format.

<div class="highlight-python notranslate">

<div class="highlight">

    import mlx.core as mx

    x = mx.ones((10, 10))
    mx.savez("my_path.npz", x=x)

    import mlx.nn as nn
    from mlx.utils import tree_flatten

    model = nn.TransformerEncoder(6, 128, 4)
    flat_params = tree_flatten(model.parameters())
    mx.savez("model.npz", **dict(flat_params))

</div>

</div>

Parameters<span class="colon">:</span>  
- **file** (*file,*
  <a href="https://docs.python.org/3/library/stdtypes.html#str"
  class="reference external" title="(in Python v3.13)"><em>str</em></a>)
  – Path to file to which the arrays are saved.

- **\*args** (*arrays*) – Arrays to be saved.

- **\*\*kwargs** (*arrays*) – Arrays to be saved. Each array will be
  saved with the associated keyword as the output file name.

</div>

<div class="prev-next-area">

<a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.save.html"
class="left-prev" title="previous page"><em></em></a>

<div class="prev-next-info">

previous

mlx.core.save

</div>

<a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.savez_compressed.html"
class="right-next" title="next page"></a>

<div class="prev-next-info">

next

mlx.core.savez_compressed

</div>

</div>

</div>

<div class="bd-sidebar-secondary bd-toc">

<div class="sidebar-secondary-items sidebar-secondary__inner">

<div class="sidebar-secondary-item">

<div class="page-toc tocsection onthispage">

Contents

</div>

- <a href="https://ml-explore.github.io/mlx/build/html/#mlx.core.savez"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">savez()</code></span></a>

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
