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
  href="https://ml-explore.github.io/mlx/build/html/_sources/python/_autosummary/mlx.core.issubdtype.rst"
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

# mlx.core.issubdtype

<div id="print-main-content">

<div id="jb-print-toc">

<div>

## Contents

</div>

- <a
  href="https://ml-explore.github.io/mlx/build/html/#mlx.core.issubdtype"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">issubdtype()</code></span></a>

</div>

</div>

</div>

<div id="searchbox">

</div>

<div id="mlx-core-issubdtype" class="section">

# mlx.core.issubdtype<a
href="https://ml-explore.github.io/mlx/build/html/#mlx-core-issubdtype"
class="headerlink" title="Link to this heading">#</a>

<span class="sig-name descname"><span class="pre">issubdtype</span></span><span class="sig-paren">(</span>*<span class="n"><span class="pre">arg1</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.Dtype.html#mlx.core.Dtype"
class="reference internal" title="mlx.core.Dtype"><span
class="pre">Dtype</span></a><span class="w"> </span><span class="p"><span class="pre">\|</span></span><span class="w"> </span><a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.DtypeCategory.html#mlx.core.DtypeCategory"
class="reference internal" title="mlx.core.DtypeCategory"><span
class="pre">DtypeCategory</span></a></span>*, *<span class="n"><span class="pre">arg2</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.Dtype.html#mlx.core.Dtype"
class="reference internal" title="mlx.core.Dtype"><span
class="pre">Dtype</span></a><span class="w"> </span><span class="p"><span class="pre">\|</span></span><span class="w"> </span><a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.DtypeCategory.html#mlx.core.DtypeCategory"
class="reference internal" title="mlx.core.DtypeCategory"><span
class="pre">DtypeCategory</span></a></span>*<span class="sig-paren">)</span> <span class="sig-return"><span class="sig-return-icon">→</span> <span class="sig-return-typehint"><a href="https://docs.python.org/3/library/functions.html#bool"
class="reference external" title="(in Python v3.13)"><span
class="pre">bool</span></a></span></span><a
href="https://ml-explore.github.io/mlx/build/html/#mlx.core.issubdtype"
class="headerlink" title="Link to this definition">#</a>  
Check if a <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.Dtype.html#mlx.core.Dtype"
class="reference internal" title="mlx.core.Dtype"><span
class="pre"><code class="sourceCode python">Dtype</code></span></a> or
<a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.DtypeCategory.html#mlx.core.DtypeCategory"
class="reference internal" title="mlx.core.DtypeCategory"><span
class="pre"><code
class="sourceCode python">DtypeCategory</code></span></a> is a subtype
of another.

Parameters<span class="colon">:</span>  
- **(Union\[Dtype** (*arg2*) – First dtype or category.

- **DtypeCategory\]** – First dtype or category.

- **(Union\[Dtype** – Second dtype or category.

- **DtypeCategory\]** – Second dtype or category.

Returns<span class="colon">:</span>  
A boolean indicating if the first input is a subtype of the second
input.

Return type<span class="colon">:</span>  
<a href="https://docs.python.org/3/library/functions.html#bool"
class="reference external" title="(in Python v3.13)"><em>bool</em></a>

Example

<div class="doctest highlight-default notranslate">

<div class="highlight">

    >>> ints = mx.array([1, 2, 3], dtype=mx.int32)
    >>> mx.issubdtype(ints.dtype, mx.integer)
    True
    >>> mx.issubdtype(ints.dtype, mx.floating)
    False

</div>

</div>

<div class="doctest highlight-default notranslate">

<div class="highlight">

    >>> floats = mx.array([1, 2, 3], dtype=mx.float32)
    >>> mx.issubdtype(floats.dtype, mx.integer)
    False
    >>> mx.issubdtype(floats.dtype, mx.floating)
    True

</div>

</div>

Similar types of different sizes are not subdtypes of each other:

<div class="doctest highlight-default notranslate">

<div class="highlight">

    >>> mx.issubdtype(mx.float64, mx.float32)
    False
    >>> mx.issubdtype(mx.float32, mx.float64)
    False

</div>

</div>

but both are subtypes of floating:

<div class="doctest highlight-default notranslate">

<div class="highlight">

    >>> mx.issubdtype(mx.float64, mx.floating)
    True
    >>> mx.issubdtype(mx.float32, mx.floating)
    True

</div>

</div>

For convenience, dtype-like objects are allowed too:

<div class="doctest highlight-default notranslate">

<div class="highlight">

    >>> mx.issubdtype(mx.float32, mx.inexact)
    True
    >>> mx.issubdtype(mx.signedinteger, mx.floating)
    False

</div>

</div>

</div>

<div class="prev-next-area">

<a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.DtypeCategory.html"
class="left-prev" title="previous page"><em></em></a>

<div class="prev-next-info">

previous

mlx.core.DtypeCategory

</div>

<a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.finfo.html"
class="right-next" title="next page"></a>

<div class="prev-next-info">

next

mlx.core.finfo

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
  href="https://ml-explore.github.io/mlx/build/html/#mlx.core.issubdtype"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">issubdtype()</code></span></a>

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
