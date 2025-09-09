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
  href="https://ml-explore.github.io/mlx/build/html/_sources/python/_autosummary/mlx.core.linalg.cholesky_inv.rst"
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

# mlx.core.linalg.cholesky_inv

<div id="print-main-content">

<div id="jb-print-toc">

<div>

## Contents

</div>

- <a
  href="https://ml-explore.github.io/mlx/build/html/#mlx.core.linalg.cholesky_inv"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">cholesky_inv()</code></span></a>

</div>

</div>

</div>

<div id="searchbox">

</div>

<div id="mlx-core-linalg-cholesky-inv" class="section">

# mlx.core.linalg.cholesky_inv<a
href="https://ml-explore.github.io/mlx/build/html/#mlx-core-linalg-cholesky-inv"
class="headerlink" title="Link to this heading">#</a>

<span class="sig-name descname"><span class="pre">cholesky_inv</span></span><span class="sig-paren">(</span>*<span class="n"><span class="pre">L</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.html#mlx.core.array"
class="reference internal" title="mlx.core.array"><span
class="pre">array</span></a></span>*, *<span class="n"><span class="pre">upper</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><a href="https://docs.python.org/3/library/functions.html#bool"
class="reference external" title="(in Python v3.13)"><span
class="pre">bool</span></a></span><span class="w"> </span><span class="o"><span class="pre">=</span></span><span class="w"> </span><span class="default_value"><span class="pre">False</span></span>*, *<span class="o"><span class="pre">\*</span></span>*, *<span class="n"><span class="pre">stream</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><a href="https://docs.python.org/3/library/constants.html#None"
class="reference external" title="(in Python v3.13)"><span
class="pre">None</span></a><span class="w"> </span><span class="p"><span class="pre">\|</span></span><span class="w"> </span><a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/stream_class.html#mlx.core.Stream"
class="reference internal" title="mlx.core.Stream"><span
class="pre">Stream</span></a><span class="w"> </span><span class="p"><span class="pre">\|</span></span><span class="w"> </span><a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.Device.html#mlx.core.Device"
class="reference internal" title="mlx.core.Device"><span
class="pre">Device</span></a></span><span class="w"> </span><span class="o"><span class="pre">=</span></span><span class="w"> </span><span class="default_value"><span class="pre">None</span></span>*<span class="sig-paren">)</span> <span class="sig-return"><span class="sig-return-icon">→</span> <span class="sig-return-typehint"><a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.html#mlx.core.array"
class="reference internal" title="mlx.core.array"><span
class="pre">array</span></a></span></span><a
href="https://ml-explore.github.io/mlx/build/html/#mlx.core.linalg.cholesky_inv"
class="headerlink" title="Link to this definition">#</a>  
Compute the inverse of a real symmetric positive semi-definite matrix
using it’s Cholesky decomposition.

Let <span class="math notranslate nohighlight">\\\mathbf{A}\\</span> be
a real symmetric positive semi-definite matrix and
<span class="math notranslate nohighlight">\\\mathbf{L}\\</span> its
Cholesky decomposition such that:

<div class="math notranslate nohighlight">

\\\begin{aligned} \mathbf{A} = \mathbf{L}\mathbf{L}^T \end{aligned}\\

</div>

This function computes
<span class="math notranslate nohighlight">\\\mathbf{A}^{-1}\\</span>.

This function supports arrays with at least 2 dimensions. When the input
has more than two dimensions, the Cholesky inverse is computed for each
matrix in the last two dimensions of
<span class="math notranslate nohighlight">\\\mathbf{L}\\</span>.

If the input matrix is not a triangular matrix behaviour is undefined.

Parameters<span class="colon">:</span>  
- **L** (<a
  href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.html#mlx.core.array"
  class="reference internal" title="mlx.core.array"><em>array</em></a>)
  – Input array.

- **upper**
  (<a href="https://docs.python.org/3/library/functions.html#bool"
  class="reference external" title="(in Python v3.13)"><em>bool</em></a>*,*
  *optional*) – If <span class="pre">`True`</span>, return the upper
  triangular Cholesky factor. If <span class="pre">`False`</span>,
  return the lower triangular Cholesky factor. Default:
  <span class="pre">`False`</span>.

- **stream** (<a
  href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/stream_class.html#mlx.core.Stream"
  class="reference internal" title="mlx.core.Stream"><em>Stream</em></a>*,*
  *optional*) – Stream or device. Defaults to
  <span class="pre">`None`</span> in which case the default stream of
  the default device is used.

Returns<span class="colon">:</span>  
<span class="math notranslate nohighlight">\\\mathbf{A^{-1}}\\</span>
where <span class="math notranslate nohighlight">\\\mathbf{A} =
\mathbf{L}\mathbf{L}^T\\</span>.

Return type<span class="colon">:</span>  
<a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.html#mlx.core.array"
class="reference internal" title="mlx.core.array"><em>array</em></a>

</div>

<div class="prev-next-area">

<a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.linalg.cholesky.html"
class="left-prev" title="previous page"><em></em></a>

<div class="prev-next-info">

previous

mlx.core.linalg.cholesky

</div>

<a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.linalg.cross.html"
class="right-next" title="next page"></a>

<div class="prev-next-info">

next

mlx.core.linalg.cross

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
  href="https://ml-explore.github.io/mlx/build/html/#mlx.core.linalg.cholesky_inv"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">cholesky_inv()</code></span></a>

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
