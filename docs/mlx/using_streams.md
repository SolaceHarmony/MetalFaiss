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
  href="https://ml-explore.github.io/mlx/build/html/_sources/usage/using_streams.rst"
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

# Using Streams

<div id="print-main-content">

<div id="jb-print-toc">

<div>

## Contents

</div>

- <a
  href="https://ml-explore.github.io/mlx/build/html/#specifying-the-stream"
  class="reference internal nav-link">Specifying the <span
  class="pre"><code class="sourceCode python">Stream</code></span></a>

</div>

</div>

</div>

<div id="searchbox">

</div>

<div id="using-streams" class="section">

<span id="id1"></span>

# Using Streams<a href="https://ml-explore.github.io/mlx/build/html/#using-streams"
class="headerlink" title="Link to this heading">#</a>

<div id="specifying-the-stream" class="section">

## Specifying the <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/stream_class.html#mlx.core.Stream"
class="reference internal" title="mlx.core.Stream"><span
class="pre"><code class="sourceCode python">Stream</code></span></a><a
href="https://ml-explore.github.io/mlx/build/html/#specifying-the-stream"
class="headerlink" title="Link to this heading">#</a>

All operations (including random number generation) take an optional
keyword argument <span class="pre">`stream`</span>. The
<span class="pre">`stream`</span> kwarg specifies which <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/stream_class.html#mlx.core.Stream"
class="reference internal" title="mlx.core.Stream"><span
class="pre"><code class="sourceCode python">Stream</code></span></a> the
operation should run on. If the stream is unspecified then the operation
is run on the default stream of the default device:
<span class="pre">`mx.default_stream(mx.default_device())`</span>. The
<span class="pre">`stream`</span> kwarg can also be a <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.Device.html#mlx.core.Device"
class="reference internal" title="mlx.core.Device"><span
class="pre"><code class="sourceCode python">Device</code></span></a>
(e.g. <span class="pre">`stream=my_device`</span>) in which case the
operation is run on the default stream of the provided device
<span class="pre">`mx.default_stream(my_device)`</span>.

</div>

</div>

<div class="prev-next-area">

<a
href="https://ml-explore.github.io/mlx/build/html/usage/distributed.html"
class="left-prev" title="previous page"><em></em></a>

<div class="prev-next-info">

previous

Distributed Communication

</div>

<a href="https://ml-explore.github.io/mlx/build/html/usage/export.html"
class="right-next" title="next page"></a>

<div class="prev-next-info">

next

Exporting Functions

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
  href="https://ml-explore.github.io/mlx/build/html/#specifying-the-stream"
  class="reference internal nav-link">Specifying the <span
  class="pre"><code class="sourceCode python">Stream</code></span></a>

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
