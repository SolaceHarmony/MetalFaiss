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
  href="https://ml-explore.github.io/mlx/build/html/_sources/python/_autosummary/mlx.core.array.at.rst"
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

# mlx.core.array.at

<div id="print-main-content">

<div id="jb-print-toc">

<div>

## Contents

</div>

- <a href="https://ml-explore.github.io/mlx/build/html/#mlx.core.array.at"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">array.at</code></span></a>

</div>

</div>

</div>

<div id="searchbox">

</div>

<div id="mlx-core-array-at" class="section">

# mlx.core.array.at<a href="https://ml-explore.github.io/mlx/build/html/#mlx-core-array-at"
class="headerlink" title="Link to this heading">#</a>

*<span class="pre">property</span><span class="w"> </span>*<span class="sig-prename descclassname"><span class="pre">array.</span></span><span class="sig-name descname"><span class="pre">at</span></span><a href="https://ml-explore.github.io/mlx/build/html/#mlx.core.array.at"
class="headerlink" title="Link to this definition">#</a>  
Used to apply updates at the given indices.

<div class="admonition note">

Note

Regular in-place updates map to assignment. For instance
<span class="pre">`x[idx]`</span>` `<span class="pre">`+=`</span>` `<span class="pre">`y`</span>
maps to
<span class="pre">`x[idx]`</span>` `<span class="pre">`=`</span>` `<span class="pre">`x[idx]`</span>` `<span class="pre">`+`</span>` `<span class="pre">`y`</span>.
As a result, assigning to the same index ignores all but one update.
Using <span class="pre">`x.at[idx].add(y)`</span> will correctly apply
all updates to all indices.

</div>

<div class="pst-scrollable-table-container">

| array.at syntax | In-place syntax |
|----|----|
| <span class="pre">`x`</span>` `<span class="pre">`=`</span>` `<span class="pre">`x.at[idx].add(y)`</span> | <span class="pre">`x[idx]`</span>` `<span class="pre">`+=`</span>` `<span class="pre">`y`</span> |
| <span class="pre">`x`</span>` `<span class="pre">`=`</span>` `<span class="pre">`x.at[idx].subtract(y)`</span> | <span class="pre">`x[idx]`</span>` `<span class="pre">`-=`</span>` `<span class="pre">`y`</span> |
| <span class="pre">`x`</span>` `<span class="pre">`=`</span>` `<span class="pre">`x.at[idx].multiply(y)`</span> | <span class="pre">`x[idx]`</span>` `<span class="pre">`*=`</span>` `<span class="pre">`y`</span> |
| <span class="pre">`x`</span>` `<span class="pre">`=`</span>` `<span class="pre">`x.at[idx].divide(y)`</span> | <span class="pre">`x[idx]`</span>` `<span class="pre">`/=`</span>` `<span class="pre">`y`</span> |
| <span class="pre">`x`</span>` `<span class="pre">`=`</span>` `<span class="pre">`x.at[idx].maximum(y)`</span> | <span class="pre">`x[idx]`</span>` `<span class="pre">`=`</span>` `<span class="pre">`mx.maximum(x[idx],`</span>` `<span class="pre">`y)`</span> |
| <span class="pre">`x`</span>` `<span class="pre">`=`</span>` `<span class="pre">`x.at[idx].minimum(y)`</span> | <span class="pre">`x[idx]`</span>` `<span class="pre">`=`</span>` `<span class="pre">`mx.minimum(x[idx],`</span>` `<span class="pre">`y)`</span> |

</div>

Example

<div class="doctest highlight-default notranslate">

<div class="highlight">

    >>> a = mx.array([0, 0])
    >>> idx = mx.array([0, 1, 0, 1])
    >>> a[idx] += 1
    >>> a
    array([1, 1], dtype=int32)
    >>>
    >>> a = mx.array([0, 0])
    >>> a.at[idx].add(1)
    array([2, 2], dtype=int32)

</div>

</div>

</div>

<div class="prev-next-area">

<a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.astype.html"
class="left-prev" title="previous page"><em></em></a>

<div class="prev-next-info">

previous

mlx.core.array.astype

</div>

<a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.item.html"
class="right-next" title="next page"></a>

<div class="prev-next-info">

next

mlx.core.array.item

</div>

</div>

</div>

<div class="bd-sidebar-secondary bd-toc">

<div class="sidebar-secondary-items sidebar-secondary__inner">

<div class="sidebar-secondary-item">

<div class="page-toc tocsection onthispage">

Contents

</div>

- <a href="https://ml-explore.github.io/mlx/build/html/#mlx.core.array.at"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">array.at</code></span></a>

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
