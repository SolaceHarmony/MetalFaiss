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
  href="https://ml-explore.github.io/mlx/build/html/_sources/python/_autosummary/mlx.utils.tree_reduce.rst"
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

# mlx.utils.tree_reduce

<div id="print-main-content">

<div id="jb-print-toc">

<div>

## Contents

</div>

- <a
  href="https://ml-explore.github.io/mlx/build/html/#mlx.utils.tree_reduce"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">tree_reduce()</code></span></a>

</div>

</div>

</div>

<div id="searchbox">

</div>

<div id="mlx-utils-tree-reduce" class="section">

# mlx.utils.tree_reduce<a
href="https://ml-explore.github.io/mlx/build/html/#mlx-utils-tree-reduce"
class="headerlink" title="Link to this heading">#</a>

<span class="sig-name descname"><span class="pre">tree_reduce</span></span><span class="sig-paren">(</span>*<span class="n"><span class="pre">fn</span></span>*, *<span class="n"><span class="pre">tree</span></span>*, *<span class="n"><span class="pre">initializer</span></span><span class="o"><span class="pre">=</span></span><span class="default_value"><span class="pre">None</span></span>*, *<span class="n"><span class="pre">is_leaf</span></span><span class="o"><span class="pre">=</span></span><span class="default_value"><span class="pre">None</span></span>*<span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#mlx.utils.tree_reduce"
class="headerlink" title="Link to this definition">#</a>  
Applies a reduction to the leaves of a Python tree.

This function reduces Python trees into an accumulated result by
applying the provided function <span class="pre">`fn`</span> to the
leaves of the tree.

Example

<div class="doctest highlight-default notranslate">

<div class="highlight">

    >>> from mlx.utils import tree_reduce
    >>> tree = {"a": [1, 2, 3], "b": [4, 5]}
    >>> tree_reduce(lambda acc, x: acc + x, tree, 0)
    15

</div>

</div>

Parameters<span class="colon">:</span>  
- **fn** (*callable*) – The reducer function that takes two arguments
  (accumulator, current value) and returns the updated accumulator.

- **tree** (*Any*) – The Python tree to reduce. It can be any nested
  combination of lists, tuples, or dictionaries.

- **initializer** (*Any,* *optional*) – The initial value to start the
  reduction. If not provided, the first leaf value is used.

- **is_leaf** (*callable,* *optional*) – A function to determine if an
  object is a leaf, returning <span class="pre">`True`</span> for leaf
  nodes and <span class="pre">`False`</span> otherwise.

Returns<span class="colon">:</span>  
The accumulated value.

Return type<span class="colon">:</span>  
*Any*

</div>

<div class="prev-next-area">

<a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.utils.tree_map_with_path.html"
class="left-prev" title="previous page"><em></em></a>

<div class="prev-next-info">

previous

mlx.utils.tree_map_with_path

</div>

<a href="https://ml-explore.github.io/mlx/build/html/cpp/ops.html"
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
  href="https://ml-explore.github.io/mlx/build/html/#mlx.utils.tree_reduce"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">tree_reduce()</code></span></a>

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
