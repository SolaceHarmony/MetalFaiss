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
  href="https://ml-explore.github.io/mlx/build/html/_sources/python/tree_utils.rst"
  class="btn btn-sm btn-download-source-button dropdown-item"
  data-bs-placement="left" data-bs-toggle="tooltip" target="_blank"
  title="Download source file"><span class="btn__icon-container">
  <em></em> </span> <span class="btn__text-container">.rst</span></a>
- <span class="btn__icon-container"> </span>
  <span class="btn__text-container">.pdf</span>

</div>

<span class="btn__icon-container"> </span>

</div>

</div>

</div>

</div>

</div>

<div id="jb-print-docs-body" class="onlyprint">

# Tree Utils

<div id="print-main-content">

<div id="jb-print-toc">

</div>

</div>

</div>

<div id="searchbox">

</div>

<div id="tree-utils" class="section">

<span id="utils"></span>

# Tree Utils<a href="https://ml-explore.github.io/mlx/build/html/#tree-utils"
class="headerlink" title="Link to this heading">#</a>

In MLX we consider a python tree to be an arbitrarily nested collection
of dictionaries, lists and tuples without cycles. Functions in this
module that return python trees will be using the default python
<span class="pre">`dict`</span>, <span class="pre">`list`</span> and
<span class="pre">`tuple`</span> but they can usually process objects
that inherit from any of these.

<div class="admonition note">

Note

Dictionaries should have keys that are valid python identifiers.

</div>

<div class="pst-scrollable-table-container">

|  |  |
|----|----|
| <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.utils.tree_flatten.html#mlx.utils.tree_flatten"
class="reference internal" title="mlx.utils.tree_flatten"><span
class="pre"><code
class="sourceCode python">tree_flatten</code></span></a>(tree\[, prefix, is_leaf\]) | Flattens a Python tree to a list of key, value tuples. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.utils.tree_unflatten.html#mlx.utils.tree_unflatten"
class="reference internal" title="mlx.utils.tree_unflatten"><span
class="pre"><code
class="sourceCode python">tree_unflatten</code></span></a>(tree) | Recreate a Python tree from its flat representation. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.utils.tree_map.html#mlx.utils.tree_map"
class="reference internal" title="mlx.utils.tree_map"><span
class="pre"><code class="sourceCode python">tree_map</code></span></a>(fn, tree, \*rest\[, is_leaf\]) | Applies <span class="pre">`fn`</span> to the leaves of the Python tree <span class="pre">`tree`</span> and returns a new collection with the results. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.utils.tree_map_with_path.html#mlx.utils.tree_map_with_path"
class="reference internal" title="mlx.utils.tree_map_with_path"><span
class="pre"><code
class="sourceCode python">tree_map_with_path</code></span></a>(fn, tree, \*rest\[, ...\]) | Applies <span class="pre">`fn`</span> to the path and leaves of the Python tree <span class="pre">`tree`</span> and returns a new collection with the results. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.utils.tree_reduce.html#mlx.utils.tree_reduce"
class="reference internal" title="mlx.utils.tree_reduce"><span
class="pre"><code
class="sourceCode python">tree_reduce</code></span></a>(fn, tree\[, initializer, is_leaf\]) | Applies a reduction to the leaves of a Python tree. |

</div>

</div>

<div class="prev-next-area">

<a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.distributed.recv_like.html"
class="left-prev" title="previous page"><em></em></a>

<div class="prev-next-info">

previous

mlx.core.distributed.recv_like

</div>

<a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.utils.tree_flatten.html"
class="right-next" title="next page"></a>

<div class="prev-next-info">

next

mlx.utils.tree_flatten

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
