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
  href="https://ml-explore.github.io/mlx/build/html/_sources/python/_autosummary/mlx.utils.tree_map_with_path.rst"
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

# mlx.utils.tree_map_with_path

<div id="print-main-content">

<div id="jb-print-toc">

<div>

## Contents

</div>

- <a
  href="https://ml-explore.github.io/mlx/build/html/#mlx.utils.tree_map_with_path"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">tree_map_with_path()</code></span></a>

</div>

</div>

</div>

<div id="searchbox">

</div>

<div id="mlx-utils-tree-map-with-path" class="section">

# mlx.utils.tree_map_with_path<a
href="https://ml-explore.github.io/mlx/build/html/#mlx-utils-tree-map-with-path"
class="headerlink" title="Link to this heading">#</a>

<span class="sig-name descname"><span class="pre">tree_map_with_path</span></span><span class="sig-paren">(</span>*<span class="n"><span class="pre">fn</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><a href="https://docs.python.org/3/library/typing.html#typing.Callable"
class="reference external" title="(in Python v3.13)"><span
class="pre">Callable</span></a></span>*, *<span class="n"><span class="pre">tree</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><a href="https://docs.python.org/3/library/typing.html#typing.Any"
class="reference external" title="(in Python v3.13)"><span
class="pre">Any</span></a></span>*, *<span class="o"><span class="pre">\*</span></span><span class="n"><span class="pre">rest</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><a href="https://docs.python.org/3/library/typing.html#typing.Any"
class="reference external" title="(in Python v3.13)"><span
class="pre">Any</span></a></span>*, *<span class="n"><span class="pre">is_leaf</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><a href="https://docs.python.org/3/library/typing.html#typing.Callable"
class="reference external" title="(in Python v3.13)"><span
class="pre">Callable</span></a><span class="w"> </span><span class="p"><span class="pre">\|</span></span><span class="w"> </span><a href="https://docs.python.org/3/library/constants.html#None"
class="reference external" title="(in Python v3.13)"><span
class="pre">None</span></a></span><span class="w"> </span><span class="o"><span class="pre">=</span></span><span class="w"> </span><span class="default_value"><span class="pre">None</span></span>*, *<span class="n"><span class="pre">path</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><a href="https://docs.python.org/3/library/typing.html#typing.Any"
class="reference external" title="(in Python v3.13)"><span
class="pre">Any</span></a><span class="w"> </span><span class="p"><span class="pre">\|</span></span><span class="w"> </span><a href="https://docs.python.org/3/library/constants.html#None"
class="reference external" title="(in Python v3.13)"><span
class="pre">None</span></a></span><span class="w"> </span><span class="o"><span class="pre">=</span></span><span class="w"> </span><span class="default_value"><span class="pre">None</span></span>*<span class="sig-paren">)</span> <span class="sig-return"><span class="sig-return-icon">→</span> <span class="sig-return-typehint"><a href="https://docs.python.org/3/library/typing.html#typing.Any"
class="reference external" title="(in Python v3.13)"><span
class="pre">Any</span></a></span></span><a
href="https://ml-explore.github.io/mlx/build/html/#mlx.utils.tree_map_with_path"
class="headerlink" title="Link to this definition">#</a>  
Applies <span class="pre">`fn`</span> to the path and leaves of the
Python tree <span class="pre">`tree`</span> and returns a new collection
with the results.

This function is the same <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.utils.tree_map.html#mlx.utils.tree_map"
class="reference internal" title="mlx.utils.tree_map"><span
class="pre"><code class="sourceCode python">tree_map()</code></span></a>
but the <span class="pre">`fn`</span> takes the path as the first
argument followed by the remaining tree nodes.

Parameters<span class="colon">:</span>  
- **fn** (*callable*) – The function that processes the leaves of the
  tree.

- **tree** (*Any*) – The main Python tree that will be iterated upon.

- **rest**
  (<a href="https://docs.python.org/3/library/stdtypes.html#tuple"
  class="reference external" title="(in Python v3.13)"><em>tuple</em></a>*\[Any\]*)
  – Extra trees to be iterated together with
  <span class="pre">`tree`</span>.

- **is_leaf** (*Optional\[Callable\]*) – An optional callable that
  returns <span class="pre">`True`</span> if the passed object is
  considered a leaf or <span class="pre">`False`</span> otherwise.

- **path** (*Optional\[Any\]*) – Prefix will be added to the result.

Returns<span class="colon">:</span>  
A Python tree with the new values returned by
<span class="pre">`fn`</span>.

Example

<div class="doctest highlight-default notranslate">

<div class="highlight">

    >>> from mlx.utils import tree_map_with_path
    >>> tree = {"model": [{"w": 0, "b": 1}, {"w": 0, "b": 1}]}
    >>> new_tree = tree_map_with_path(lambda path, _: print(path), tree)
    model.0.w
    model.0.b
    model.1.w
    model.1.b

</div>

</div>

</div>

<div class="prev-next-area">

<a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.utils.tree_map.html"
class="left-prev" title="previous page"><em></em></a>

<div class="prev-next-info">

previous

mlx.utils.tree_map

</div>

<a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.utils.tree_reduce.html"
class="right-next" title="next page"></a>

<div class="prev-next-info">

next

mlx.utils.tree_reduce

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
  href="https://ml-explore.github.io/mlx/build/html/#mlx.utils.tree_map_with_path"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">tree_map_with_path()</code></span></a>

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
