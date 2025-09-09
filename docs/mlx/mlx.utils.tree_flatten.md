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
  href="https://ml-explore.github.io/mlx/build/html/_sources/python/_autosummary/mlx.utils.tree_flatten.rst"
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

# mlx.utils.tree_flatten

<div id="print-main-content">

<div id="jb-print-toc">

<div>

## Contents

</div>

- <a
  href="https://ml-explore.github.io/mlx/build/html/#mlx.utils.tree_flatten"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">tree_flatten()</code></span></a>

</div>

</div>

</div>

<div id="searchbox">

</div>

<div id="mlx-utils-tree-flatten" class="section">

# mlx.utils.tree_flatten<a
href="https://ml-explore.github.io/mlx/build/html/#mlx-utils-tree-flatten"
class="headerlink" title="Link to this heading">#</a>

<span class="sig-name descname"><span class="pre">tree_flatten</span></span><span class="sig-paren">(</span>*<span class="n"><span class="pre">tree</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><a href="https://docs.python.org/3/library/typing.html#typing.Any"
class="reference external" title="(in Python v3.13)"><span
class="pre">Any</span></a></span>*, *<span class="n"><span class="pre">prefix</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><a href="https://docs.python.org/3/library/stdtypes.html#str"
class="reference external" title="(in Python v3.13)"><span
class="pre">str</span></a></span><span class="w"> </span><span class="o"><span class="pre">=</span></span><span class="w"> </span><span class="default_value"><span class="pre">''</span></span>*, *<span class="n"><span class="pre">is_leaf</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><a href="https://docs.python.org/3/library/typing.html#typing.Callable"
class="reference external" title="(in Python v3.13)"><span
class="pre">Callable</span></a><span class="w"> </span><span class="p"><span class="pre">\|</span></span><span class="w"> </span><a href="https://docs.python.org/3/library/constants.html#None"
class="reference external" title="(in Python v3.13)"><span
class="pre">None</span></a></span><span class="w"> </span><span class="o"><span class="pre">=</span></span><span class="w"> </span><span class="default_value"><span class="pre">None</span></span>*<span class="sig-paren">)</span> <span class="sig-return"><span class="sig-return-icon">→</span> <span class="sig-return-typehint"><a href="https://docs.python.org/3/library/typing.html#typing.Any"
class="reference external" title="(in Python v3.13)"><span
class="pre">Any</span></a></span></span><a
href="https://ml-explore.github.io/mlx/build/html/#mlx.utils.tree_flatten"
class="headerlink" title="Link to this definition">#</a>  
Flattens a Python tree to a list of key, value tuples.

The keys are using the dot notation to define trees of arbitrary depth
and complexity.

<div class="highlight-python notranslate">

<div class="highlight">

    from mlx.utils import tree_flatten

    print(tree_flatten([[[0]]]))
    # [("0.0.0", 0)]

    print(tree_flatten([[[0]]], ".hello"))
    # [("hello.0.0.0", 0)]

</div>

</div>

<div class="admonition note">

Note

Dictionaries should have keys that are valid Python identifiers.

</div>

Parameters<span class="colon">:</span>  
- **tree** (*Any*) – The Python tree to be flattened.

- **prefix**
  (<a href="https://docs.python.org/3/library/stdtypes.html#str"
  class="reference external" title="(in Python v3.13)"><em>str</em></a>)
  – A prefix to use for the keys. The first character is always
  discarded.

- **is_leaf** (*callable*) – An optional callable that returns True if
  the passed object is considered a leaf or False otherwise.

Returns<span class="colon">:</span>  
The flat representation of the Python tree.

Return type<span class="colon">:</span>  
*List*\[*Tuple*\[<a href="https://docs.python.org/3/library/stdtypes.html#str"
class="reference external" title="(in Python v3.13)"><em>str</em></a>,
*Any*\]\]

</div>

<div class="prev-next-area">

<a
href="https://ml-explore.github.io/mlx/build/html/python/tree_utils.html"
class="left-prev" title="previous page"><em></em></a>

<div class="prev-next-info">

previous

Tree Utils

</div>

<a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.utils.tree_unflatten.html"
class="right-next" title="next page"></a>

<div class="prev-next-info">

next

mlx.utils.tree_unflatten

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
  href="https://ml-explore.github.io/mlx/build/html/#mlx.utils.tree_flatten"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">tree_flatten()</code></span></a>

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
