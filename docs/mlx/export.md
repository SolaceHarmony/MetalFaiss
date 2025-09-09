Curator's note: Prefer the human-authored MLX guides for clarity.
- ../docs_curated/README.md
- ../docs_curated/PYTORCH_DISSONANCE.md
- ../docs_curated/NUMPY_USERS.md
- ../docs_curated/COMMON_PITFALLS.md

<!--
Per-file analysis (export.md):
- Lists export utilities for compiled graphs/functions.
- Users benefit from quick examples and a DOT visualization hint.
-->

## Curated Example

```python
import mlx.core as mx

def f(x, y):
    return mx.tanh(x @ y + 0.1)

mx.export_function("f.mlx", f, mx.ones((4, 4)), mx.ones((4, 4)))
g = mx.import_function("f.mlx")
print(g(mx.ones((4, 4)), mx.ones((4, 4))))

mx.export_to_dot("f.dot", mx.ones((4, 4)), mx.ones((4, 4)))
```


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
  href="https://ml-explore.github.io/mlx/build/html/_sources/python/export.rst"
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

# Export Functions

<div id="print-main-content">

<div id="jb-print-toc">

</div>

</div>

</div>

<div id="searchbox">

</div>

<div id="export-functions" class="section">

<span id="export"></span>

# Export Functions<a href="https://ml-explore.github.io/mlx/build/html/#export-functions"
class="headerlink" title="Link to this heading">#</a>

<div class="pst-scrollable-table-container">

|  |  |
|----|----|
| <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.export_function.html#mlx.core.export_function"
class="reference internal" title="mlx.core.export_function"><span
class="pre"><code
class="sourceCode python">export_function</code></span></a>(file, fun, \*args\[, shapeless\]) | Export a function to a file. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.import_function.html#mlx.core.import_function"
class="reference internal" title="mlx.core.import_function"><span
class="pre"><code
class="sourceCode python">import_function</code></span></a>(file) | Import a function from a file. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.exporter.html#mlx.core.exporter"
class="reference internal" title="mlx.core.exporter"><span
class="pre"><code class="sourceCode python">exporter</code></span></a>(file, fun, \*\[, shapeless\]) | Make a callable object to export multiple traces of a function to a file. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.export_to_dot.html#mlx.core.export_to_dot"
class="reference internal" title="mlx.core.export_to_dot"><span
class="pre"><code
class="sourceCode python">export_to_dot</code></span></a>(file, \*args, \*\*kwargs) | Export a graph to DOT format for visualization. |

</div>

</div>

<div class="prev-next-area">

<a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.synchronize.html"
class="left-prev" title="previous page"><em></em></a>

<div class="prev-next-info">

previous

mlx.core.synchronize

</div>

<a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.export_function.html"
class="right-next" title="next page"></a>

<div class="prev-next-info">

next

mlx.core.export_function

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
