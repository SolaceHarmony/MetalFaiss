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
  href="https://ml-explore.github.io/mlx/build/html/_sources/python/transforms.rst"
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

# Transforms

<div id="print-main-content">

<div id="jb-print-toc">

</div>

</div>

</div>

<div id="searchbox">

</div>

<div id="transforms" class="section">

<span id="id1"></span>

# Transforms<a href="https://ml-explore.github.io/mlx/build/html/#transforms"
class="headerlink" title="Link to this heading">#</a>

<div class="pst-scrollable-table-container">

|  |  |
|----|----|
| <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.eval.html#mlx.core.eval"
class="reference internal" title="mlx.core.eval"><span class="pre"><code
class="sourceCode python"><span class="bu">eval</span></code></span></a>(\*args) | Evaluate an <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.html#mlx.core.array"
class="reference internal" title="mlx.core.array"><span
class="pre"><code class="sourceCode python">array</code></span></a> or tree of <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.html#mlx.core.array"
class="reference internal" title="mlx.core.array"><span
class="pre"><code class="sourceCode python">array</code></span></a>. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.async_eval.html#mlx.core.async_eval"
class="reference internal" title="mlx.core.async_eval"><span
class="pre"><code class="sourceCode python">async_eval</code></span></a>(\*args) | Asynchronously evaluate an <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.html#mlx.core.array"
class="reference internal" title="mlx.core.array"><span
class="pre"><code class="sourceCode python">array</code></span></a> or tree of <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.html#mlx.core.array"
class="reference internal" title="mlx.core.array"><span
class="pre"><code class="sourceCode python">array</code></span></a>. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.compile.html#mlx.core.compile"
class="reference internal" title="mlx.core.compile"><span
class="pre"><code
class="sourceCode python"><span class="bu">compile</span></code></span></a>(fun\[, inputs, outputs, shapeless\]) | Returns a compiled function which produces the same output as <span class="pre">`fun`</span>. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.custom_function.html#mlx.core.custom_function"
class="reference internal" title="mlx.core.custom_function"><span
class="pre"><code
class="sourceCode python">custom_function</code></span></a> | Set up a function for custom gradient and vmap definitions. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.disable_compile.html#mlx.core.disable_compile"
class="reference internal" title="mlx.core.disable_compile"><span
class="pre"><code
class="sourceCode python">disable_compile</code></span></a>() | Globally disable compilation. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.enable_compile.html#mlx.core.enable_compile"
class="reference internal" title="mlx.core.enable_compile"><span
class="pre"><code
class="sourceCode python">enable_compile</code></span></a>() | Globally enable compilation. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.grad.html#mlx.core.grad"
class="reference internal" title="mlx.core.grad"><span class="pre"><code
class="sourceCode python">grad</code></span></a>(fun\[, argnums, argnames\]) | Returns a function which computes the gradient of <span class="pre">`fun`</span>. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.value_and_grad.html#mlx.core.value_and_grad"
class="reference internal" title="mlx.core.value_and_grad"><span
class="pre"><code
class="sourceCode python">value_and_grad</code></span></a>(fun\[, argnums, argnames\]) | Returns a function which computes the value and gradient of <span class="pre">`fun`</span>. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.jvp.html#mlx.core.jvp"
class="reference internal" title="mlx.core.jvp"><span class="pre"><code
class="sourceCode python">jvp</code></span></a>(fun, primals, tangents) | Compute the Jacobian-vector product. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.vjp.html#mlx.core.vjp"
class="reference internal" title="mlx.core.vjp"><span class="pre"><code
class="sourceCode python">vjp</code></span></a>(fun, primals, cotangents) | Compute the vector-Jacobian product. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.vmap.html#mlx.core.vmap"
class="reference internal" title="mlx.core.vmap"><span class="pre"><code
class="sourceCode python">vmap</code></span></a>(fun\[, in_axes, out_axes\]) | Returns a vectorized version of <span class="pre">`fun`</span>. |

</div>

</div>

<div class="prev-next-area">

<a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.random.permutation.html"
class="left-prev" title="previous page"><em></em></a>

<div class="prev-next-info">

previous

mlx.core.random.permutation

</div>

<a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.eval.html"
class="right-next" title="next page"></a>

<div class="prev-next-info">

next

mlx.core.eval

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
