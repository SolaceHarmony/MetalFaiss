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
  href="https://ml-explore.github.io/mlx/build/html/_sources/usage/quick_start.rst"
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

# Quick Start Guide

<div id="print-main-content">

<div id="jb-print-toc">

<div>

## Contents

</div>

- <a href="https://ml-explore.github.io/mlx/build/html/#basics"
  class="reference internal nav-link">Basics</a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#function-and-graph-transformations"
  class="reference internal nav-link">Function and Graph
  Transformations</a>

</div>

</div>

</div>

<div id="searchbox">

</div>

<div id="quick-start-guide" class="section">

# Quick Start Guide<a href="https://ml-explore.github.io/mlx/build/html/#quick-start-guide"
class="headerlink" title="Link to this heading">#</a>

<div id="basics" class="section">

## Basics<a href="https://ml-explore.github.io/mlx/build/html/#basics"
class="headerlink" title="Link to this heading">#</a>

Import <span class="pre">`mlx.core`</span> and make an <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.html#mlx.core.array"
class="reference internal" title="mlx.core.array"><span
class="pre"><code class="sourceCode python">array</code></span></a>:

<div class="highlight-python notranslate">

<div class="highlight">

    >> import mlx.core as mx
    >> a = mx.array([1, 2, 3, 4])
    >> a.shape
    [4]
    >> a.dtype
    int32
    >> b = mx.array([1.0, 2.0, 3.0, 4.0])
    >> b.dtype
    float32

</div>

</div>

Operations in MLX are lazy. The outputs of MLX operations are not
computed until they are needed. To force an array to be evaluated use <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.eval.html#mlx.core.eval"
class="reference internal" title="mlx.core.eval"><span class="pre"><code
class="sourceCode python"><span class="bu">eval</span>()</code></span></a>.
Arrays will automatically be evaluated in a few cases. For example,
inspecting a scalar with <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.item.html#mlx.core.array.item"
class="reference internal" title="mlx.core.array.item"><span
class="pre"><code
class="sourceCode python">array.item()</code></span></a>, printing an
array, or converting an array from <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.html#mlx.core.array"
class="reference internal" title="mlx.core.array"><span
class="pre"><code class="sourceCode python">array</code></span></a> to
<a
href="https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray"
class="reference external" title="(in NumPy v2.2)"><span
class="pre"><code
class="sourceCode python">numpy.ndarray</code></span></a> all
automatically evaluate the array.

<div class="highlight-python notranslate">

<div class="highlight">

    >> c = a + b    # c not yet evaluated
    >> mx.eval(c)  # evaluates c
    >> c = a + b
    >> print(c)     # Also evaluates c
    array([2, 4, 6, 8], dtype=float32)
    >> c = a + b
    >> import numpy as np
    >> np.array(c)   # Also evaluates c
    array([2., 4., 6., 8.], dtype=float32)

</div>

</div>

See the page on <a
href="https://ml-explore.github.io/mlx/build/html/usage/lazy_evaluation.html#lazy-eval"
class="reference internal"><span class="std std-ref">Lazy
Evaluation</span></a> for more details.

</div>

<div id="function-and-graph-transformations" class="section">

## Function and Graph Transformations<a
href="https://ml-explore.github.io/mlx/build/html/#function-and-graph-transformations"
class="headerlink" title="Link to this heading">#</a>

MLX has standard function transformations like <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.grad.html#mlx.core.grad"
class="reference internal" title="mlx.core.grad"><span class="pre"><code
class="sourceCode python">grad()</code></span></a> and <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.vmap.html#mlx.core.vmap"
class="reference internal" title="mlx.core.vmap"><span class="pre"><code
class="sourceCode python">vmap()</code></span></a>. Transformations can
be composed arbitrarily. For example
<span class="pre">`grad(vmap(grad(fn)))`</span> (or any other
composition) is allowed.

<div class="highlight-python notranslate">

<div class="highlight">

    >> x = mx.array(0.0)
    >> mx.sin(x)
    array(0, dtype=float32)
    >> mx.grad(mx.sin)(x)
    array(1, dtype=float32)
    >> mx.grad(mx.grad(mx.sin))(x)
    array(-0, dtype=float32)

</div>

</div>

Other gradient transformations include <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.vjp.html#mlx.core.vjp"
class="reference internal" title="mlx.core.vjp"><span class="pre"><code
class="sourceCode python">vjp()</code></span></a> for vector-Jacobian
products and <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.jvp.html#mlx.core.jvp"
class="reference internal" title="mlx.core.jvp"><span class="pre"><code
class="sourceCode python">jvp()</code></span></a> for Jacobian-vector
products.

Use <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.value_and_grad.html#mlx.core.value_and_grad"
class="reference internal" title="mlx.core.value_and_grad"><span
class="pre"><code
class="sourceCode python">value_and_grad()</code></span></a> to
efficiently compute both a function’s output and gradient with respect
to the function’s input.

</div>

</div>

<div class="prev-next-area">

<a href="https://ml-explore.github.io/mlx/build/html/install.html"
class="left-prev" title="previous page"><em></em></a>

<div class="prev-next-info">

previous

Build and Install

</div>

<a
href="https://ml-explore.github.io/mlx/build/html/usage/lazy_evaluation.html"
class="right-next" title="next page"></a>

<div class="prev-next-info">

next

Lazy Evaluation

</div>

</div>

</div>

<div class="bd-sidebar-secondary bd-toc">

<div class="sidebar-secondary-items sidebar-secondary__inner">

<div class="sidebar-secondary-item">

<div class="page-toc tocsection onthispage">

Contents

</div>

- <a href="https://ml-explore.github.io/mlx/build/html/#basics"
  class="reference internal nav-link">Basics</a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#function-and-graph-transformations"
  class="reference internal nav-link">Function and Graph
  Transformations</a>

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
