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
  href="https://ml-explore.github.io/mlx/build/html/_sources/usage/unified_memory.rst"
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

# Unified Memory

<div id="print-main-content">

<div id="jb-print-toc">

<div>

## Contents

</div>

- <a href="https://ml-explore.github.io/mlx/build/html/#a-simple-example"
  class="reference internal nav-link">A Simple Example</a>

</div>

</div>

</div>

<div id="searchbox">

</div>

<div id="unified-memory" class="section">

<span id="id1"></span>

# Unified Memory<a href="https://ml-explore.github.io/mlx/build/html/#unified-memory"
class="headerlink" title="Link to this heading">#</a>

Apple silicon has a unified memory architecture. The CPU and GPU have
direct access to the same memory pool. MLX is designed to take advantage
of that.

Concretely, when you make an array in MLX you don’t have to specify its
location:

<div class="highlight-python notranslate">

<div class="highlight">

    a = mx.random.normal((100,))
    b = mx.random.normal((100,))

</div>

</div>

Both <span class="pre">`a`</span> and <span class="pre">`b`</span> live
in unified memory.

In MLX, rather than moving arrays to devices, you specify the device
when you run the operation. Any device can perform any operation on
<span class="pre">`a`</span> and <span class="pre">`b`</span> without
needing to move them from one memory location to another. For example:

<div class="highlight-python notranslate">

<div class="highlight">

    mx.add(a, b, stream=mx.cpu)
    mx.add(a, b, stream=mx.gpu)

</div>

</div>

In the above, both the CPU and the GPU will perform the same add
operation. The operations can (and likely will) be run in parallel since
there are no dependencies between them. See <a
href="https://ml-explore.github.io/mlx/build/html/usage/using_streams.html#using-streams"
class="reference internal"><span class="std std-ref">Using
Streams</span></a> for more information the semantics of streams in MLX.

In the above <span class="pre">`add`</span> example, there are no
dependencies between operations, so there is no possibility for race
conditions. If there are dependencies, the MLX scheduler will
automatically manage them. For example:

<div class="highlight-python notranslate">

<div class="highlight">

    c = mx.add(a, b, stream=mx.cpu)
    d = mx.add(a, c, stream=mx.gpu)

</div>

</div>

In the above case, the second <span class="pre">`add`</span> runs on the
GPU but it depends on the output of the first
<span class="pre">`add`</span> which is running on the CPU. MLX will
automatically insert a dependency between the two streams so that the
second <span class="pre">`add`</span> only starts executing after the
first is complete and <span class="pre">`c`</span> is available.

<div id="a-simple-example" class="section">

## A Simple Example<a href="https://ml-explore.github.io/mlx/build/html/#a-simple-example"
class="headerlink" title="Link to this heading">#</a>

Here is a more interesting (albeit slightly contrived example) of how
unified memory can be helpful. Suppose we have the following
computation:

<div class="highlight-python notranslate">

<div class="highlight">

    def fun(a, b, d1, d2):
      x = mx.matmul(a, b, stream=d1)
      for _ in range(500):
          b = mx.exp(b, stream=d2)
      return x, b

</div>

</div>

which we want to run with the following arguments:

<div class="highlight-python notranslate">

<div class="highlight">

    a = mx.random.uniform(shape=(4096, 512))
    b = mx.random.uniform(shape=(512, 4))

</div>

</div>

The first <span class="pre">`matmul`</span> operation is a good fit for
the GPU since it’s more compute dense. The second sequence of operations
are a better fit for the CPU, since they are very small and would
probably be overhead bound on the GPU.

If we time the computation fully on the GPU, we get 2.8 milliseconds.
But if we run the computation with <span class="pre">`d1=mx.gpu`</span>
and <span class="pre">`d2=mx.cpu`</span>, then the time is only about
1.4 milliseconds, about twice as fast. These times were measured on an
M1 Max.

</div>

</div>

<div class="prev-next-area">

<a
href="https://ml-explore.github.io/mlx/build/html/usage/lazy_evaluation.html"
class="left-prev" title="previous page"><em></em></a>

<div class="prev-next-info">

previous

Lazy Evaluation

</div>

<a
href="https://ml-explore.github.io/mlx/build/html/usage/indexing.html"
class="right-next" title="next page"></a>

<div class="prev-next-info">

next

Indexing Arrays

</div>

</div>

</div>

<div class="bd-sidebar-secondary bd-toc">

<div class="sidebar-secondary-items sidebar-secondary__inner">

<div class="sidebar-secondary-item">

<div class="page-toc tocsection onthispage">

Contents

</div>

- <a href="https://ml-explore.github.io/mlx/build/html/#a-simple-example"
  class="reference internal nav-link">A Simple Example</a>

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
