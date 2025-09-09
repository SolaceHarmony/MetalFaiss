Curator's note: Prefer the human-authored MLX guides for clarity.
- ../docs_curated/README.md
- ../docs_curated/PYTORCH_DISSONANCE.md
- ../docs_curated/NUMPY_USERS.md
- ../docs_curated/COMMON_PITFALLS.md

<!--
Per-file analysis (index.md):
- Landing page (heavy Sphinx markup) describing MLX pillars.
- Best addition: a curated on-ramp with links to practical guides in this repo.
-->

## Curated On‑Ramp

- New to MLX? Start here: `../docs_curated/GETTING_STARTED.md`
- Arrays and shapes: `../docs_curated/ARRAYS.md`, `../docs_curated/SHAPES_AND_BROADCASTING.md`
- Training and optimizers: `../docs_curated/NN.md`, `../docs_curated/OPTIMIZERS.md`, `../docs_curated/OPTIMIZERS_DEEP_DIVE.md`
- PyTorch/NumPy users: `../docs_curated/PORTING_FROM_PYTORCH.md`, `../docs_curated/NUMPY_USERS.md`, `../docs_curated/PYTORCH_DISSONANCE.md`


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

- <a href="https://ml-explore.github.io/mlx/build/html/_sources/index.rst"
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

# MLX

<div id="print-main-content">

<div id="jb-print-toc">

</div>

</div>

</div>

<div id="searchbox">

</div>

<div id="mlx" class="section">

# MLX<a href="https://ml-explore.github.io/mlx/build/html/#mlx"
class="headerlink" title="Link to this heading">#</a>

MLX is a NumPy-like array framework designed for efficient and flexible
machine learning on Apple silicon, brought to you by Apple machine
learning research.

The Python API closely follows NumPy with a few exceptions. MLX also has
a fully featured C++ API which closely follows the Python API.

The main differences between MLX and NumPy are:

> <div>
>
> - **Composable function transformations**: MLX has composable function
>   transformations for automatic differentiation, automatic
>   vectorization, and computation graph optimization.
>
> - **Lazy computation**: Computations in MLX are lazy. Arrays are only
>   materialized when needed.
>
> - **Multi-device**: Operations can run on any of the supported devices
>   (CPU, GPU, …)
>
> </div>

The design of MLX is inspired by frameworks like
<a href="https://pytorch.org/" class="reference external">PyTorch</a>,
<a href="https://github.com/google/jax"
class="reference external">Jax</a>, and <a href="https://arrayfire.org/"
class="reference external">ArrayFire</a>. A notable difference from
these frameworks and MLX is the *unified memory model*. Arrays in MLX
live in shared memory. Operations on MLX arrays can be performed on any
of the supported device types without performing data copies. Currently
supported device types are the CPU and GPU.

<div class="toctree-wrapper compound">

<span class="caption-text">Install</span>

- <a href="https://ml-explore.github.io/mlx/build/html/install.html"
  class="reference internal">Build and Install</a>

</div>

<div class="toctree-wrapper compound">

<span class="caption-text">Usage</span>

- <a
  href="https://ml-explore.github.io/mlx/build/html/usage/quick_start.html"
  class="reference internal">Quick Start Guide</a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/usage/lazy_evaluation.html"
  class="reference internal">Lazy Evaluation</a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/usage/unified_memory.html"
  class="reference internal">Unified Memory</a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/usage/indexing.html"
  class="reference internal">Indexing Arrays</a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/usage/saving_and_loading.html"
  class="reference internal">Saving and Loading Arrays</a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/usage/function_transforms.html"
  class="reference internal">Function Transforms</a>
- <a href="https://ml-explore.github.io/mlx/build/html/usage/compile.html"
  class="reference internal">Compilation</a>
- <a href="https://ml-explore.github.io/mlx/build/html/usage/numpy.html"
  class="reference internal">Conversion to NumPy and Other Frameworks</a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/usage/distributed.html"
  class="reference internal">Distributed Communication</a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/usage/using_streams.html"
  class="reference internal">Using Streams</a>
- <a href="https://ml-explore.github.io/mlx/build/html/usage/export.html"
  class="reference internal">Exporting Functions</a>

</div>

<div class="toctree-wrapper compound">

<span class="caption-text">Examples</span>

- <a
  href="https://ml-explore.github.io/mlx/build/html/examples/linear_regression.html"
  class="reference internal">Linear Regression</a>
- <a href="https://ml-explore.github.io/mlx/build/html/examples/mlp.html"
  class="reference internal">Multi-Layer Perceptron</a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/examples/llama-inference.html"
  class="reference internal">LLM inference</a>

</div>

<div class="toctree-wrapper compound">

<span class="caption-text">Python API Reference</span>

- <a href="https://ml-explore.github.io/mlx/build/html/python/array.html"
  class="reference internal">Array</a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/python/data_types.html"
  class="reference internal">Data Types</a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/python/devices_and_streams.html"
  class="reference internal">Devices and Streams</a>
- <a href="https://ml-explore.github.io/mlx/build/html/python/export.html"
  class="reference internal">Export Functions</a>
- <a href="https://ml-explore.github.io/mlx/build/html/python/ops.html"
  class="reference internal">Operations</a>
- <a href="https://ml-explore.github.io/mlx/build/html/python/random.html"
  class="reference internal">Random</a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/python/transforms.html"
  class="reference internal">Transforms</a>
- <a href="https://ml-explore.github.io/mlx/build/html/python/fast.html"
  class="reference internal">Fast</a>
- <a href="https://ml-explore.github.io/mlx/build/html/python/fft.html"
  class="reference internal">FFT</a>
- <a href="https://ml-explore.github.io/mlx/build/html/python/linalg.html"
  class="reference internal">Linear Algebra</a>
- <a href="https://ml-explore.github.io/mlx/build/html/python/metal.html"
  class="reference internal">Metal</a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/python/memory_management.html"
  class="reference internal">Memory Management</a>
- <a href="https://ml-explore.github.io/mlx/build/html/python/nn.html"
  class="reference internal">Neural Networks</a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/python/optimizers.html"
  class="reference internal">Optimizers</a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/python/distributed.html"
  class="reference internal">Distributed Communication</a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/python/tree_utils.html"
  class="reference internal">Tree Utils</a>

</div>

<div class="toctree-wrapper compound">

<span class="caption-text">C++ API Reference</span>

- <a href="https://ml-explore.github.io/mlx/build/html/cpp/ops.html"
  class="reference internal">Operations</a>

</div>

<div class="toctree-wrapper compound">

<span class="caption-text">Further Reading</span>

- <a
  href="https://ml-explore.github.io/mlx/build/html/dev/extensions.html"
  class="reference internal">Custom Extensions in MLX</a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/dev/metal_debugger.html"
  class="reference internal">Metal Debugger</a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html"
  class="reference internal">Custom Metal Kernels</a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/dev/mlx_in_cpp.html"
  class="reference internal">Using MLX in C++</a>

</div>

</div>

<div class="prev-next-area">

<a href="https://ml-explore.github.io/mlx/build/html/install.html"
class="right-next" title="next page"></a>

<div class="prev-next-info">

next

Build and Install

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
