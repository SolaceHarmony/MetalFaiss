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
  href="https://ml-explore.github.io/mlx/build/html/_sources/usage/numpy.rst"
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

# Conversion to NumPy and Other Frameworks

<div id="print-main-content">

<div id="jb-print-toc">

<div>

## Contents

</div>

- <a href="https://ml-explore.github.io/mlx/build/html/#pytorch"
  class="reference internal nav-link">PyTorch</a>
- <a href="https://ml-explore.github.io/mlx/build/html/#jax"
  class="reference internal nav-link">JAX</a>
- <a href="https://ml-explore.github.io/mlx/build/html/#tensorflow"
  class="reference internal nav-link">TensorFlow</a>

</div>

</div>

</div>

<div id="searchbox">

</div>

<div id="conversion-to-numpy-and-other-frameworks" class="section">

<span id="numpy"></span>

# Conversion to NumPy and Other Frameworks<a
href="https://ml-explore.github.io/mlx/build/html/#conversion-to-numpy-and-other-frameworks"
class="headerlink" title="Link to this heading">#</a>

MLX array supports conversion between other frameworks with either:

- The <a href="https://docs.python.org/3/c-api/buffer.html"
  class="reference external">Python Buffer Protocol</a>.

- <a href="https://dmlc.github.io/dlpack/latest/"
  class="reference external">DLPack</a>.

Let’s convert an array to NumPy and back.

<div class="highlight-python notranslate">

<div class="highlight">

    import mlx.core as mx
    import numpy as np

    a = mx.arange(3)
    b = np.array(a) # copy of a
    c = mx.array(b) # copy of b

</div>

</div>

<div class="admonition note">

Note

Since NumPy does not support <span class="pre">`bfloat16`</span> arrays,
you will need to convert to <span class="pre">`float16`</span> or
<span class="pre">`float32`</span> first:
<span class="pre">`np.array(a.astype(mx.float32))`</span>. Otherwise,
you will receive an error like:
<span class="pre">`Item`</span>` `<span class="pre">`size`</span>` `<span class="pre">`2`</span>` `<span class="pre">`for`</span>` `<span class="pre">`PEP`</span>` `<span class="pre">`3118`</span>` `<span class="pre">`buffer`</span>` `<span class="pre">`format`</span>` `<span class="pre">`string`</span>` `<span class="pre">`does`</span>` `<span class="pre">`not`</span>` `<span class="pre">`match`</span>` `<span class="pre">`the`</span>` `<span class="pre">`dtype`</span>` `<span class="pre">`V`</span>` `<span class="pre">`item`</span>` `<span class="pre">`size`</span>` `<span class="pre">`0.`</span>

</div>

By default, NumPy copies data to a new array. This can be prevented by
creating an array view:

<div class="highlight-python notranslate">

<div class="highlight">

    a = mx.arange(3)
    a_view = np.array(a, copy=False)
    print(a_view.flags.owndata) # False
    a_view[0] = 1
    print(a[0].item()) # 1

</div>

</div>

<div class="admonition note">

Note

NumPy arrays with type <span class="pre">`float64`</span> will be
default converted to MLX arrays with type
<span class="pre">`float32`</span>.

</div>

A NumPy array view is a normal NumPy array, except that it does not own
its memory. This means writing to the view is reflected in the original
array.

While this is quite powerful to prevent copying arrays, it should be
noted that external changes to the memory of arrays cannot be reflected
in gradients.

Let’s demonstrate this in an example:

<div class="highlight-python notranslate">

<div class="highlight">

    def f(x):
        x_view = np.array(x, copy=False)
        x_view[:] *= x_view # modify memory without telling mx
        return x.sum()

    x = mx.array([3.0])
    y, df = mx.value_and_grad(f)(x)
    print("f(x) = x² =", y.item()) # 9.0
    print("f'(x) = 2x !=", df.item()) # 1.0

</div>

</div>

The function <span class="pre">`f`</span> indirectly modifies the array
<span class="pre">`x`</span> through a memory view. However, this
modification is not reflected in the gradient, as seen in the last line
outputting <span class="pre">`1.0`</span>, representing the gradient of
the sum operation alone. The squaring of <span class="pre">`x`</span>
occurs externally to MLX, meaning that no gradient is incorporated. It’s
important to note that a similar issue arises during array conversion
and copying. For instance, a function defined as
<span class="pre">`mx.array(np.array(x)**2).sum()`</span> would also
result in an incorrect gradient, even though no in-place operations on
MLX memory are executed.

<div id="pytorch" class="section">

## PyTorch<a href="https://ml-explore.github.io/mlx/build/html/#pytorch"
class="headerlink" title="Link to this heading">#</a>

<div class="admonition warning">

Warning

PyTorch Support for
<a href="https://docs.python.org/3/library/stdtypes.html#memoryview"
class="reference external" title="(in Python v3.13)"><span
class="pre"><code
class="sourceCode python"><span class="bu">memoryview</span></code></span></a>
is experimental and can break for multi-dimensional arrays. Casting to
NumPy first is advised for now.

</div>

PyTorch supports the buffer protocol, but it requires an explicit
<a href="https://docs.python.org/3/library/stdtypes.html#memoryview"
class="reference external" title="(in Python v3.13)"><span
class="pre"><code
class="sourceCode python"><span class="bu">memoryview</span></code></span></a>.

<div class="highlight-python notranslate">

<div class="highlight">

    import mlx.core as mx
    import torch

    a = mx.arange(3)
    b = torch.tensor(memoryview(a))
    c = mx.array(b.numpy())

</div>

</div>

Conversion from PyTorch tensors back to arrays must be done via
intermediate NumPy arrays with <span class="pre">`numpy()`</span>.

</div>

<div id="jax" class="section">

## JAX<a href="https://ml-explore.github.io/mlx/build/html/#jax"
class="headerlink" title="Link to this heading">#</a>

JAX fully supports the buffer protocol.

<div class="highlight-python notranslate">

<div class="highlight">

    import mlx.core as mx
    import jax.numpy as jnp

    a = mx.arange(3)
    b = jnp.array(a)
    c = mx.array(b)

</div>

</div>

</div>

<div id="tensorflow" class="section">

## TensorFlow<a href="https://ml-explore.github.io/mlx/build/html/#tensorflow"
class="headerlink" title="Link to this heading">#</a>

TensorFlow supports the buffer protocol, but it requires an explicit
<a href="https://docs.python.org/3/library/stdtypes.html#memoryview"
class="reference external" title="(in Python v3.13)"><span
class="pre"><code
class="sourceCode python"><span class="bu">memoryview</span></code></span></a>.

<div class="highlight-python notranslate">

<div class="highlight">

    import mlx.core as mx
    import tensorflow as tf

    a = mx.arange(3)
    b = tf.constant(memoryview(a))
    c = mx.array(b)

</div>

</div>

</div>

</div>

<div class="prev-next-area">

<a href="https://ml-explore.github.io/mlx/build/html/usage/compile.html"
class="left-prev" title="previous page"><em></em></a>

<div class="prev-next-info">

previous

Compilation

</div>

<a
href="https://ml-explore.github.io/mlx/build/html/usage/distributed.html"
class="right-next" title="next page"></a>

<div class="prev-next-info">

next

Distributed Communication

</div>

</div>

</div>

<div class="bd-sidebar-secondary bd-toc">

<div class="sidebar-secondary-items sidebar-secondary__inner">

<div class="sidebar-secondary-item">

<div class="page-toc tocsection onthispage">

Contents

</div>

- <a href="https://ml-explore.github.io/mlx/build/html/#pytorch"
  class="reference internal nav-link">PyTorch</a>
- <a href="https://ml-explore.github.io/mlx/build/html/#jax"
  class="reference internal nav-link">JAX</a>
- <a href="https://ml-explore.github.io/mlx/build/html/#tensorflow"
  class="reference internal nav-link">TensorFlow</a>

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
