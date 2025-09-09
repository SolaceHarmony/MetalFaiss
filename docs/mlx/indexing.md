Curator's note: Prefer the human-authored MLX guides for clarity.
- ../docs_curated/README.md
- ../docs_curated/PYTORCH_DISSONANCE.md
- ../docs_curated/NUMPY_USERS.md
- ../docs_curated/COMMON_PITFALLS.md

<!--
Per-file analysis (indexing.md):
- Solid overview mirroring NumPy indexing; includes slices and Ellipsis.
- MLX-specific helpers worth calling out: mx.take, mx.put_along_axis, slice/slice_update typing.
-->

## Curated Notes

- `mx.take(x, indices, axis=...)` gathers by index along a chosen axis; pair with `mx.put_along_axis` for scatter‑style updates.
- `mx.slice`/`mx.slice_update` take MLX array for `start_indices` and Python list/tuple for `axes`/`slice_size`.
- Non‑contiguous slice views can impact heavy kernels; call `mx.contiguous(view)` when performance matters.
- Device control: indexing ops do not take a `device=` argument; placement follows the current default device or per‑op `stream` where available.

### Examples

```python
import mlx.core as mx

a = mx.arange(12).reshape(3, 4)
# slice_update: write a 2x2 block at (0,1)
start = mx.array([0, 1])
axes = [0, 1]
update = mx.array([[10, 11], [12, 13]])
b = mx.slice_update(a, update, start, axes)

# put_along_axis: scatter values along an axis
x = mx.zeros((3, 4))
idx = mx.array([[1, 3, 0, 2], [0, 1, 2, 3], [3, 2, 1, 0]])
vals = mx.arange(12).reshape(3, 4)
y = mx.put_along_axis(x, idx, vals, axis=1)
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
  href="https://ml-explore.github.io/mlx/build/html/_sources/usage/indexing.rst"
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

# Indexing Arrays

<div id="print-main-content">

<div id="jb-print-toc">

<div>

## Contents

</div>

- <a
  href="https://ml-explore.github.io/mlx/build/html/#differences-from-numpy"
  class="reference internal nav-link">Differences from NumPy</a>
- <a href="https://ml-explore.github.io/mlx/build/html/#in-place-updates"
  class="reference internal nav-link">In Place Updates</a>

</div>

</div>

</div>

<div id="searchbox">

</div>

<div id="indexing-arrays" class="section">

<span id="indexing"></span>

# Indexing Arrays<a href="https://ml-explore.github.io/mlx/build/html/#indexing-arrays"
class="headerlink" title="Link to this heading">#</a>

For the most part, indexing an MLX <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.html#mlx.core.array"
class="reference internal" title="mlx.core.array"><span
class="pre"><code class="sourceCode python">array</code></span></a>
works the same as indexing a NumPy <a
href="https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray"
class="reference external" title="(in NumPy v2.2)"><span
class="pre"><code
class="sourceCode python">numpy.ndarray</code></span></a>. See the
<a href="https://numpy.org/doc/stable/user/basics.indexing.html"
class="reference external">NumPy documentation</a> for more details on
how that works.

For example, you can use regular integers and slices (<a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.slice.html#mlx.core.slice"
class="reference internal" title="mlx.core.slice"><span
class="pre"><code
class="sourceCode python"><span class="bu">slice</span></code></span></a>)
to index arrays:

<div class="highlight-shell notranslate">

<div class="highlight">

    >>> arr = mx.arange(10)
    >>> arr[3]
    array(3, dtype=int32)
    >>> arr[-2]  # negative indexing works
    array(8, dtype=int32)
    >>> arr[2:8:2] # start, stop, stride
    array([2, 4, 6], dtype=int32)

</div>

</div>

For multi-dimensional arrays, the <span class="pre">`...`</span> or
<a href="https://docs.python.org/3/library/constants.html#Ellipsis"
class="reference external" title="(in Python v3.13)"><span
class="pre"><code
class="sourceCode python"><span class="va">Ellipsis</span></code></span></a>
syntax works as in NumPy:

<div class="highlight-shell notranslate">

<div class="highlight">

    >>> arr = mx.arange(8).reshape(2, 2, 2)
    >>> arr[:, :, 0]
    array(3, dtype=int32)
    array([[0, 2],
           [4, 6]], dtype=int32
    >>> arr[..., 0]
    array([[0, 2],
           [4, 6]], dtype=int32

</div>

</div>

You can index with <span class="pre">`None`</span> to create a new axis:

<div class="highlight-shell notranslate">

<div class="highlight">

    >>> arr = mx.arange(8)
    >>> arr.shape
    [8]
    >>> arr[None].shape
    [1, 8]

</div>

</div>

You can also use an <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.html#mlx.core.array"
class="reference internal" title="mlx.core.array"><span
class="pre"><code class="sourceCode python">array</code></span></a> to
index another <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.html#mlx.core.array"
class="reference internal" title="mlx.core.array"><span
class="pre"><code class="sourceCode python">array</code></span></a>:

<div class="highlight-shell notranslate">

<div class="highlight">

    >>> arr = mx.arange(10)
    >>> idx = mx.array([5, 7])
    >>> arr[idx]
    array([5, 7], dtype=int32)

</div>

</div>

Mixing and matching integers, <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.slice.html#mlx.core.slice"
class="reference internal" title="mlx.core.slice"><span
class="pre"><code
class="sourceCode python"><span class="bu">slice</span></code></span></a>,
<span class="pre">`...`</span>, and <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.html#mlx.core.array"
class="reference internal" title="mlx.core.array"><span
class="pre"><code class="sourceCode python">array</code></span></a>
indices works just as in NumPy.

Other functions which may be useful for indexing arrays are <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.take.html#mlx.core.take"
class="reference internal" title="mlx.core.take"><span class="pre"><code
class="sourceCode python">take()</code></span></a> and <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.take_along_axis.html#mlx.core.take_along_axis"
class="reference internal" title="mlx.core.take_along_axis"><span
class="pre"><code
class="sourceCode python">take_along_axis()</code></span></a>.

<div id="differences-from-numpy" class="section">

## Differences from NumPy<a
href="https://ml-explore.github.io/mlx/build/html/#differences-from-numpy"
class="headerlink" title="Link to this heading">#</a>

<div class="admonition note">

Note

MLX indexing is different from NumPy indexing in two important ways:

- Indexing does not perform bounds checking. Indexing out of bounds is
  undefined behavior.

- Boolean mask based indexing is not yet supported.

</div>

The reason for the lack of bounds checking is that exceptions cannot
propagate from the GPU. Performing bounds checking for array indices
before launching the kernel would be extremely inefficient.

Indexing with boolean masks is something that MLX may support in the
future. In general, MLX has limited support for operations for which
output *shapes* are dependent on input *data*. Other examples of these
types of operations which MLX does not yet support include <a
href="https://numpy.org/doc/stable/reference/generated/numpy.nonzero.html#numpy.nonzero"
class="reference external" title="(in NumPy v2.2)"><span
class="pre"><code
class="sourceCode python">numpy.nonzero()</code></span></a> and the
single input version of <a
href="https://numpy.org/doc/stable/reference/generated/numpy.where.html#numpy.where"
class="reference external" title="(in NumPy v2.2)"><span
class="pre"><code
class="sourceCode python">numpy.where()</code></span></a>.

</div>

<div id="in-place-updates" class="section">

## In Place Updates<a href="https://ml-explore.github.io/mlx/build/html/#in-place-updates"
class="headerlink" title="Link to this heading">#</a>

In place updates to indexed arrays are possible in MLX. For example:

<div class="highlight-shell notranslate">

<div class="highlight">

    >>> a = mx.array([1, 2, 3])
    >>> a[2] = 0
    >>> a
    array([1, 2, 0], dtype=int32)

</div>

</div>

Just as in NumPy, in place updates will be reflected in all references
to the same array:

<div class="highlight-shell notranslate">

<div class="highlight">

    >>> a = mx.array([1, 2, 3])
    >>> b = a
    >>> b[2] = 0
    >>> b
    array([1, 2, 0], dtype=int32)
    >>> a
    array([1, 2, 0], dtype=int32)

</div>

</div>

Transformations of functions which use in-place updates are allowed and
work as expected. For example:

<div class="highlight-python notranslate">

<div class="highlight">

    def fun(x, idx):
        x[idx] = 2.0
        return x.sum()

    dfdx = mx.grad(fun)(mx.array([1.0, 2.0, 3.0]), mx.array([1]))
    print(dfdx)  # Prints: array([1, 0, 1], dtype=float32)

</div>

</div>

In the above <span class="pre">`dfdx`</span> will have the correct
gradient, namely zeros at <span class="pre">`idx`</span> and ones
elsewhere.

</div>

</div>

<div class="prev-next-area">

<a
href="https://ml-explore.github.io/mlx/build/html/usage/unified_memory.html"
class="left-prev" title="previous page"><em></em></a>

<div class="prev-next-info">

previous

Unified Memory

</div>

<a
href="https://ml-explore.github.io/mlx/build/html/usage/saving_and_loading.html"
class="right-next" title="next page"></a>

<div class="prev-next-info">

next

Saving and Loading Arrays

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
  href="https://ml-explore.github.io/mlx/build/html/#differences-from-numpy"
  class="reference internal nav-link">Differences from NumPy</a>
- <a href="https://ml-explore.github.io/mlx/build/html/#in-place-updates"
  class="reference internal nav-link">In Place Updates</a>

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
