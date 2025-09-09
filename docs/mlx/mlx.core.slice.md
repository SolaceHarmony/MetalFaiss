Curator's note: Prefer the human-authored MLX guides for clarity.
- ../docs_curated/README.md
- ../docs_curated/PYTORCH_DISSONANCE.md
- ../docs_curated/NUMPY_USERS.md
- ../docs_curated/COMMON_PITFALLS.md

<!--
Per-file analysis (mlx.core.slice.md):
- MLX-specific helper; argument types differ from NumPy’s implicit slicing.
- Hotspot for AI mistakes: wrong types for axes/sizes, expecting device=.
-->

## Curated Notes

- Arg types: `start_indices` is an MLX array; `axes` and `slice_size` are Python list/tuple.
- No `device=` argument: placement follows default device or per‑op `stream` when supported.
- Slices can be non‑contiguous; call `mx.contiguous(view)` when performance matters downstream.

### MLX Slicing vs Python Slicing

- Return value: `a[... ]` returns a view; the original array is unchanged (functional, immutable).
- In‑place assignment: `a[1:3] = ...` does not mutate in MLX. Use `.at[...]` or `mx.slice_update` to create a new array with updates.
- Boolean masks: direct masked assignment is not supported inside `.at`. Use `mx.where` for conditional updates.

### Examples

```python
import mlx.core as mx

a = mx.arange(12).reshape(3, 4)
start = mx.array([0, 1])        # row 0, col 1
axes = [0, 1]
size = [2, 2]
view = mx.slice(a, start, axes, size)   # [[1,2],[5,6]]

# Ensure contiguous when needed
view_c = mx.contiguous(view)

# Route to CPU explicitly, if required
view_cpu = mx.slice(a, start, axes, size, stream=mx.cpu)

# Updates: use .at[...] or mx.slice_update
x = mx.array([1, 2, 3, 4, 5])
x_new = x.at[1:3].set(mx.array([-1, -2]))   # array([1,-1,-2,4,5])

# Conditional (boolean mask) update via where
mask = x > 3
y = mx.where(mask, 99, x)                   # array([1,2,3,99,99])
```

Notes:
- `.at[...]` returns a new array with the update; bind it to a new name.
- `mx.slice_update(a, update, start_indices, axes)` mirrors the explicit helper signature for multi‑dimensional updates.
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
  href="https://ml-explore.github.io/mlx/build/html/_sources/python/_autosummary/mlx.core.slice.rst"
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

# mlx.core.slice

<div id="print-main-content">

<div id="jb-print-toc">

<div>

## Contents

</div>

- <a href="https://ml-explore.github.io/mlx/build/html/#mlx.core.slice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">slice()</code></span></a>

</div>

</div>

</div>

<div id="searchbox">

</div>

<div id="mlx-core-slice" class="section">

# mlx.core.slice<a href="https://ml-explore.github.io/mlx/build/html/#mlx-core-slice"
class="headerlink" title="Link to this heading">#</a>

<span class="sig-name descname"><span class="pre">slice</span></span><span class="sig-paren">(</span>*<span class="n"><span class="pre">a</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.html#mlx.core.array"
class="reference internal" title="mlx.core.array"><span
class="pre">array</span></a></span>*, *<span class="n"><span class="pre">start_indices</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.html#mlx.core.array"
class="reference internal" title="mlx.core.array"><span
class="pre">array</span></a></span>*, *<span class="n"><span class="pre">axes</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><span class="pre">Sequence</span><span class="p"><span class="pre">\[</span></span><a href="https://docs.python.org/3/library/functions.html#int"
class="reference external" title="(in Python v3.13)"><span
class="pre">int</span></a><span class="p"><span class="pre">\]</span></span></span>*, *<span class="n"><span class="pre">slice_size</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><span class="pre">Sequence</span><span class="p"><span class="pre">\[</span></span><a href="https://docs.python.org/3/library/functions.html#int"
class="reference external" title="(in Python v3.13)"><span
class="pre">int</span></a><span class="p"><span class="pre">\]</span></span></span>*, *<span class="o"><span class="pre">\*</span></span>*, *<span class="n"><span class="pre">stream</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><a href="https://docs.python.org/3/library/constants.html#None"
class="reference external" title="(in Python v3.13)"><span
class="pre">None</span></a><span class="w"> </span><span class="p"><span class="pre">\|</span></span><span class="w"> </span><a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/stream_class.html#mlx.core.Stream"
class="reference internal" title="mlx.core.Stream"><span
class="pre">Stream</span></a><span class="w"> </span><span class="p"><span class="pre">\|</span></span><span class="w"> </span><a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.Device.html#mlx.core.Device"
class="reference internal" title="mlx.core.Device"><span
class="pre">Device</span></a></span><span class="w"> </span><span class="o"><span class="pre">=</span></span><span class="w"> </span><span class="default_value"><span class="pre">None</span></span>*<span class="sig-paren">)</span> <span class="sig-return"><span class="sig-return-icon">→</span> <span class="sig-return-typehint"><a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.html#mlx.core.array"
class="reference internal" title="mlx.core.array"><span
class="pre">array</span></a></span></span><a href="https://ml-explore.github.io/mlx/build/html/#mlx.core.slice"
class="headerlink" title="Link to this definition">#</a>  
Extract a sub-array from the input array.

Parameters<span class="colon">:</span>  
- **a** (<a
  href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.html#mlx.core.array"
  class="reference internal" title="mlx.core.array"><em>array</em></a>)
  – Input array

- **start_indices** (<a
  href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.html#mlx.core.array"
  class="reference internal" title="mlx.core.array"><em>array</em></a>)
  – The index location to start the slice at.

- **axes**
  (<a href="https://docs.python.org/3/library/stdtypes.html#tuple"
  class="reference external" title="(in Python v3.13)"><em>tuple</em></a>*(*<a href="https://docs.python.org/3/library/functions.html#int"
  class="reference external" title="(in Python v3.13)"><em>int</em></a>*)*)
  – The axes corresponding to the indices in
  <span class="pre">`start_indices`</span>.

- **slice_size**
  (<a href="https://docs.python.org/3/library/stdtypes.html#tuple"
  class="reference external" title="(in Python v3.13)"><em>tuple</em></a>*(*<a href="https://docs.python.org/3/library/functions.html#int"
  class="reference external" title="(in Python v3.13)"><em>int</em></a>*)*)
  – The size of the slice.

Returns<span class="colon">:</span>  
The sliced output array.

Return type<span class="colon">:</span>  
<a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.html#mlx.core.array"
class="reference internal" title="mlx.core.array"><em>array</em></a>

Example

<div class="doctest highlight-default notranslate">

<div class="highlight">

    >>> a = mx.array([[1, 2, 3], [4, 5, 6]])
    >>> mx.slice(a, start_indices=mx.array(1), axes=(0,), slice_size=(1, 2))
    array([[4, 5]], dtype=int32)
    >>>
    >>> mx.slice(a, start_indices=mx.array(1), axes=(1,), slice_size=(2, 1))
    array([[2],
           [5]], dtype=int32)

</div>

</div>

</div>

<div class="prev-next-area">

<a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.sinh.html"
class="left-prev" title="previous page"><em></em></a>

<div class="prev-next-info">

previous

mlx.core.sinh

</div>

<a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.slice_update.html"
class="right-next" title="next page"></a>

<div class="prev-next-info">

next

mlx.core.slice_update

</div>

</div>

</div>

<div class="bd-sidebar-secondary bd-toc">

<div class="sidebar-secondary-items sidebar-secondary__inner">

<div class="sidebar-secondary-item">

<div class="page-toc tocsection onthispage">

Contents

</div>

- <a href="https://ml-explore.github.io/mlx/build/html/#mlx.core.slice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">slice()</code></span></a>

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
