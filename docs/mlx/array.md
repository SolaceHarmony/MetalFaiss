Curator's note: Prefer the human-authored MLX guides for clarity.
- ../docs_curated/README.md
- ../docs_curated/PYTORCH_DISSONANCE.md
- ../docs_curated/NUMPY_USERS.md
- ../docs_curated/COMMON_PITFALLS.md

<!--
Per-file analysis (array.md):
- This page is the Sphinx-converted index for mlx.core.array class and its instance methods.
- It contains significant theme markup (pydata-sphinx HTML wrappers) and duplicated "Array" headings.
- Value add here: a concise Quick Start for arrays, plus NumPy-minded tips specific to array scalars and dtypes.
-->

# Quick Start (curated)

```python
import mlx.core as mx

a = mx.array([1, 2, 3], dtype=mx.int32)
b = mx.ones((3,), dtype=mx.float32)
print((a + b).astype(mx.float16))

# Scalar extraction (host Python number)
s = mx.sum(b)
print(s.item())  # prefer .item() for 0-D arrays
```

Tips:
- Operators or functions both work: `a + b` or `mx.add(a, b)`; `a @ c` or `mx.matmul(a, c)`.
- For NumPy users: see ../docs_curated/NUMPY_USERS.md for `mx.slice`/`mx.slice_update` argument types.

## Copying and Immutability

- MLX arrays are immutable; there is no `.clone()` or in‑place mutation. Any "update" returns a new array.
- To copy an array, create a new one: `y = mx.array(x)` or `y = mx.copy(x)`.

```python
import mlx.core as mx
orig = mx.array([1, 2, 3, 4])
cpy1 = mx.array(orig)
cpy2 = mx.copy(orig)
assert mx.allclose(orig, cpy1) and mx.allclose(orig, cpy2)
```

## Device and Dtype Guidance (important)

- No `device=` argument: MLX does not take a `device=` keyword on most ops. Select placement via default device/streams:
  - Global: `mx.set_default_device(mx.cpu)` (or `mx.gpu`)
  - Scoped: `with mx.default_device(mx.cpu): ...`
  - Per‑op (when supported): `op(..., stream=mx.cpu)`
- Float64 on CPU only: double‑precision (`mx.float64`) is not supported on GPU; attempting to run float64 on GPU raises an error.
  - Use CPU for double‑precision work: `mx.set_default_device(mx.cpu)` or `stream=mx.cpu`.
- Some constructors don’t accept `dtype=`; the robust pattern is: construct, then cast with `astype(...)`.

### Examples

```python
import mlx.core as mx

# Robust dtype pattern: construct then cast
a = mx.array([1, 2, 3]).astype(mx.float32)
print(a.dtype)

# Route double‑precision to CPU
mx.set_default_device(mx.cpu)
dd = mx.ones((8,), dtype=mx.float64)
s = mx.sum(dd)  # ok on CPU

# Per‑op CPU routing without changing global default
mx.set_default_device(mx.gpu)  # if available
b = mx.arange(10)
s64 = mx.sum(b.astype(mx.float64), stream=mx.cpu)
```

## Slicing and Updates (Important)

- Slicing returns a view; arrays are immutable. `x[1:3] = ...` does not mutate `x`.
- Use `array.at[...]` or `mx.slice_update` to produce an updated copy.
- Boolean masked updates: use `mx.where` to build a new array conditionally.

```python
import mlx.core as mx

x = mx.array([1, 2, 3, 4, 5])

# Wrong mental model (no effect on x)
_ = x[1:3]

# Correct: create a new array with updates
x2 = x.at[1:3].set(mx.array([-1, -2]))     # array([1,-1,-2,4,5])

# Explicit helper form
x3 = mx.slice_update(x, mx.array([-1, -2]), start_indices=mx.array([1]), axes=[0])

# Boolean mask via where
mask = x > 3
x4 = mx.where(mask, 99, x)                 # array([1,2,3,99,99])
```

## Performance: `mx.where` vs Python Loops

- Prefer `mx.where` for elementwise conditionals: it is vectorized, lazy, and fuses well; Python `for` + `if/else` is interpreted and slow.
- On GPU, `mx.where` avoids divergent branches by evaluating both sides and selecting, which is usually faster than branching.
- See `mlx.core.where` for deeper notes and examples.

```python
# Slow: Python loop (don’t do this)
out = [xi if (xi % 2 == 0) else -xi for xi in range(10)]

# Fast: vectorized
import mlx.core as mx
x = mx.arange(10)
y = mx.where((x % 2) == 0, x, -x)
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
  href="https://ml-explore.github.io/mlx/build/html/_sources/python/array.rst"
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

# Array

<div id="print-main-content">

<div id="jb-print-toc">

</div>

</div>

</div>

<div id="searchbox">

</div>

<div id="array" class="section">

<span id="id1"></span>

# Array<a href="https://ml-explore.github.io/mlx/build/html/#array"
class="headerlink" title="Link to this heading">#</a>

<div class="pst-scrollable-table-container">

|  |  |
|----|----|
| <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.html#mlx.core.array"
class="reference internal" title="mlx.core.array"><span
class="pre"><code class="sourceCode python">array</code></span></a> | An N-dimensional array object. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.astype.html#mlx.core.array.astype"
class="reference internal" title="mlx.core.array.astype"><span
class="pre"><code
class="sourceCode python">array.astype</code></span></a>(self, dtype\[, stream\]) | Cast the array to a specified type. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.at.html#mlx.core.array.at"
class="reference internal" title="mlx.core.array.at"><span
class="pre"><code class="sourceCode python">array.at</code></span></a> | Used to apply updates at the given indices. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.item.html#mlx.core.array.item"
class="reference internal" title="mlx.core.array.item"><span
class="pre"><code class="sourceCode python">array.item</code></span></a>(self) | Access the value of a scalar array. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.tolist.html#mlx.core.array.tolist"
class="reference internal" title="mlx.core.array.tolist"><span
class="pre"><code
class="sourceCode python">array.tolist</code></span></a>(self) | Convert the array to a Python <a href="https://docs.python.org/3/library/stdtypes.html#list"
class="reference external" title="(in Python v3.13)"><span
class="pre"><code
class="sourceCode python"><span class="bu">list</span></code></span></a>. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.dtype.html#mlx.core.array.dtype"
class="reference internal" title="mlx.core.array.dtype"><span
class="pre"><code
class="sourceCode python">array.dtype</code></span></a> | The array's <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.Dtype.html#mlx.core.Dtype"
class="reference internal" title="mlx.core.Dtype"><span
class="pre"><code class="sourceCode python">Dtype</code></span></a>. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.itemsize.html#mlx.core.array.itemsize"
class="reference internal" title="mlx.core.array.itemsize"><span
class="pre"><code
class="sourceCode python">array.itemsize</code></span></a> | The size of the array's datatype in bytes. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.nbytes.html#mlx.core.array.nbytes"
class="reference internal" title="mlx.core.array.nbytes"><span
class="pre"><code
class="sourceCode python">array.nbytes</code></span></a> | The number of bytes in the array. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.ndim.html#mlx.core.array.ndim"
class="reference internal" title="mlx.core.array.ndim"><span
class="pre"><code class="sourceCode python">array.ndim</code></span></a> | The array's dimension. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.shape.html#mlx.core.array.shape"
class="reference internal" title="mlx.core.array.shape"><span
class="pre"><code
class="sourceCode python">array.shape</code></span></a> | The shape of the array as a Python tuple. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.size.html#mlx.core.array.size"
class="reference internal" title="mlx.core.array.size"><span
class="pre"><code class="sourceCode python">array.size</code></span></a> | Number of elements in the array. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.abs.html#mlx.core.array.abs"
class="reference internal" title="mlx.core.array.abs"><span
class="pre"><code
class="sourceCode python">array.<span class="bu">abs</span></code></span></a>(self, \*\[, stream\]) | See <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.abs.html#mlx.core.abs"
class="reference internal" title="mlx.core.abs"><span class="pre"><code
class="sourceCode python"><span class="bu">abs</span>()</code></span></a>. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.all.html#mlx.core.array.all"
class="reference internal" title="mlx.core.array.all"><span
class="pre"><code
class="sourceCode python">array.<span class="bu">all</span></code></span></a>(self\[, axis, keepdims, stream\]) | See <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.all.html#mlx.core.all"
class="reference internal" title="mlx.core.all"><span class="pre"><code
class="sourceCode python"><span class="bu">all</span>()</code></span></a>. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.any.html#mlx.core.array.any"
class="reference internal" title="mlx.core.array.any"><span
class="pre"><code
class="sourceCode python">array.<span class="bu">any</span></code></span></a>(self\[, axis, keepdims, stream\]) | See <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.any.html#mlx.core.any"
class="reference internal" title="mlx.core.any"><span class="pre"><code
class="sourceCode python"><span class="bu">any</span>()</code></span></a>. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.argmax.html#mlx.core.array.argmax"
class="reference internal" title="mlx.core.array.argmax"><span
class="pre"><code
class="sourceCode python">array.argmax</code></span></a>(self\[, axis, keepdims, stream\]) | See <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.argmax.html#mlx.core.argmax"
class="reference internal" title="mlx.core.argmax"><span
class="pre"><code class="sourceCode python">argmax()</code></span></a>. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.argmin.html#mlx.core.array.argmin"
class="reference internal" title="mlx.core.array.argmin"><span
class="pre"><code
class="sourceCode python">array.argmin</code></span></a>(self\[, axis, keepdims, stream\]) | See <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.argmin.html#mlx.core.argmin"
class="reference internal" title="mlx.core.argmin"><span
class="pre"><code class="sourceCode python">argmin()</code></span></a>. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.conj.html#mlx.core.array.conj"
class="reference internal" title="mlx.core.array.conj"><span
class="pre"><code class="sourceCode python">array.conj</code></span></a>(self, \*\[, stream\]) | See <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.conj.html#mlx.core.conj"
class="reference internal" title="mlx.core.conj"><span class="pre"><code
class="sourceCode python">conj()</code></span></a>. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.cos.html#mlx.core.array.cos"
class="reference internal" title="mlx.core.array.cos"><span
class="pre"><code class="sourceCode python">array.cos</code></span></a>(self, \*\[, stream\]) | See <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.cos.html#mlx.core.cos"
class="reference internal" title="mlx.core.cos"><span class="pre"><code
class="sourceCode python">cos()</code></span></a>. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.cummax.html#mlx.core.array.cummax"
class="reference internal" title="mlx.core.array.cummax"><span
class="pre"><code
class="sourceCode python">array.cummax</code></span></a>(self\[, axis, reverse, ...\]) | See <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.cummax.html#mlx.core.cummax"
class="reference internal" title="mlx.core.cummax"><span
class="pre"><code class="sourceCode python">cummax()</code></span></a>. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.cummin.html#mlx.core.array.cummin"
class="reference internal" title="mlx.core.array.cummin"><span
class="pre"><code
class="sourceCode python">array.cummin</code></span></a>(self\[, axis, reverse, ...\]) | See <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.cummin.html#mlx.core.cummin"
class="reference internal" title="mlx.core.cummin"><span
class="pre"><code class="sourceCode python">cummin()</code></span></a>. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.cumprod.html#mlx.core.array.cumprod"
class="reference internal" title="mlx.core.array.cumprod"><span
class="pre"><code
class="sourceCode python">array.cumprod</code></span></a>(self\[, axis, reverse, ...\]) | See <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.cumprod.html#mlx.core.cumprod"
class="reference internal" title="mlx.core.cumprod"><span
class="pre"><code class="sourceCode python">cumprod()</code></span></a>. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.cumsum.html#mlx.core.array.cumsum"
class="reference internal" title="mlx.core.array.cumsum"><span
class="pre"><code
class="sourceCode python">array.cumsum</code></span></a>(self\[, axis, reverse, ...\]) | See <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.cumsum.html#mlx.core.cumsum"
class="reference internal" title="mlx.core.cumsum"><span
class="pre"><code class="sourceCode python">cumsum()</code></span></a>. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.diag.html#mlx.core.array.diag"
class="reference internal" title="mlx.core.array.diag"><span
class="pre"><code class="sourceCode python">array.diag</code></span></a>(self\[, k, stream\]) | Extract a diagonal or construct a diagonal matrix. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.diagonal.html#mlx.core.array.diagonal"
class="reference internal" title="mlx.core.array.diagonal"><span
class="pre"><code
class="sourceCode python">array.diagonal</code></span></a>(self\[, offset, axis1, axis2, ...\]) | See <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.diagonal.html#mlx.core.diagonal"
class="reference internal" title="mlx.core.diagonal"><span
class="pre"><code class="sourceCode python">diagonal()</code></span></a>. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.exp.html#mlx.core.array.exp"
class="reference internal" title="mlx.core.array.exp"><span
class="pre"><code class="sourceCode python">array.exp</code></span></a>(self, \*\[, stream\]) | See <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.exp.html#mlx.core.exp"
class="reference internal" title="mlx.core.exp"><span class="pre"><code
class="sourceCode python">exp()</code></span></a>. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.flatten.html#mlx.core.array.flatten"
class="reference internal" title="mlx.core.array.flatten"><span
class="pre"><code
class="sourceCode python">array.flatten</code></span></a>(self\[, start_axis, end_axis, ...\]) | See <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.flatten.html#mlx.core.flatten"
class="reference internal" title="mlx.core.flatten"><span
class="pre"><code class="sourceCode python">flatten()</code></span></a>. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.log.html#mlx.core.array.log"
class="reference internal" title="mlx.core.array.log"><span
class="pre"><code class="sourceCode python">array.log</code></span></a>(self, \*\[, stream\]) | See <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.log.html#mlx.core.log"
class="reference internal" title="mlx.core.log"><span class="pre"><code
class="sourceCode python">log()</code></span></a>. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.log10.html#mlx.core.array.log10"
class="reference internal" title="mlx.core.array.log10"><span
class="pre"><code
class="sourceCode python">array.log10</code></span></a>(self, \*\[, stream\]) | See <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.log10.html#mlx.core.log10"
class="reference internal" title="mlx.core.log10"><span
class="pre"><code class="sourceCode python">log10()</code></span></a>. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.log1p.html#mlx.core.array.log1p"
class="reference internal" title="mlx.core.array.log1p"><span
class="pre"><code
class="sourceCode python">array.log1p</code></span></a>(self, \*\[, stream\]) | See <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.log1p.html#mlx.core.log1p"
class="reference internal" title="mlx.core.log1p"><span
class="pre"><code class="sourceCode python">log1p()</code></span></a>. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.log2.html#mlx.core.array.log2"
class="reference internal" title="mlx.core.array.log2"><span
class="pre"><code class="sourceCode python">array.log2</code></span></a>(self, \*\[, stream\]) | See <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.log2.html#mlx.core.log2"
class="reference internal" title="mlx.core.log2"><span class="pre"><code
class="sourceCode python">log2()</code></span></a>. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.logcumsumexp.html#mlx.core.array.logcumsumexp"
class="reference internal" title="mlx.core.array.logcumsumexp"><span
class="pre"><code
class="sourceCode python">array.logcumsumexp</code></span></a>(self\[, axis, reverse, ...\]) | See <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.logcumsumexp.html#mlx.core.logcumsumexp"
class="reference internal" title="mlx.core.logcumsumexp"><span
class="pre"><code
class="sourceCode python">logcumsumexp()</code></span></a>. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.logsumexp.html#mlx.core.array.logsumexp"
class="reference internal" title="mlx.core.array.logsumexp"><span
class="pre"><code
class="sourceCode python">array.logsumexp</code></span></a>(self\[, axis, keepdims, stream\]) | See <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.logsumexp.html#mlx.core.logsumexp"
class="reference internal" title="mlx.core.logsumexp"><span
class="pre"><code
class="sourceCode python">logsumexp()</code></span></a>. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.max.html#mlx.core.array.max"
class="reference internal" title="mlx.core.array.max"><span
class="pre"><code
class="sourceCode python">array.<span class="bu">max</span></code></span></a>(self\[, axis, keepdims, stream\]) | See <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.max.html#mlx.core.max"
class="reference internal" title="mlx.core.max"><span class="pre"><code
class="sourceCode python"><span class="bu">max</span>()</code></span></a>. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.mean.html#mlx.core.array.mean"
class="reference internal" title="mlx.core.array.mean"><span
class="pre"><code class="sourceCode python">array.mean</code></span></a>(self\[, axis, keepdims, stream\]) | See <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.mean.html#mlx.core.mean"
class="reference internal" title="mlx.core.mean"><span class="pre"><code
class="sourceCode python">mean()</code></span></a>. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.min.html#mlx.core.array.min"
class="reference internal" title="mlx.core.array.min"><span
class="pre"><code
class="sourceCode python">array.<span class="bu">min</span></code></span></a>(self\[, axis, keepdims, stream\]) | See <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.min.html#mlx.core.min"
class="reference internal" title="mlx.core.min"><span class="pre"><code
class="sourceCode python"><span class="bu">min</span>()</code></span></a>. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.moveaxis.html#mlx.core.array.moveaxis"
class="reference internal" title="mlx.core.array.moveaxis"><span
class="pre"><code
class="sourceCode python">array.moveaxis</code></span></a>(self, source, destination, \*) | See <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.moveaxis.html#mlx.core.moveaxis"
class="reference internal" title="mlx.core.moveaxis"><span
class="pre"><code class="sourceCode python">moveaxis()</code></span></a>. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.prod.html#mlx.core.array.prod"
class="reference internal" title="mlx.core.array.prod"><span
class="pre"><code class="sourceCode python">array.prod</code></span></a>(self\[, axis, keepdims, stream\]) | See <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.prod.html#mlx.core.prod"
class="reference internal" title="mlx.core.prod"><span class="pre"><code
class="sourceCode python">prod()</code></span></a>. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.reciprocal.html#mlx.core.array.reciprocal"
class="reference internal" title="mlx.core.array.reciprocal"><span
class="pre"><code
class="sourceCode python">array.reciprocal</code></span></a>(self, \*\[, stream\]) | See <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.reciprocal.html#mlx.core.reciprocal"
class="reference internal" title="mlx.core.reciprocal"><span
class="pre"><code
class="sourceCode python">reciprocal()</code></span></a>. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.reshape.html#mlx.core.array.reshape"
class="reference internal" title="mlx.core.array.reshape"><span
class="pre"><code
class="sourceCode python">array.reshape</code></span></a>(self, \*shape\[, stream\]) | Equivalent to <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.reshape.html#mlx.core.reshape"
class="reference internal" title="mlx.core.reshape"><span
class="pre"><code class="sourceCode python">reshape()</code></span></a> but the shape can be passed either as a <a href="https://docs.python.org/3/library/stdtypes.html#tuple"
class="reference external" title="(in Python v3.13)"><span
class="pre"><code
class="sourceCode python"><span class="bu">tuple</span></code></span></a> or as separate arguments. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.round.html#mlx.core.array.round"
class="reference internal" title="mlx.core.array.round"><span
class="pre"><code
class="sourceCode python">array.<span class="bu">round</span></code></span></a>(self\[, decimals, stream\]) | See <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.round.html#mlx.core.round"
class="reference internal" title="mlx.core.round"><span
class="pre"><code
class="sourceCode python"><span class="bu">round</span>()</code></span></a>. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.rsqrt.html#mlx.core.array.rsqrt"
class="reference internal" title="mlx.core.array.rsqrt"><span
class="pre"><code
class="sourceCode python">array.rsqrt</code></span></a>(self, \*\[, stream\]) | See <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.rsqrt.html#mlx.core.rsqrt"
class="reference internal" title="mlx.core.rsqrt"><span
class="pre"><code class="sourceCode python">rsqrt()</code></span></a>. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.sin.html#mlx.core.array.sin"
class="reference internal" title="mlx.core.array.sin"><span
class="pre"><code class="sourceCode python">array.sin</code></span></a>(self, \*\[, stream\]) | See <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.sin.html#mlx.core.sin"
class="reference internal" title="mlx.core.sin"><span class="pre"><code
class="sourceCode python">sin()</code></span></a>. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.split.html#mlx.core.array.split"
class="reference internal" title="mlx.core.array.split"><span
class="pre"><code
class="sourceCode python">array.split</code></span></a>(self, indices_or_sections\[, ...\]) | See <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.split.html#mlx.core.split"
class="reference internal" title="mlx.core.split"><span
class="pre"><code class="sourceCode python">split()</code></span></a>. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.sqrt.html#mlx.core.array.sqrt"
class="reference internal" title="mlx.core.array.sqrt"><span
class="pre"><code class="sourceCode python">array.sqrt</code></span></a>(self, \*\[, stream\]) | See <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.sqrt.html#mlx.core.sqrt"
class="reference internal" title="mlx.core.sqrt"><span class="pre"><code
class="sourceCode python">sqrt()</code></span></a>. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.square.html#mlx.core.array.square"
class="reference internal" title="mlx.core.array.square"><span
class="pre"><code
class="sourceCode python">array.square</code></span></a>(self, \*\[, stream\]) | See <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.square.html#mlx.core.square"
class="reference internal" title="mlx.core.square"><span
class="pre"><code class="sourceCode python">square()</code></span></a>. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.squeeze.html#mlx.core.array.squeeze"
class="reference internal" title="mlx.core.array.squeeze"><span
class="pre"><code
class="sourceCode python">array.squeeze</code></span></a>(self\[, axis, stream\]) | See <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.squeeze.html#mlx.core.squeeze"
class="reference internal" title="mlx.core.squeeze"><span
class="pre"><code class="sourceCode python">squeeze()</code></span></a>. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.std.html#mlx.core.array.std"
class="reference internal" title="mlx.core.array.std"><span
class="pre"><code class="sourceCode python">array.std</code></span></a>(self\[, axis, keepdims, ddof, stream\]) | See <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.std.html#mlx.core.std"
class="reference internal" title="mlx.core.std"><span class="pre"><code
class="sourceCode python">std()</code></span></a>. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.sum.html#mlx.core.array.sum"
class="reference internal" title="mlx.core.array.sum"><span
class="pre"><code
class="sourceCode python">array.<span class="bu">sum</span></code></span></a>(self\[, axis, keepdims, stream\]) | See <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.sum.html#mlx.core.sum"
class="reference internal" title="mlx.core.sum"><span class="pre"><code
class="sourceCode python"><span class="bu">sum</span>()</code></span></a>. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.swapaxes.html#mlx.core.array.swapaxes"
class="reference internal" title="mlx.core.array.swapaxes"><span
class="pre"><code
class="sourceCode python">array.swapaxes</code></span></a>(self, axis1, axis2, \*\[, stream\]) | See <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.swapaxes.html#mlx.core.swapaxes"
class="reference internal" title="mlx.core.swapaxes"><span
class="pre"><code class="sourceCode python">swapaxes()</code></span></a>. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.transpose.html#mlx.core.array.transpose"
class="reference internal" title="mlx.core.array.transpose"><span
class="pre"><code
class="sourceCode python">array.transpose</code></span></a>(self, \*axes\[, stream\]) | Equivalent to <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.transpose.html#mlx.core.transpose"
class="reference internal" title="mlx.core.transpose"><span
class="pre"><code
class="sourceCode python">transpose()</code></span></a> but the axes can be passed either as a tuple or as separate arguments. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.T.html#mlx.core.array.T"
class="reference internal" title="mlx.core.array.T"><span
class="pre"><code class="sourceCode python">array.T</code></span></a> | Equivalent to calling <span class="pre">`self.transpose()`</span> with no arguments. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.var.html#mlx.core.array.var"
class="reference internal" title="mlx.core.array.var"><span
class="pre"><code class="sourceCode python">array.var</code></span></a>(self\[, axis, keepdims, ddof, stream\]) | See <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.var.html#mlx.core.var"
class="reference internal" title="mlx.core.var"><span class="pre"><code
class="sourceCode python">var()</code></span></a>. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.view.html#mlx.core.array.view"
class="reference internal" title="mlx.core.array.view"><span
class="pre"><code class="sourceCode python">array.view</code></span></a>(self, dtype, \*\[, stream\]) | See <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.view.html#mlx.core.view"
class="reference internal" title="mlx.core.view"><span class="pre"><code
class="sourceCode python">view()</code></span></a>. |

</div>

</div>

<div class="prev-next-area">

<a
href="https://ml-explore.github.io/mlx/build/html/examples/llama-inference.html"
class="left-prev" title="previous page"><em></em></a>

<div class="prev-next-info">

previous

LLM inference

</div>

<a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.html"
class="right-next" title="next page"></a>

<div class="prev-next-info">

next

mlx.core.array

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
