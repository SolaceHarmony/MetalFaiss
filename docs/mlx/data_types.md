Curator's note: Prefer the human-authored MLX guides for clarity.
- ../docs_curated/README.md
- ../docs_curated/PYTORCH_DISSONANCE.md
- ../docs_curated/NUMPY_USERS.md
- ../docs_curated/COMMON_PITFALLS.md

<!--
Per-file analysis (data_types.md):
- Sphinx table of supported dtypes; includes an important GPU caveat for float64.
- Notable quirks: complex64 availability, bfloat16 vs float16 tradeoffs, and dtype up/down-casting with ops.
-->

## Curated Notes

- Defaults: floats are `float32`, integers are `int32` unless specified.
- GPU caveat: `float64` runs on CPU only; prefer `float32` or `bfloat16`/`float16` on GPU.
- Mixed precision: keep numerically sensitive reductions (e.g., loss accumulation) in `float32` even if model weights use lower precision.
- Casting: use `x.astype(mx.float16)` explicitly; avoid implicit Python casts on MLX arrays.

### Examples

```python
import mlx.core as mx

# Inspect dtype and shape
x = mx.ones((2, 3), dtype=mx.float16)
print(x.dtype, x.shape)

# Cast explicitly
x32 = x.astype(mx.float32)

# Mixed precision: keep reductions in float32
fp16_vals = mx.random.normal((1024,), dtype=mx.float16)
mean32 = mx.mean(fp16_vals.astype(mx.float32))

# Complex dtype
z = mx.array([1+2j, 3-4j])
print(z.dtype)

# Integer widths
i16 = mx.arange(0, 5, dtype=mx.int16)
i64 = i16.astype(mx.int64)
```

## Float64 (double precision) Support

MLX has limited float64 support tied to the execution device.

- CPU: float64 is supported on CPU and participates in automatic differentiation.
- GPU: float64 is not supported by Metal; attempting to run float64 ops on GPU raises an exception.
- NumPy interop: converting a NumPy float64 array to MLX may downcast to float32 unless you target CPU and request `dtype=mx.float64`.

### Using float64 on CPU

```python
import mlx.core as mx

# Per-op routing to CPU
a64 = mx.ones((4,), dtype=mx.float64)
out = mx.add(a64, 2.0, stream=mx.cpu)

# Or set default device globally to CPU for double-precision sections
mx.set_default_device(mx.cpu)
x = mx.random.normal((1024,), dtype=mx.float64)
y = mx.random.normal((1024,), dtype=mx.float64)
z = mx.sum(x * y)  # runs on CPU; retains float64
```

### Limitations on GPU

- Unsupported dtype: float64 ops on GPU will error at runtime.
- Be explicit about dtypes when moving between frameworks; verify with `arr.dtype` after conversions.

```python
import numpy as np
import mlx.core as mx

np_arr = np.arange(5, dtype=np.float64)
mlx_arr = mx.array(np_arr)          # may be float32 if default device is GPU
print(mlx_arr.dtype)

# Ensure float64 on CPU
mlx_arr_cpu64 = mx.array(np_arr, dtype=mx.float64)
with mx.default_device(mx.cpu):
    print(mlx_arr_cpu64.dtype)      # float64
```

### Precision‑Sensitive Workflows

- CPU‑only segments: run double‑precision sections entirely on CPU with `mx.float64`.
- Mixed precision: keep numerically sensitive reductions/solves in float64 on CPU and cast results to float32 for GPU‑accelerated parts.
- Explicit conversions: insert `astype(...)` at the device/precision boundaries; validate numerics with small probes.
- Advanced: double‑double techniques can emulate higher precision in software when float64 is unavailable on your target device.

```python
# Example: high‑precision reduce on CPU, rest on default device
mx.set_default_device(mx.gpu)  # assume your default is GPU
v = mx.random.normal((1_000_000,), dtype=mx.float32)

sum64 = mx.sum(v.astype(mx.float64), stream=mx.cpu)  # precise sum on CPU
sum32 = sum64.astype(mx.float32)                     # cast back for GPU work

result = mx.sqrt(sum32 + 1.0)                        # continue on default device
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
  href="https://ml-explore.github.io/mlx/build/html/_sources/python/data_types.rst"
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

# Data Types

<div id="print-main-content">

<div id="jb-print-toc">

</div>

</div>

</div>

<div id="searchbox">

</div>

<div id="data-types" class="section">

<span id="id1"></span>

# Data Types<a href="https://ml-explore.github.io/mlx/build/html/#data-types"
class="headerlink" title="Link to this heading">#</a>

The default floating point type is <span class="pre">`float32`</span>
and the default integer type is <span class="pre">`int32`</span>. The
table below shows supported values for <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.Dtype.html#mlx.core.Dtype"
class="reference internal" title="mlx.core.Dtype"><span
class="pre"><code class="sourceCode python">Dtype</code></span></a>.

<div class="pst-scrollable-table-container">

| Type | Bytes | Description |
|----|----|----|
| <span class="pre">`bool_`</span> | 1 | Boolean (<span class="pre">`True`</span>, <span class="pre">`False`</span>) data type |
| <span class="pre">`uint8`</span> | 1 | 8-bit unsigned integer |
| <span class="pre">`uint16`</span> | 2 | 16-bit unsigned integer |
| <span class="pre">`uint32`</span> | 4 | 32-bit unsigned integer |
| <span class="pre">`uint64`</span> | 8 | 64-bit unsigned integer |
| <span class="pre">`int8`</span> | 1 | 8-bit signed integer |
| <span class="pre">`int16`</span> | 2 | 16-bit signed integer |
| <span class="pre">`int32`</span> | 4 | 32-bit signed integer |
| <span class="pre">`int64`</span> | 8 | 64-bit signed integer |
| <span class="pre">`bfloat16`</span> | 2 | 16-bit brain float (e8, m7) |
| <span class="pre">`float16`</span> | 2 | 16-bit IEEE float (e5, m10) |
| <span class="pre">`float32`</span> | 4 | 32-bit float |
| <span class="pre">`float64`</span> | 4 | 64-bit double |
| <span class="pre">`complex64`</span> | 8 | 64-bit complex float |

<span class="caption-text">Supported Data
Types</span><a href="https://ml-explore.github.io/mlx/build/html/#id2"
class="headerlink" title="Link to this table">#</a> {#id2}

</div>

<div class="admonition note">

Note

Arrays with type <span class="pre">`float64`</span> only work with CPU
operations. Using <span class="pre">`float64`</span> arrays on the GPU
will result in an exception.

</div>

Data type are aranged in a hierarchy. See the <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.DtypeCategory.html#mlx.core.DtypeCategory"
class="reference internal" title="mlx.core.DtypeCategory"><span
class="pre"><code
class="sourceCode python">DtypeCategory</code></span></a> object
documentation for more information. Use <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.issubdtype.html#mlx.core.issubdtype"
class="reference internal" title="mlx.core.issubdtype"><span
class="pre"><code
class="sourceCode python">issubdtype()</code></span></a> to determine if
one <span class="pre">`dtype`</span> (or category) is a subtype of
another category.

<div class="pst-scrollable-table-container">

|  |  |
|----|----|
| <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.Dtype.html#mlx.core.Dtype"
class="reference internal" title="mlx.core.Dtype"><span
class="pre"><code class="sourceCode python">Dtype</code></span></a> | An object to hold the type of a <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.html#mlx.core.array"
class="reference internal" title="mlx.core.array"><span
class="pre"><code class="sourceCode python">array</code></span></a>. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.DtypeCategory.html#mlx.core.DtypeCategory"
class="reference internal" title="mlx.core.DtypeCategory"><span
class="pre"><code
class="sourceCode python">DtypeCategory</code></span></a>(value) | Type to hold categories of <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.Dtype.html#mlx.core.Dtype"
class="reference internal" title="mlx.core.Dtype"><span
class="pre"><code class="sourceCode python">dtypes</code></span></a>. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.issubdtype.html#mlx.core.issubdtype"
class="reference internal" title="mlx.core.issubdtype"><span
class="pre"><code class="sourceCode python">issubdtype</code></span></a>(arg1, arg2) | Check if a <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.Dtype.html#mlx.core.Dtype"
class="reference internal" title="mlx.core.Dtype"><span
class="pre"><code class="sourceCode python">Dtype</code></span></a> or <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.DtypeCategory.html#mlx.core.DtypeCategory"
class="reference internal" title="mlx.core.DtypeCategory"><span
class="pre"><code
class="sourceCode python">DtypeCategory</code></span></a> is a subtype of another. |
| <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.finfo.html#mlx.core.finfo"
class="reference internal" title="mlx.core.finfo"><span
class="pre"><code class="sourceCode python">finfo</code></span></a> | Get information on floating-point types. |

</div>

</div>

<div class="prev-next-area">

<a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.view.html"
class="left-prev" title="previous page"><em></em></a>

<div class="prev-next-info">

previous

mlx.core.array.view

</div>

<a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.Dtype.html"
class="right-next" title="next page"></a>

<div class="prev-next-info">

next

mlx.core.Dtype

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
