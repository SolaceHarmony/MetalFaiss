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
  href="https://ml-explore.github.io/mlx/build/html/_sources/python/_autosummary/mlx.core.random.multivariate_normal.rst"
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

# mlx.core.random.multivariate_normal

<div id="print-main-content">

<div id="jb-print-toc">

<div>

## Contents

</div>

- <a
  href="https://ml-explore.github.io/mlx/build/html/#mlx.core.random.multivariate_normal"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">multivariate_normal()</code></span></a>

</div>

</div>

</div>

<div id="searchbox">

</div>

<div id="mlx-core-random-multivariate-normal" class="section">

# mlx.core.random.multivariate_normal<a
href="https://ml-explore.github.io/mlx/build/html/#mlx-core-random-multivariate-normal"
class="headerlink" title="Link to this heading">#</a>

<span class="sig-name descname"><span class="pre">multivariate_normal</span></span><span class="sig-paren">(</span>*<span class="n"><span class="pre">mean</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.html#mlx.core.array"
class="reference internal" title="mlx.core.array"><span
class="pre">array</span></a></span>*, *<span class="n"><span class="pre">cov</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.html#mlx.core.array"
class="reference internal" title="mlx.core.array"><span
class="pre">array</span></a></span>*, *<span class="n"><span class="pre">shape</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><span class="pre">Sequence</span><span class="p"><span class="pre">\[</span></span><a href="https://docs.python.org/3/library/functions.html#int"
class="reference external" title="(in Python v3.13)"><span
class="pre">int</span></a><span class="p"><span class="pre">\]</span></span></span><span class="w"> </span><span class="o"><span class="pre">=</span></span><span class="w"> </span><span class="default_value"><span class="pre">\[\]</span></span>*, *<span class="n"><span class="pre">dtype</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.Dtype.html#mlx.core.Dtype"
class="reference internal" title="mlx.core.Dtype"><span
class="pre">Dtype</span></a><span class="w"> </span><span class="p"><span class="pre">\|</span></span><span class="w"> </span><a href="https://docs.python.org/3/library/constants.html#None"
class="reference external" title="(in Python v3.13)"><span
class="pre">None</span></a></span><span class="w"> </span><span class="o"><span class="pre">=</span></span><span class="w"> </span><span class="default_value"><span class="pre">float32</span></span>*, *<span class="n"><span class="pre">key</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.html#mlx.core.array"
class="reference internal" title="mlx.core.array"><span
class="pre">array</span></a><span class="w"> </span><span class="p"><span class="pre">\|</span></span><span class="w"> </span><a href="https://docs.python.org/3/library/constants.html#None"
class="reference external" title="(in Python v3.13)"><span
class="pre">None</span></a></span><span class="w"> </span><span class="o"><span class="pre">=</span></span><span class="w"> </span><span class="default_value"><span class="pre">None</span></span>*, *<span class="n"><span class="pre">stream</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><a href="https://docs.python.org/3/library/constants.html#None"
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
class="pre">array</span></a></span></span><a
href="https://ml-explore.github.io/mlx/build/html/#mlx.core.random.multivariate_normal"
class="headerlink" title="Link to this definition">#</a>  
Generate jointly-normal random samples given a mean and covariance.

The matrix <span class="pre">`cov`</span> must be positive
semi-definite. The behavior is undefined if it is not. The only
supported <span class="pre">`dtype`</span> is
<span class="pre">`float32`</span>.

## Curated Notes

- Inputs: `mean` shape `(d,)` or `(..., d)`; `cov` shape `(d, d)` or `(..., d, d)`. `cov` should be symmetric positive semidefinite.
- Implementation typically uses a factorization of `cov` (e.g., Cholesky): sample `z ~ N(0, I)`, return `mean + L @ z`.
- Batched shapes are supported; parameters broadcast across draws; you can also pass `shape=(num_samples,)` to draw multiple i.i.d. samples.
- For reproducibility and parallel code, pass `key=` or split subkeys.

### Examples

```python
import mlx.core as mx

d = 3
mu = mx.zeros((d,))
A = mx.random.normal((d, d))
cov = (A @ A.T) + 1e-3 * mx.eye(d)         # SPD

# One sample
x = mx.random.multivariate_normal(mu, cov)

# Many samples with key
k = mx.random.key(0)
xs = mx.random.multivariate_normal(mu, cov, shape=(1024,), key=k)

# Manual construction via Cholesky (ensure double on CPU if needed)
with mx.default_device(mx.cpu):
    L = mx.linalg.cholesky(cov.astype(mx.float64))
z = mx.random.normal((d,))
sample = mu + (L @ z)
```

Parameters<span class="colon">:</span>  
- **mean** (<a
  href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.html#mlx.core.array"
  class="reference internal" title="mlx.core.array"><em>array</em></a>)
  – array of shape
  <span class="pre">`(...,`</span>` `<span class="pre">`n)`</span>, the
  mean of the distribution.

- **cov** (<a
  href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.html#mlx.core.array"
  class="reference internal" title="mlx.core.array"><em>array</em></a>)
  – array of shape
  <span class="pre">`(...,`</span>` `<span class="pre">`n,`</span>` `<span class="pre">`n)`</span>,
  the covariance matrix of the distribution. The batch shape
  <span class="pre">`...`</span> must be broadcast-compatible with that
  of <span class="pre">`mean`</span>.

- **shape**
  (<a href="https://docs.python.org/3/library/stdtypes.html#list"
  class="reference external" title="(in Python v3.13)"><em>list</em></a>*(*<a href="https://docs.python.org/3/library/functions.html#int"
  class="reference external" title="(in Python v3.13)"><em>int</em></a>*),*
  *optional*) – The output shape must be broadcast-compatible with
  <span class="pre">`mean.shape[:-1]`</span> and
  <span class="pre">`cov.shape[:-2]`</span>. If empty, the result shape
  is determined by broadcasting the batch shapes of
  <span class="pre">`mean`</span> and <span class="pre">`cov`</span>.
  Default: <span class="pre">`[]`</span>.

- **dtype** (<a
  href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.Dtype.html#mlx.core.Dtype"
  class="reference internal" title="mlx.core.Dtype"><em>Dtype</em></a>*,*
  *optional*) – The output type. Default:
  <span class="pre">`float32`</span>.

- **key** (<a
  href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.html#mlx.core.array"
  class="reference internal" title="mlx.core.array"><em>array</em></a>*,*
  *optional*) – A PRNG key. Default: <span class="pre">`None`</span>.

Returns<span class="colon">:</span>  
The output array of random values.

Return type<span class="colon">:</span>  
<a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.html#mlx.core.array"
class="reference internal" title="mlx.core.array"><em>array</em></a>

</div>

<div class="prev-next-area">

<a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.random.normal.html"
class="left-prev" title="previous page"><em></em></a>

<div class="prev-next-info">

previous

mlx.core.random.normal

</div>

<a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.random.randint.html"
class="right-next" title="next page"></a>

<div class="prev-next-info">

next

mlx.core.random.randint

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
  href="https://ml-explore.github.io/mlx/build/html/#mlx.core.random.multivariate_normal"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">multivariate_normal()</code></span></a>

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
