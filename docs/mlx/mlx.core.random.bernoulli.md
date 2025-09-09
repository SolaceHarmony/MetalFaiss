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
  href="https://ml-explore.github.io/mlx/build/html/_sources/python/_autosummary/mlx.core.random.bernoulli.rst"
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

# mlx.core.random.bernoulli

<div id="print-main-content">

<div id="jb-print-toc">

<div>

## Contents

</div>

- <a
  href="https://ml-explore.github.io/mlx/build/html/#mlx.core.random.bernoulli"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">bernoulli()</code></span></a>

</div>

</div>

</div>

<div id="searchbox">

</div>

<div id="mlx-core-random-bernoulli" class="section">

# mlx.core.random.bernoulli<a
href="https://ml-explore.github.io/mlx/build/html/#mlx-core-random-bernoulli"
class="headerlink" title="Link to this heading">#</a>

<span class="sig-name descname"><span class="pre">bernoulli</span></span><span class="sig-paren">(</span>*<span class="n"><span class="pre">p</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><span class="pre">scalar</span><span class="w"> </span><span class="p"><span class="pre">\|</span></span><span class="w"> </span><a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.html#mlx.core.array"
class="reference internal" title="mlx.core.array"><span
class="pre">array</span></a></span><span class="w"> </span><span class="o"><span class="pre">=</span></span><span class="w"> </span><span class="default_value"><span class="pre">0.5</span></span>*, *<span class="n"><span class="pre">shape</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><span class="pre">Sequence</span><span class="p"><span class="pre">\[</span></span><a href="https://docs.python.org/3/library/functions.html#int"
class="reference external" title="(in Python v3.13)"><span
class="pre">int</span></a><span class="p"><span class="pre">\]</span></span><span class="w"> </span><span class="p"><span class="pre">\|</span></span><span class="w"> </span><a href="https://docs.python.org/3/library/constants.html#None"
class="reference external" title="(in Python v3.13)"><span
class="pre">None</span></a></span><span class="w"> </span><span class="o"><span class="pre">=</span></span><span class="w"> </span><span class="default_value"><span class="pre">None</span></span>*, *<span class="n"><span class="pre">key</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><a
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
href="https://ml-explore.github.io/mlx/build/html/#mlx.core.random.bernoulli"
class="headerlink" title="Link to this definition">#</a>  
Generate Bernoulli random values.

## Curated Notes

- Outputs 0/1 samples; cast to your desired dtype as needed (e.g., `.astype(mx.int32)` or `.astype(mx.float32)`).
- `p` may be scalar or array and broadcasts over `shape`.
- For reproducibility, pass `key=` or split subkeys; avoid calling `mx.random.seed` inside functions used with `mx.vmap`, `mx.value_and_grad`, or `mx.compile`.

### Examples

```python
import mlx.core as mx

# Basic draw
b = mx.random.bernoulli(p=0.3, shape=(4, 5))

# Deterministic with key
k = mx.random.key(7)
b1 = mx.random.bernoulli(0.5, (3, 3), key=k)
b2 = mx.random.bernoulli(0.5, (3, 3), key=k)  # identical

# Broadcast p over shape
p = mx.array([0.1, 0.9])
s = mx.random.bernoulli(p=p, shape=(5, 2)).astype(mx.int32)
```

The values are sampled from the bernoulli distribution with parameter
<span class="pre">`p`</span>. The parameter <span class="pre">`p`</span>
can be a
<a href="https://docs.python.org/3/library/functions.html#float"
class="reference external" title="(in Python v3.13)"><span
class="pre"><code
class="sourceCode python"><span class="bu">float</span></code></span></a>
or <a href="https://docs.python.org/3/library/array.html#module-array"
class="reference external" title="(in Python v3.13)"><span
class="pre"><code class="sourceCode python">array</code></span></a> and
must be broadcastable to <span class="pre">`shape`</span>.

Parameters<span class="colon">:</span>  
- **p**
  (<a href="https://docs.python.org/3/library/functions.html#float"
  class="reference external" title="(in Python v3.13)"><em>float</em></a>
  *or* <a
  href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.html#mlx.core.array"
  class="reference internal" title="mlx.core.array"><em>array</em></a>*,*
  *optional*) – Parameter of the Bernoulli distribution. Default:
  <span class="pre">`0.5`</span>.

- **shape**
  (<a href="https://docs.python.org/3/library/stdtypes.html#list"
  class="reference external" title="(in Python v3.13)"><em>list</em></a>*(*<a href="https://docs.python.org/3/library/functions.html#int"
  class="reference external" title="(in Python v3.13)"><em>int</em></a>*),*
  *optional*) – Shape of the output. Default:
  <span class="pre">`p.shape`</span>.

- **key** (<a
  href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.html#mlx.core.array"
  class="reference internal" title="mlx.core.array"><em>array</em></a>*,*
  *optional*) – A PRNG key. Default: <span class="pre">`None`</span>.

Returns<span class="colon">:</span>  
The array of random integers.

Return type<span class="colon">:</span>  
<a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.html#mlx.core.array"
class="reference internal" title="mlx.core.array"><em>array</em></a>

</div>

<div class="prev-next-area">

<a href="https://ml-explore.github.io/mlx/build/html/python/random.html"
class="left-prev" title="previous page"><em></em></a>

<div class="prev-next-info">

previous

Random

</div>

<a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.random.categorical.html"
class="right-next" title="next page"></a>

<div class="prev-next-info">

next

mlx.core.random.categorical

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
  href="https://ml-explore.github.io/mlx/build/html/#mlx.core.random.bernoulli"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">bernoulli()</code></span></a>

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
