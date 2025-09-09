Curator's note: Prefer the human-authored MLX guides for clarity.
- ../docs_curated/README.md
- ../docs_curated/PYTORCH_DISSONANCE.md
- ../docs_curated/NUMPY_USERS.md
- ../docs_curated/COMMON_PITFALLS.md

<!--
Per-file analysis (mlx.core.random.key.md):
- Entry point for explicit, functional PRNG keys.
- Hotspot: JAX-like keys vs PyTorch’s global state; splitting for reproducibility/parallelism.
-->

## Curated Notes

- Creates an explicit PRNG key from a seed. Prefer passing `key=` to random ops for reproducibility and parallel, deterministic workflows.
- Split keys with `mx.random.split(key, num=...)` to create independent subkeys without re-seeding.
- No `device=`; placement follows default device or per‑op `stream`.

### Examples

```python
import mlx.core as mx

# Make a key and use it
a = mx.random.normal(shape=(3,), key=mx.random.key(0))

# Reproducible draws with the same key
k = mx.random.key(123)
u1 = mx.random.uniform(shape=(2, 2), key=k)
u2 = mx.random.uniform(shape=(2, 2), key=k)  # identical to u1

# Independent subkeys
k1, k2 = mx.random.split(mx.random.key(7), num=2)
n1 = mx.random.normal(shape=(4,), key=k1)
n2 = mx.random.normal(shape=(4,), key=k2)
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
  href="https://ml-explore.github.io/mlx/build/html/_sources/python/_autosummary/mlx.core.random.key.rst"
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

# mlx.core.random.key

<div id="print-main-content">

<div id="jb-print-toc">

<div>

## Contents

</div>

- <a
  href="https://ml-explore.github.io/mlx/build/html/#mlx.core.random.key"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">key()</code></span></a>

</div>

</div>

</div>

<div id="searchbox">

</div>

<div id="mlx-core-random-key" class="section">

# mlx.core.random.key<a
href="https://ml-explore.github.io/mlx/build/html/#mlx-core-random-key"
class="headerlink" title="Link to this heading">#</a>

<span class="sig-name descname"><span class="pre">key</span></span><span class="sig-paren">(</span>*<span class="n"><span class="pre">seed</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><a href="https://docs.python.org/3/library/functions.html#int"
class="reference external" title="(in Python v3.13)"><span
class="pre">int</span></a></span>*<span class="sig-paren">)</span> <span class="sig-return"><span class="sig-return-icon">→</span> <span class="sig-return-typehint"><a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.html#mlx.core.array"
class="reference internal" title="mlx.core.array"><span
class="pre">array</span></a></span></span><a
href="https://ml-explore.github.io/mlx/build/html/#mlx.core.random.key"
class="headerlink" title="Link to this definition">#</a>  
Get a PRNG key from a seed.

Parameters<span class="colon">:</span>  
**seed** (<a href="https://docs.python.org/3/library/functions.html#int"
class="reference external" title="(in Python v3.13)"><em>int</em></a>) –
Seed for the PRNG.

Returns<span class="colon">:</span>  
The PRNG key array.

Return type<span class="colon">:</span>  
<a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.html#mlx.core.array"
class="reference internal" title="mlx.core.array"><em>array</em></a>

</div>

<div class="prev-next-area">

<a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.random.gumbel.html"
class="left-prev" title="previous page"><em></em></a>

<div class="prev-next-info">

previous

mlx.core.random.gumbel

</div>

<a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.random.normal.html"
class="right-next" title="next page"></a>

<div class="prev-next-info">

next

mlx.core.random.normal

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
  href="https://ml-explore.github.io/mlx/build/html/#mlx.core.random.key"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">key()</code></span></a>

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
