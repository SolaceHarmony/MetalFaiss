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
  href="https://ml-explore.github.io/mlx/build/html/_sources/python/_autosummary/mlx.core.fast.scaled_dot_product_attention.rst"
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

# mlx.core.fast.scaled_dot_product_attention

<div id="print-main-content">

<div id="jb-print-toc">

<div>

## Contents

</div>

- <a
  href="https://ml-explore.github.io/mlx/build/html/#mlx.core.fast.scaled_dot_product_attention"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">scaled_dot_product_attention()</code></span></a>

</div>

</div>

</div>

<div id="searchbox">

</div>

<div id="mlx-core-fast-scaled-dot-product-attention" class="section">

# mlx.core.fast.scaled_dot_product_attention<a
href="https://ml-explore.github.io/mlx/build/html/#mlx-core-fast-scaled-dot-product-attention"
class="headerlink" title="Link to this heading">#</a>

<span class="sig-name descname"><span class="pre">scaled_dot_product_attention</span></span><span class="sig-paren">(</span>*<span class="n"><span class="pre">q</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.html#mlx.core.array"
class="reference internal" title="mlx.core.array"><span
class="pre">array</span></a></span>*, *<span class="n"><span class="pre">k</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.html#mlx.core.array"
class="reference internal" title="mlx.core.array"><span
class="pre">array</span></a></span>*, *<span class="n"><span class="pre">v</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.html#mlx.core.array"
class="reference internal" title="mlx.core.array"><span
class="pre">array</span></a></span>*, *<span class="o"><span class="pre">\*</span></span>*, *<span class="n"><span class="pre">scale</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><a href="https://docs.python.org/3/library/functions.html#float"
class="reference external" title="(in Python v3.13)"><span
class="pre">float</span></a></span>*, *<span class="n"><span class="pre">mask</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><a href="https://docs.python.org/3/library/constants.html#None"
class="reference external" title="(in Python v3.13)"><span
class="pre">None</span></a><span class="w"> </span><span class="p"><span class="pre">\|</span></span><span class="w"> </span><a href="https://docs.python.org/3/library/stdtypes.html#str"
class="reference external" title="(in Python v3.13)"><span
class="pre">str</span></a><span class="w"> </span><span class="p"><span class="pre">\|</span></span><span class="w"> </span><a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.html#mlx.core.array"
class="reference internal" title="mlx.core.array"><span
class="pre">array</span></a></span><span class="w"> </span><span class="o"><span class="pre">=</span></span><span class="w"> </span><span class="default_value"><span class="pre">None</span></span>*, *<span class="n"><span class="pre">stream</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><a href="https://docs.python.org/3/library/constants.html#None"
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
href="https://ml-explore.github.io/mlx/build/html/#mlx.core.fast.scaled_dot_product_attention"
class="headerlink" title="Link to this definition">#</a>  
A fast implementation of multi-head attention:
<span class="pre">`O`</span>` `<span class="pre">`=`</span>` `<span class="pre">`softmax(Q`</span>` `<span class="pre">`@`</span>` `<span class="pre">`K.T,`</span>` `<span class="pre">`dim=-1)`</span>` `<span class="pre">`@`</span>` `<span class="pre">`V`</span>.

Supports:

- <a href="https://arxiv.org/abs/1706.03762"
  class="reference external">Multi-Head Attention</a>

- <a href="https://arxiv.org/abs/2305.13245"
  class="reference external">Grouped Query Attention</a>

- <a href="https://arxiv.org/abs/1911.02150"
  class="reference external">Multi-Query Attention</a>

Note: The softmax operation is performed in
<span class="pre">`float32`</span> regardless of the input precision.

Note: For Grouped Query Attention and Multi-Query Attention, the
<span class="pre">`k`</span> and <span class="pre">`v`</span> inputs
should not be pre-tiled to match <span class="pre">`q`</span>.

In the following the dimensions are given by:

- <span class="pre">`B`</span>: The batch size.

- <span class="pre">`N_q`</span>: The number of query heads.

- <span class="pre">`N_kv`</span>: The number of key and value heads.

- <span class="pre">`T_q`</span>: The number of queries per example.

- <span class="pre">`T_kv`</span>: The number of keys and values per
  example.

- <span class="pre">`D`</span>: The per-head dimension.

Parameters<span class="colon">:</span>  
- **q** (<a
  href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.html#mlx.core.array"
  class="reference internal" title="mlx.core.array"><em>array</em></a>)
  – Queries with shape
  <span class="pre">`[B,`</span>` `<span class="pre">`N_q,`</span>` `<span class="pre">`T_q,`</span>` `<span class="pre">`D]`</span>.

- **k** (<a
  href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.html#mlx.core.array"
  class="reference internal" title="mlx.core.array"><em>array</em></a>)
  – Keys with shape
  <span class="pre">`[B,`</span>` `<span class="pre">`N_kv,`</span>` `<span class="pre">`T_kv,`</span>` `<span class="pre">`D]`</span>.

- **v** (<a
  href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.html#mlx.core.array"
  class="reference internal" title="mlx.core.array"><em>array</em></a>)
  – Values with shape
  <span class="pre">`[B,`</span>` `<span class="pre">`N_kv,`</span>` `<span class="pre">`T_kv,`</span>` `<span class="pre">`D]`</span>.

- **scale**
  (<a href="https://docs.python.org/3/library/functions.html#float"
  class="reference external" title="(in Python v3.13)"><em>float</em></a>)
  – Scale for queries (typically
  <span class="pre">`1.0`</span>` `<span class="pre">`/`</span>` `<span class="pre">`sqrt(q.shape(-1)`</span>)

- **mask** (*Union\[None,*
  <a href="https://docs.python.org/3/library/stdtypes.html#str"
  class="reference external" title="(in Python v3.13)"><em>str</em></a>*,*
  <a
  href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.html#mlx.core.array"
  class="reference internal" title="mlx.core.array"><em>array</em></a>*\],*
  *optional*) – A causal, boolean or additive mask to apply to the
  query-key scores. The mask can have at most 4 dimensions and must be
  broadcast-compatible with the shape
  <span class="pre">`[B,`</span>` `<span class="pre">`N,`</span>` `<span class="pre">`T_q,`</span>` `<span class="pre">`T_kv]`</span>.
  If an additive mask is given its type must promote to the promoted
  type of <span class="pre">`q`</span>, <span class="pre">`k`</span>,
  and <span class="pre">`v`</span>.

Returns<span class="colon">:</span>  
The output array.

Return type<span class="colon">:</span>  
<a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.html#mlx.core.array"
class="reference internal" title="mlx.core.array"><em>array</em></a>

</div>

<div class="prev-next-area">

<a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fast.rope.html"
class="left-prev" title="previous page"><em></em></a>

<div class="prev-next-info">

previous

mlx.core.fast.rope

</div>

<a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fast.metal_kernel.html"
class="right-next" title="next page"></a>

<div class="prev-next-info">

next

mlx.core.fast.metal_kernel

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
  href="https://ml-explore.github.io/mlx/build/html/#mlx.core.fast.scaled_dot_product_attention"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">scaled_dot_product_attention()</code></span></a>

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
