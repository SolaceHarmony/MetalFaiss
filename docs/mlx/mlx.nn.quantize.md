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
  href="https://ml-explore.github.io/mlx/build/html/_sources/python/_autosummary/mlx.nn.quantize.rst"
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

# mlx.nn.quantize

<div id="print-main-content">

<div id="jb-print-toc">

<div>

## Contents

</div>

- <a href="https://ml-explore.github.io/mlx/build/html/#mlx.nn.quantize"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">quantize()</code></span></a>

</div>

</div>

</div>

<div id="searchbox">

</div>

<div id="mlx-nn-quantize" class="section">

# mlx.nn.quantize<a href="https://ml-explore.github.io/mlx/build/html/#mlx-nn-quantize"
class="headerlink" title="Link to this heading">#</a>

<span class="sig-name descname"><span class="pre">quantize</span></span><span class="sig-paren">(</span>*<span class="n"><span class="pre">model</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><a
href="https://ml-explore.github.io/mlx/build/html/python/nn/module.html#mlx.nn.Module"
class="reference internal" title="mlx.nn.layers.base.Module"><span
class="pre">Module</span></a></span>*, *<span class="n"><span class="pre">group_size</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><a href="https://docs.python.org/3/library/functions.html#int"
class="reference external" title="(in Python v3.13)"><span
class="pre">int</span></a></span><span class="w"> </span><span class="o"><span class="pre">=</span></span><span class="w"> </span><span class="default_value"><span class="pre">64</span></span>*, *<span class="n"><span class="pre">bits</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><a href="https://docs.python.org/3/library/functions.html#int"
class="reference external" title="(in Python v3.13)"><span
class="pre">int</span></a></span><span class="w"> </span><span class="o"><span class="pre">=</span></span><span class="w"> </span><span class="default_value"><span class="pre">4</span></span>*, *<span class="n"><span class="pre">class_predicate</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><a href="https://docs.python.org/3/library/typing.html#typing.Callable"
class="reference external" title="(in Python v3.13)"><span
class="pre">Callable</span></a><span class="p"><span class="pre">\[</span></span><span class="p"><span class="pre">\[</span></span><a href="https://docs.python.org/3/library/stdtypes.html#str"
class="reference external" title="(in Python v3.13)"><span
class="pre">str</span></a><span class="p"><span class="pre">,</span></span><span class="w"> </span><a
href="https://ml-explore.github.io/mlx/build/html/python/nn/module.html#mlx.nn.Module"
class="reference internal" title="mlx.nn.layers.base.Module"><span
class="pre">Module</span></a><span class="p"><span class="pre">\]</span></span><span class="p"><span class="pre">,</span></span><span class="w"> </span><a href="https://docs.python.org/3/library/functions.html#bool"
class="reference external" title="(in Python v3.13)"><span
class="pre">bool</span></a><span class="w"> </span><span class="p"><span class="pre">\|</span></span><span class="w"> </span><a href="https://docs.python.org/3/library/stdtypes.html#dict"
class="reference external" title="(in Python v3.13)"><span
class="pre">dict</span></a><span class="p"><span class="pre">\]</span></span><span class="w"> </span><span class="p"><span class="pre">\|</span></span><span class="w"> </span><a href="https://docs.python.org/3/library/constants.html#None"
class="reference external" title="(in Python v3.13)"><span
class="pre">None</span></a></span><span class="w"> </span><span class="o"><span class="pre">=</span></span><span class="w"> </span><span class="default_value"><span class="pre">None</span></span>*<span class="sig-paren">)</span><a href="https://ml-explore.github.io/mlx/build/html/#mlx.nn.quantize"
class="headerlink" title="Link to this definition">#</a>  
Quantize the sub-modules of a module according to a predicate.

By default all layers that define a
<span class="pre">`to_quantized(group_size,`</span>` `<span class="pre">`bits)`</span>
method will be quantized. Both <a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Linear.html#mlx.nn.Linear"
class="reference internal" title="mlx.nn.Linear"><span class="pre"><code
class="sourceCode python">Linear</code></span></a> and <a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Embedding.html#mlx.nn.Embedding"
class="reference internal" title="mlx.nn.Embedding"><span
class="pre"><code class="sourceCode python">Embedding</code></span></a>
layers will be quantized. Note also, the module is updated in-place.

Parameters<span class="colon">:</span>  
- **model** (<a
  href="https://ml-explore.github.io/mlx/build/html/python/nn/module.html#mlx.nn.Module"
  class="reference internal" title="mlx.nn.Module"><em>Module</em></a>)
  – The model whose leaf modules may be quantized.

- **group_size**
  (<a href="https://docs.python.org/3/library/functions.html#int"
  class="reference external" title="(in Python v3.13)"><em>int</em></a>)
  – The quantization group size (see <a
  href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.quantize.html#mlx.core.quantize"
  class="reference internal" title="mlx.core.quantize"><span
  class="pre"><code
  class="sourceCode python">mlx.core.quantize()</code></span></a>).
  Default: <span class="pre">`64`</span>.

- **bits**
  (<a href="https://docs.python.org/3/library/functions.html#int"
  class="reference external" title="(in Python v3.13)"><em>int</em></a>)
  – The number of bits per parameter (see <a
  href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.quantize.html#mlx.core.quantize"
  class="reference internal" title="mlx.core.quantize"><span
  class="pre"><code
  class="sourceCode python">mlx.core.quantize()</code></span></a>).
  Default: <span class="pre">`4`</span>.

- **class_predicate** (*Optional\[Callable\]*) – A callable which
  receives the <a
  href="https://ml-explore.github.io/mlx/build/html/python/nn/module.html#mlx.nn.Module"
  class="reference internal" title="mlx.nn.Module"><span class="pre"><code
  class="sourceCode python">Module</code></span></a> path and <a
  href="https://ml-explore.github.io/mlx/build/html/python/nn/module.html#mlx.nn.Module"
  class="reference internal" title="mlx.nn.Module"><span class="pre"><code
  class="sourceCode python">Module</code></span></a> itself and returns
  <span class="pre">`True`</span> or a dict of params for to_quantized
  if it should be quantized and <span class="pre">`False`</span>
  otherwise. If <span class="pre">`None`</span>, then all layers that
  define a
  <span class="pre">`to_quantized(group_size,`</span>` `<span class="pre">`bits)`</span>
  method are quantized. Default: <span class="pre">`None`</span>.

</div>

<div class="prev-next-area">

<a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.nn.value_and_grad.html"
class="left-prev" title="previous page"><em></em></a>

<div class="prev-next-info">

previous

mlx.nn.value_and_grad

</div>

<a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.nn.average_gradients.html"
class="right-next" title="next page"></a>

<div class="prev-next-info">

next

mlx.nn.average_gradients

</div>

</div>

</div>

<div class="bd-sidebar-secondary bd-toc">

<div class="sidebar-secondary-items sidebar-secondary__inner">

<div class="sidebar-secondary-item">

<div class="page-toc tocsection onthispage">

Contents

</div>

- <a href="https://ml-explore.github.io/mlx/build/html/#mlx.nn.quantize"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">quantize()</code></span></a>

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
