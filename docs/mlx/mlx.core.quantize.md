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
  href="https://ml-explore.github.io/mlx/build/html/_sources/python/_autosummary/mlx.core.quantize.rst"
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

# mlx.core.quantize

<div id="print-main-content">

<div id="jb-print-toc">

<div>

## Contents

</div>

- <a href="https://ml-explore.github.io/mlx/build/html/#mlx.core.quantize"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">quantize()</code></span></a>

</div>

</div>

</div>

<div id="searchbox">

</div>

<div id="mlx-core-quantize" class="section">

# mlx.core.quantize<a href="https://ml-explore.github.io/mlx/build/html/#mlx-core-quantize"
class="headerlink" title="Link to this heading">#</a>

<span class="sig-name descname"><span class="pre">quantize</span></span><span class="sig-paren">(</span>*<span class="n"><span class="pre">w</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.html#mlx.core.array"
class="reference internal" title="mlx.core.array"><span
class="pre">array</span></a></span>*, *<span class="o"><span class="pre">/</span></span>*, *<span class="n"><span class="pre">group_size</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><a href="https://docs.python.org/3/library/functions.html#int"
class="reference external" title="(in Python v3.13)"><span
class="pre">int</span></a></span><span class="w"> </span><span class="o"><span class="pre">=</span></span><span class="w"> </span><span class="default_value"><span class="pre">64</span></span>*, *<span class="n"><span class="pre">bits</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><a href="https://docs.python.org/3/library/functions.html#int"
class="reference external" title="(in Python v3.13)"><span
class="pre">int</span></a></span><span class="w"> </span><span class="o"><span class="pre">=</span></span><span class="w"> </span><span class="default_value"><span class="pre">4</span></span>*, *<span class="o"><span class="pre">\*</span></span>*, *<span class="n"><span class="pre">stream</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><a href="https://docs.python.org/3/library/constants.html#None"
class="reference external" title="(in Python v3.13)"><span
class="pre">None</span></a><span class="w"> </span><span class="p"><span class="pre">\|</span></span><span class="w"> </span><a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/stream_class.html#mlx.core.Stream"
class="reference internal" title="mlx.core.Stream"><span
class="pre">Stream</span></a><span class="w"> </span><span class="p"><span class="pre">\|</span></span><span class="w"> </span><a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.Device.html#mlx.core.Device"
class="reference internal" title="mlx.core.Device"><span
class="pre">Device</span></a></span><span class="w"> </span><span class="o"><span class="pre">=</span></span><span class="w"> </span><span class="default_value"><span class="pre">None</span></span>*<span class="sig-paren">)</span> <span class="sig-return"><span class="sig-return-icon">→</span> <span class="sig-return-typehint"><a href="https://docs.python.org/3/library/stdtypes.html#tuple"
class="reference external" title="(in Python v3.13)"><span
class="pre">tuple</span></a><span class="p"><span class="pre">\[</span></span><a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.html#mlx.core.array"
class="reference internal" title="mlx.core.array"><span
class="pre">array</span></a><span class="p"><span class="pre">,</span></span><span class="w"> </span><a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.html#mlx.core.array"
class="reference internal" title="mlx.core.array"><span
class="pre">array</span></a><span class="p"><span class="pre">,</span></span><span class="w"> </span><a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.html#mlx.core.array"
class="reference internal" title="mlx.core.array"><span
class="pre">array</span></a><span class="p"><span class="pre">\]</span></span></span></span><a href="https://ml-explore.github.io/mlx/build/html/#mlx.core.quantize"
class="headerlink" title="Link to this definition">#</a>  
Quantize the matrix <span class="pre">`w`</span> using
<span class="pre">`bits`</span> bits per element.

Note, every <span class="pre">`group_size`</span> elements in a row of
<span class="pre">`w`</span> are quantized together. Hence, number of
columns of <span class="pre">`w`</span> should be divisible by
<span class="pre">`group_size`</span>. In particular, the rows of
<span class="pre">`w`</span> are divided into groups of size
<span class="pre">`group_size`</span> which are quantized together.

<div class="admonition warning">

Warning

<span class="pre">`quantize`</span> currently only supports 2D inputs
with dimensions which are multiples of 32

</div>

Formally, for a group of
<span class="math notranslate nohighlight">\\g\\</span> consecutive
elements <span class="math notranslate nohighlight">\\w_1\\</span> to
<span class="math notranslate nohighlight">\\w_g\\</span> in a row of
<span class="pre">`w`</span> we compute the quantized representation of
each element
<span class="math notranslate nohighlight">\\\hat{w_i}\\</span> as
follows

<div class="math notranslate nohighlight">

\\\begin{split}\begin{aligned} \alpha &= \max_i w_i \\ \beta &= \min_i
w_i \\ s &= \frac{\alpha - \beta}{2^b - 1} \\ \hat{w_i} &=
\textrm{round}\left( \frac{w_i - \beta}{s}\right).
\end{aligned}\end{split}\\

</div>

After the above computation,
<span class="math notranslate nohighlight">\\\hat{w_i}\\</span> fits in
<span class="math notranslate nohighlight">\\b\\</span> bits and is
packed in an unsigned 32-bit integer from the lower to upper bits. For
instance, for 4-bit quantization we fit 8 elements in an unsigned 32 bit
integer where the 1st element occupies the 4 least significant bits, the
2nd bits 4-7 etc.

In order to be able to dequantize the elements of
<span class="pre">`w`</span> we also need to save
<span class="math notranslate nohighlight">\\s\\</span> and
<span class="math notranslate nohighlight">\\\beta\\</span> which are
the returned <span class="pre">`scales`</span> and
<span class="pre">`biases`</span> respectively.

Parameters<span class="colon">:</span>  
- **w** (<a
  href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.html#mlx.core.array"
  class="reference internal" title="mlx.core.array"><em>array</em></a>)
  – Matrix to be quantized

- **group_size**
  (<a href="https://docs.python.org/3/library/functions.html#int"
  class="reference external" title="(in Python v3.13)"><em>int</em></a>*,*
  *optional*) – The size of the group in <span class="pre">`w`</span>
  that shares a scale and bias. Default: <span class="pre">`64`</span>.

- **bits**
  (<a href="https://docs.python.org/3/library/functions.html#int"
  class="reference external" title="(in Python v3.13)"><em>int</em></a>*,*
  *optional*) – The number of bits occupied by each element of
  <span class="pre">`w`</span> in the returned quantized matrix.
  Default: <span class="pre">`4`</span>.

Returns<span class="colon">:</span>  
A tuple containing

- w_q (array): The quantized version of <span class="pre">`w`</span>

- scales (array): The scale to multiply each element with, namely
  <span class="math notranslate nohighlight">\\s\\</span>

- biases (array): The biases to add to each element, namely
  <span class="math notranslate nohighlight">\\\beta\\</span>

Return type<span class="colon">:</span>  
<a href="https://docs.python.org/3/library/stdtypes.html#tuple"
class="reference external" title="(in Python v3.13)"><em>tuple</em></a>

</div>

<div class="prev-next-area">

<a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.put_along_axis.html"
class="left-prev" title="previous page"><em></em></a>

<div class="prev-next-info">

previous

mlx.core.put_along_axis

</div>

<a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.quantized_matmul.html"
class="right-next" title="next page"></a>

<div class="prev-next-info">

next

mlx.core.quantized_matmul

</div>

</div>

</div>

<div class="bd-sidebar-secondary bd-toc">

<div class="sidebar-secondary-items sidebar-secondary__inner">

<div class="sidebar-secondary-item">

<div class="page-toc tocsection onthispage">

Contents

</div>

- <a href="https://ml-explore.github.io/mlx/build/html/#mlx.core.quantize"
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
