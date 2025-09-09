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
  href="https://ml-explore.github.io/mlx/build/html/_sources/python/nn/_autosummary/mlx.nn.Upsample.rst"
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

# mlx.nn.Upsample

<div id="print-main-content">

<div id="jb-print-toc">

<div>

## Contents

</div>

- <a href="https://ml-explore.github.io/mlx/build/html/#mlx.nn.Upsample"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">Upsample</code></span></a>

</div>

</div>

</div>

<div id="searchbox">

</div>

<div id="mlx-nn-upsample" class="section">

# mlx.nn.Upsample<a href="https://ml-explore.github.io/mlx/build/html/#mlx-nn-upsample"
class="headerlink" title="Link to this heading">#</a>

*<span class="pre">class</span><span class="w"> </span>*<span class="sig-name descname"><span class="pre">Upsample</span></span><span class="sig-paren">(</span>*<span class="n"><span class="pre">scale_factor</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><a href="https://docs.python.org/3/library/functions.html#float"
class="reference external" title="(in Python v3.13)"><span
class="pre">float</span></a><span class="w"> </span><span class="p"><span class="pre">\|</span></span><span class="w"> </span><a href="https://docs.python.org/3/library/typing.html#typing.Tuple"
class="reference external" title="(in Python v3.13)"><span
class="pre">Tuple</span></a></span>*, *<span class="n"><span class="pre">mode</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><a href="https://docs.python.org/3/library/typing.html#typing.Literal"
class="reference external" title="(in Python v3.13)"><span
class="pre">Literal</span></a><span class="p"><span class="pre">\[</span></span><span class="s"><span class="pre">'nearest'</span></span><span class="p"><span class="pre">,</span></span><span class="w"> </span><span class="s"><span class="pre">'linear'</span></span><span class="p"><span class="pre">,</span></span><span class="w"> </span><span class="s"><span class="pre">'cubic'</span></span><span class="p"><span class="pre">\]</span></span></span><span class="w"> </span><span class="o"><span class="pre">=</span></span><span class="w"> </span><span class="default_value"><span class="pre">'nearest'</span></span>*, *<span class="n"><span class="pre">align_corners</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><a href="https://docs.python.org/3/library/functions.html#bool"
class="reference external" title="(in Python v3.13)"><span
class="pre">bool</span></a></span><span class="w"> </span><span class="o"><span class="pre">=</span></span><span class="w"> </span><span class="default_value"><span class="pre">False</span></span>*<span class="sig-paren">)</span><a href="https://ml-explore.github.io/mlx/build/html/#mlx.nn.Upsample"
class="headerlink" title="Link to this definition">#</a>  
Upsample the input signal spatially.

The spatial dimensions are by convention dimensions
<span class="pre">`1`</span> to
<span class="pre">`x.ndim`</span>` `<span class="pre">`-`</span>` `<span class="pre">`2`</span>.
The first is the batch dimension and the last is the feature dimension.

For example, an audio signal would be 3D with 1 spatial dimension, an
image 4D with 2 and so on and so forth.

There are three upsampling algorithms implemented nearest neighbor
upsampling, linear interpolation, and cubic interpolation. All can be
applied to any number of spatial dimensions. The linear interpolation
will be bilinear, trilinear etc when applied to more than one spatial
dimension. And cubic interpolation will be bicubic when there are 2
spatial dimensions.

<div class="admonition note">

Note

When using one of the linear or cubic interpolation modes the
<span class="pre">`align_corners`</span> argument changes how the
corners are treated in the input image. If
<span class="pre">`align_corners=True`</span> then the top and left edge
of the input and output will be matching as will the bottom right edge.

</div>

Parameters<span class="colon">:</span>  
- **scale_factor**
  (<a href="https://docs.python.org/3/library/functions.html#float"
  class="reference external" title="(in Python v3.13)"><em>float</em></a>
  *or* <a href="https://docs.python.org/3/library/stdtypes.html#tuple"
  class="reference external" title="(in Python v3.13)"><em>tuple</em></a>)
  – The multiplier for the spatial size. If a
  <span class="pre">`float`</span> is provided, it is the multiplier for
  all spatial dimensions. Otherwise, the number of scale factors
  provided must match the number of spatial dimensions.

- **mode**
  (<a href="https://docs.python.org/3/library/stdtypes.html#str"
  class="reference external" title="(in Python v3.13)"><em>str</em></a>*,*
  *optional*) – The upsampling algorithm, either
  <span class="pre">`"nearest"`</span>,
  <span class="pre">`"linear"`</span> or
  <span class="pre">`"cubic"`</span>. Default:
  <span class="pre">`"nearest"`</span>.

- **align_corners**
  (<a href="https://docs.python.org/3/library/functions.html#bool"
  class="reference external" title="(in Python v3.13)"><em>bool</em></a>*,*
  *optional*) – Changes the way the corners are treated during
  <span class="pre">`"linear"`</span> and
  <span class="pre">`"cubic"`</span> upsampling. See the note above and
  the examples below for more details. Default:
  <span class="pre">`False`</span>.

Examples

<div class="doctest highlight-default notranslate">

<div class="highlight">

    >>> import mlx.core as mx
    >>> import mlx.nn as nn
    >>> x = mx.arange(1, 5).reshape((1, 2, 2, 1))
    >>> x
    array([[[[1],
             [2]],
            [[3],
             [4]]]], dtype=int32)
    >>> n = nn.Upsample(scale_factor=2, mode='nearest')
    >>> n(x).squeeze()
    array([[1, 1, 2, 2],
           [1, 1, 2, 2],
           [3, 3, 4, 4],
           [3, 3, 4, 4]], dtype=int32)
    >>> b = nn.Upsample(scale_factor=2, mode='linear')
    >>> b(x).squeeze()
    array([[1, 1.25, 1.75, 2],
           [1.5, 1.75, 2.25, 2.5],
           [2.5, 2.75, 3.25, 3.5],
           [3, 3.25, 3.75, 4]], dtype=float32)
    >>> b = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
    >>> b(x).squeeze()
    array([[1, 1.33333, 1.66667, 2],
           [1.66667, 2, 2.33333, 2.66667],
           [2.33333, 2.66667, 3, 3.33333],
           [3, 3.33333, 3.66667, 4]], dtype=float32)

</div>

</div>

Methods

<div class="pst-scrollable-table-container">

</div>

</div>

<div class="prev-next-area">

<a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Transformer.html"
class="left-prev" title="previous page"><em></em></a>

<div class="prev-next-info">

previous

mlx.nn.Transformer

</div>

<a
href="https://ml-explore.github.io/mlx/build/html/python/nn/functions.html"
class="right-next" title="next page"></a>

<div class="prev-next-info">

next

Functions

</div>

</div>

</div>

<div class="bd-sidebar-secondary bd-toc">

<div class="sidebar-secondary-items sidebar-secondary__inner">

<div class="sidebar-secondary-item">

<div class="page-toc tocsection onthispage">

Contents

</div>

- <a href="https://ml-explore.github.io/mlx/build/html/#mlx.nn.Upsample"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">Upsample</code></span></a>

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
