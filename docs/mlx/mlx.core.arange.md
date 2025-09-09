Curator's note: Prefer the human-authored MLX guides for clarity.
- ../docs_curated/README.md
- ../docs_curated/PYTORCH_DISSONANCE.md
- ../docs_curated/NUMPY_USERS.md
- ../docs_curated/COMMON_PITFALLS.md

<!--
Per-file analysis (mlx.core.arange.md):
- Range creation like NumPy; supports int/float step and dtype.
- Add notes on exclusivity, step sign, and float stepping pitfalls.
-->

## Curated Notes

- Stop is exclusive; negative steps count down: `mx.arange(5, 0, -1) -> [5,4,3,2,1]`.
- Floating steps can accumulate rounding error; prefer `mx.linspace` when you need an exact number of points.
- Specify `dtype` to avoid unintended up/down‑casting, especially when mixing ints and floats.

### Examples

```python
import mlx.core as mx

print(mx.arange(5))            # [0,1,2,3,4]
print(mx.arange(2, 6))         # [2,3,4,5]
print(mx.arange(5, 0, -2))     # [5,3,1]

# Floating step vs linspace
f = mx.arange(0.0, 1.0, 0.2)   # may have rounding artifacts
g = mx.linspace(0.0, 1.0, 6)   # exact count of points
print(f, g)

# Dtype control
i16 = mx.arange(0, 5, dtype=mx.int16)
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
  href="https://ml-explore.github.io/mlx/build/html/_sources/python/_autosummary/mlx.core.arange.rst"
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

# mlx.core.arange

<div id="print-main-content">

<div id="jb-print-toc">

<div>

## Contents

</div>

- <a href="https://ml-explore.github.io/mlx/build/html/#mlx.core.arange"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">arange()</code></span></a>

</div>

</div>

</div>

<div id="searchbox">

</div>

<div id="mlx-core-arange" class="section">

# mlx.core.arange<a href="https://ml-explore.github.io/mlx/build/html/#mlx-core-arange"
class="headerlink" title="Link to this heading">#</a>

<span class="sig-name descname"><span class="pre">arange</span></span><span class="sig-paren">(</span>*<span class="n"><span class="pre">start</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><a href="https://docs.python.org/3/library/functions.html#int"
class="reference external" title="(in Python v3.13)"><span
class="pre">int</span></a><span class="w"> </span><span class="p"><span class="pre">\|</span></span><span class="w"> </span><a href="https://docs.python.org/3/library/functions.html#float"
class="reference external" title="(in Python v3.13)"><span
class="pre">float</span></a></span>*, *<span class="n"><span class="pre">stop</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><a href="https://docs.python.org/3/library/functions.html#int"
class="reference external" title="(in Python v3.13)"><span
class="pre">int</span></a><span class="w"> </span><span class="p"><span class="pre">\|</span></span><span class="w"> </span><a href="https://docs.python.org/3/library/functions.html#float"
class="reference external" title="(in Python v3.13)"><span
class="pre">float</span></a></span>*, *<span class="n"><span class="pre">step</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><a href="https://docs.python.org/3/library/constants.html#None"
class="reference external" title="(in Python v3.13)"><span
class="pre">None</span></a><span class="w"> </span><span class="p"><span class="pre">\|</span></span><span class="w"> </span><a href="https://docs.python.org/3/library/functions.html#int"
class="reference external" title="(in Python v3.13)"><span
class="pre">int</span></a><span class="w"> </span><span class="p"><span class="pre">\|</span></span><span class="w"> </span><a href="https://docs.python.org/3/library/functions.html#float"
class="reference external" title="(in Python v3.13)"><span
class="pre">float</span></a></span>*, *<span class="n"><span class="pre">dtype</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.Dtype.html#mlx.core.Dtype"
class="reference internal" title="mlx.core.Dtype"><span
class="pre">Dtype</span></a><span class="w"> </span><span class="p"><span class="pre">\|</span></span><span class="w"> </span><a href="https://docs.python.org/3/library/constants.html#None"
class="reference external" title="(in Python v3.13)"><span
class="pre">None</span></a></span><span class="w"> </span><span class="o"><span class="pre">=</span></span><span class="w"> </span><span class="default_value"><span class="pre">None</span></span>*, *<span class="o"><span class="pre">\*</span></span>*, *<span class="n"><span class="pre">stream</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><a href="https://docs.python.org/3/library/constants.html#None"
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
class="pre">array</span></a></span></span><a href="https://ml-explore.github.io/mlx/build/html/#mlx.core.arange"
class="headerlink" title="Link to this definition">#</a>  
<span class="sig-name descname"><span class="pre">arange</span></span><span class="sig-paren">(</span>*<span class="n"><span class="pre">stop</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><a href="https://docs.python.org/3/library/functions.html#int"
class="reference external" title="(in Python v3.13)"><span
class="pre">int</span></a><span class="w"> </span><span class="p"><span class="pre">\|</span></span><span class="w"> </span><a href="https://docs.python.org/3/library/functions.html#float"
class="reference external" title="(in Python v3.13)"><span
class="pre">float</span></a></span>*, *<span class="n"><span class="pre">step</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><a href="https://docs.python.org/3/library/constants.html#None"
class="reference external" title="(in Python v3.13)"><span
class="pre">None</span></a><span class="w"> </span><span class="p"><span class="pre">\|</span></span><span class="w"> </span><a href="https://docs.python.org/3/library/functions.html#int"
class="reference external" title="(in Python v3.13)"><span
class="pre">int</span></a><span class="w"> </span><span class="p"><span class="pre">\|</span></span><span class="w"> </span><a href="https://docs.python.org/3/library/functions.html#float"
class="reference external" title="(in Python v3.13)"><span
class="pre">float</span></a></span><span class="w"> </span><span class="o"><span class="pre">=</span></span><span class="w"> </span><span class="default_value"><span class="pre">None</span></span>*, *<span class="n"><span class="pre">dtype</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.Dtype.html#mlx.core.Dtype"
class="reference internal" title="mlx.core.Dtype"><span
class="pre">Dtype</span></a><span class="w"> </span><span class="p"><span class="pre">\|</span></span><span class="w"> </span><a href="https://docs.python.org/3/library/constants.html#None"
class="reference external" title="(in Python v3.13)"><span
class="pre">None</span></a></span><span class="w"> </span><span class="o"><span class="pre">=</span></span><span class="w"> </span><span class="default_value"><span class="pre">None</span></span>*, *<span class="o"><span class="pre">\*</span></span>*, *<span class="n"><span class="pre">stream</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><a href="https://docs.python.org/3/library/constants.html#None"
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
class="pre">array</span></a></span></span>  
Overloaded function.

1.  <span class="pre">`arange(start`</span>` `<span class="pre">`:`</span>` `<span class="pre">`Union[int,`</span>` `<span class="pre">`float],`</span>` `<span class="pre">`stop`</span>` `<span class="pre">`:`</span>` `<span class="pre">`Union[int,`</span>` `<span class="pre">`float],`</span>` `<span class="pre">`step`</span>` `<span class="pre">`:`</span>` `<span class="pre">`Union[None,`</span>` `<span class="pre">`int,`</span>` `<span class="pre">`float],`</span>` `<span class="pre">`dtype:`</span>` `<span class="pre">`Optional[Dtype]`</span>` `<span class="pre">`=`</span>` `<span class="pre">`None,`</span>` `<span class="pre">`*,`</span>` `<span class="pre">`stream:`</span>` `<span class="pre">`Union[None,`</span>` `<span class="pre">`Stream,`</span>` `<span class="pre">`Device]`</span>` `<span class="pre">`=`</span>` `<span class="pre">`None)`</span>` `<span class="pre">`->`</span>` `<span class="pre">`array`</span>

    > <div>
    >
    > Generates ranges of numbers.
    >
    > Generate numbers in the half-open interval
    > <span class="pre">`[start,`</span>` `<span class="pre">`stop)`</span>
    > in increments of <span class="pre">`step`</span>.
    >
    > Args:  
    > start (float or int, optional): Starting value which defaults to
    > <span class="pre">`0`</span>. stop (float or int): Stopping value.
    > step (float or int, optional): Increment which defaults to
    > <span class="pre">`1`</span>. dtype (Dtype, optional): Specifies
    > the data type of the output. If unspecified will default to
    > <span class="pre">`float32`</span> if any of
    > <span class="pre">`start`</span>, <span class="pre">`stop`</span>,
    > or <span class="pre">`step`</span> are
    > <span class="pre">`float`</span>. Otherwise will default to
    > <span class="pre">`int32`</span>.
    >
    > Returns:  
    > array: The range of values.
    >
    > Note:  
    > Following the Numpy convention the actual increment used to
    > generate numbers is
    > <span class="pre">`dtype(start`</span>` `<span class="pre">`+`</span>` `<span class="pre">`step)`</span>` `<span class="pre">`-`</span>` `<span class="pre">`dtype(start)`</span>.
    > This can lead to unexpected results for example if start + step is
    > a fractional value and the dtype is integral.
    >
    > </div>

2.  <span class="pre">`arange(stop`</span>` `<span class="pre">`:`</span>` `<span class="pre">`Union[int,`</span>` `<span class="pre">`float],`</span>` `<span class="pre">`step`</span>` `<span class="pre">`:`</span>` `<span class="pre">`Union[None,`</span>` `<span class="pre">`int,`</span>` `<span class="pre">`float]`</span>` `<span class="pre">`=`</span>` `<span class="pre">`None,`</span>` `<span class="pre">`dtype:`</span>` `<span class="pre">`Optional[Dtype]`</span>` `<span class="pre">`=`</span>` `<span class="pre">`None,`</span>` `<span class="pre">`*,`</span>` `<span class="pre">`stream:`</span>` `<span class="pre">`Union[None,`</span>` `<span class="pre">`Stream,`</span>` `<span class="pre">`Device]`</span>` `<span class="pre">`=`</span>` `<span class="pre">`None)`</span>` `<span class="pre">`->`</span>` `<span class="pre">`array`</span>

</div>

<div class="prev-next-area">

<a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.any.html"
class="left-prev" title="previous page"><em></em></a>

<div class="prev-next-info">

previous

mlx.core.any

</div>

<a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.arccos.html"
class="right-next" title="next page"></a>

<div class="prev-next-info">

next

mlx.core.arccos

</div>

</div>

</div>

<div class="bd-sidebar-secondary bd-toc">

<div class="sidebar-secondary-items sidebar-secondary__inner">

<div class="sidebar-secondary-item">

<div class="page-toc tocsection onthispage">

Contents

</div>

- <a href="https://ml-explore.github.io/mlx/build/html/#mlx.core.arange"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">arange()</code></span></a>

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
