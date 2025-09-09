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
  href="https://ml-explore.github.io/mlx/build/html/_sources/python/_autosummary/mlx.core.custom_function.rst"
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

# mlx.core.custom_function

<div id="print-main-content">

<div id="jb-print-toc">

<div>

## Contents

</div>

- <a
  href="https://ml-explore.github.io/mlx/build/html/#mlx.core.custom_function"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">custom_function</code></span></a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/#mlx.core.custom_function.__init__"
    class="reference internal nav-link"><span class="pre"><code
    class="docutils literal notranslate">custom_function.__init__()</code></span></a>

</div>

</div>

</div>

<div id="searchbox">

</div>

<div id="mlx-core-custom-function" class="section">

# mlx.core.custom_function<a
href="https://ml-explore.github.io/mlx/build/html/#mlx-core-custom-function"
class="headerlink" title="Link to this heading">#</a>

*<span class="pre">class</span><span class="w"> </span>*<span class="sig-name descname"><span class="pre">custom_function</span></span><a
href="https://ml-explore.github.io/mlx/build/html/#mlx.core.custom_function"
class="headerlink" title="Link to this definition">#</a>  
Set up a function for custom gradient and vmap definitions.

This class is meant to be used as a function decorator. Instances are
callables that behave identically to the wrapped function. However, when
a function transformation is used (e.g. computing gradients using <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.value_and_grad.html#mlx.core.value_and_grad"
class="reference internal" title="mlx.core.value_and_grad"><span
class="pre"><code
class="sourceCode python">value_and_grad()</code></span></a>) then the
functions defined via <span class="pre">`custom_function.vjp()`</span>,
<span class="pre">`custom_function.jvp()`</span> and
<span class="pre">`custom_function.vmap()`</span> are used instead of
the default transformation.

Note, all custom transformations are optional. Undefined transformations
fall back to the default behaviour.

Example

<div class="highlight-python notranslate">

<div class="highlight">

    import mlx.core as mx

    @mx.custom_function
    def f(x, y):
        return mx.sin(x) * y

    @f.vjp
    def f_vjp(primals, cotangent, output):
        x, y = primals
        return cotan * mx.cos(x) * y, cotan * mx.sin(x)

    @f.jvp
    def f_jvp(primals, tangents):
      x, y = primals
      dx, dy = tangents
      return dx * mx.cos(x) * y + dy * mx.sin(x)

    @f.vmap
    def f_vmap(inputs, axes):
      x, y = inputs
      ax, ay = axes
      if ay != ax and ax is not None:
          y = y.swapaxes(ay, ax)
      return mx.sin(x) * y, (ax or ay)

</div>

</div>

All <span class="pre">`custom_function`</span> instances behave as pure
functions. Namely, any variables captured will be treated as constants
and no gradients will be computed with respect to the captured arrays.
For instance:

> <div>
>
> <div class="highlight-python notranslate">
>
> <div class="highlight">
>
>     import mlx.core as mx
>
>     def g(x, y):
>       @mx.custom_function
>       def f(x):
>         return x * y
>
>       @f.vjp
>       def f_vjp(x, dx, fx):
>         # Note that we have only x, dx and fx and nothing with respect to y
>         raise ValueError("Abort!")
>
>       return f(x)
>
>     x = mx.array(2.0)
>     y = mx.array(3.0)
>     print(g(x, y))                     # prints 6.0
>     print(mx.grad(g)(x, y))            # Raises exception
>     print(mx.grad(g, argnums=1)(x, y)) # prints 0.0
>
> </div>
>
> </div>
>
> </div>

<span class="sig-name descname"><span class="pre">\_\_init\_\_</span></span><span class="sig-paren">(</span>*<span class="n"><span class="pre">self</span></span>*, *<span class="n"><span class="pre">f</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><span class="pre">Callable</span></span>*<span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#mlx.core.custom_function.__init__"
class="headerlink" title="Link to this definition">#</a>  

Methods

<div class="pst-scrollable-table-container">

|  |  |
|----|----|
| <a
href="https://ml-explore.github.io/mlx/build/html/#mlx.core.custom_function.__init__"
class="reference internal"
title="mlx.core.custom_function.__init__"><span class="pre"><code
class="sourceCode python"><span class="fu">__init__</span></code></span></a>(self, f) |  |
| <span class="pre">`jvp`</span>(self, f) | Define a custom jvp for the wrapped function. |
| <span class="pre">`vjp`</span>(self, f) | Define a custom vjp for the wrapped function. |
| <span class="pre">`vmap`</span>(self, f) | Define a custom vectorization transformation for the wrapped function. |

</div>

</div>

<div class="prev-next-area">

<a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.compile.html"
class="left-prev" title="previous page"><em></em></a>

<div class="prev-next-info">

previous

mlx.core.compile

</div>

<a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.disable_compile.html"
class="right-next" title="next page"></a>

<div class="prev-next-info">

next

mlx.core.disable_compile

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
  href="https://ml-explore.github.io/mlx/build/html/#mlx.core.custom_function"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">custom_function</code></span></a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/#mlx.core.custom_function.__init__"
    class="reference internal nav-link"><span class="pre"><code
    class="docutils literal notranslate">custom_function.__init__()</code></span></a>

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
