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
  href="https://ml-explore.github.io/mlx/build/html/_sources/python/_autosummary/mlx.core.value_and_grad.rst"
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

# mlx.core.value_and_grad

<div id="print-main-content">

<div id="jb-print-toc">

<div>

## Contents

</div>

- <a
  href="https://ml-explore.github.io/mlx/build/html/#mlx.core.value_and_grad"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">value_and_grad()</code></span></a>

</div>

</div>

</div>

<div id="searchbox">

</div>

<div id="mlx-core-value-and-grad" class="section">

# mlx.core.value_and_grad<a
href="https://ml-explore.github.io/mlx/build/html/#mlx-core-value-and-grad"
class="headerlink" title="Link to this heading">#</a>

## Curated Notes

- Functional training: wrap a pure loss function; gradients are returned, not attached to arrays.
- Scalar loss: the first element of the return must be a scalar for gradients to be well‑defined.
- No in‑place updates: optimizers return new parameter trees; rebind your `params` variable.

### PyTorch mental shift

- Instead of `loss.backward(); optimizer.step()`, use:

```python
value, grads = mx.value_and_grad(loss_fn)(params, batch)
params = opt.update(params, grads)
```

### Example

```python
import mlx.core as mx
import mlx.nn as nn
from mlx.optimizers import AdamW

net = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 10))
params = net.parameters(); opt = AdamW(1e-3)

def loss_fn(p, x, y):
    logits = net.apply(p, x)
    return mx.mean((logits - y) ** 2)

for step in range(1000):
    x = mx.random.normal((128, 32)); y = mx.random.normal((128, 10))
    val, grads = mx.value_and_grad(loss_fn)(params, x, y)
    params = opt.update(params, grads)
```

<span class="sig-name descname"><span class="pre">value_and_grad</span></span><span class="sig-paren">(</span>*<span class="n"><span class="pre">fun</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><span class="pre">Callable</span></span>*, *<span class="n"><span class="pre">argnums</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><a href="https://docs.python.org/3/library/functions.html#int"
class="reference external" title="(in Python v3.13)"><span
class="pre">int</span></a><span class="w"> </span><span class="p"><span class="pre">\|</span></span><span class="w"> </span><span class="pre">Sequence</span><span class="p"><span class="pre">\[</span></span><a href="https://docs.python.org/3/library/functions.html#int"
class="reference external" title="(in Python v3.13)"><span
class="pre">int</span></a><span class="p"><span class="pre">\]</span></span><span class="w"> </span><span class="p"><span class="pre">\|</span></span><span class="w"> </span><a href="https://docs.python.org/3/library/constants.html#None"
class="reference external" title="(in Python v3.13)"><span
class="pre">None</span></a></span><span class="w"> </span><span class="o"><span class="pre">=</span></span><span class="w"> </span><span class="default_value"><span class="pre">None</span></span>*, *<span class="n"><span class="pre">argnames</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><a href="https://docs.python.org/3/library/stdtypes.html#str"
class="reference external" title="(in Python v3.13)"><span
class="pre">str</span></a><span class="w"> </span><span class="p"><span class="pre">\|</span></span><span class="w"> </span><span class="pre">Sequence</span><span class="p"><span class="pre">\[</span></span><a href="https://docs.python.org/3/library/stdtypes.html#str"
class="reference external" title="(in Python v3.13)"><span
class="pre">str</span></a><span class="p"><span class="pre">\]</span></span></span><span class="w"> </span><span class="o"><span class="pre">=</span></span><span class="w"> </span><span class="default_value"><span class="pre">\[\]</span></span>*<span class="sig-paren">)</span> <span class="sig-return"><span class="sig-return-icon">→</span> <span class="sig-return-typehint"><span class="pre">Callable</span></span></span><a
href="https://ml-explore.github.io/mlx/build/html/#mlx.core.value_and_grad"
class="headerlink" title="Link to this definition">#</a>  
Returns a function which computes the value and gradient of
<span class="pre">`fun`</span>.

The function passed to <a
href="https://ml-explore.github.io/mlx/build/html/#mlx.core.value_and_grad"
class="reference internal" title="mlx.core.value_and_grad"><span
class="pre"><code
class="sourceCode python">value_and_grad()</code></span></a> should
return either a scalar loss or a tuple in which the first element is a
scalar loss and the remaining elements can be anything.

<div class="highlight-python notranslate">

<div class="highlight">

    import mlx.core as mx

    def mse(params, inputs, targets):
        outputs = forward(params, inputs)
        lvalue = (outputs - targets).square().mean()
        return lvalue

    # Returns lvalue, dlvalue/dparams
    lvalue, grads = mx.value_and_grad(mse)(params, inputs, targets)

    def lasso(params, inputs, targets, a=1.0, b=1.0):
        outputs = forward(params, inputs)
        mse = (outputs - targets).square().mean()
        l1 = mx.abs(outputs - targets).mean()

        loss = a*mse + b*l1

        return loss, mse, l1

    (loss, mse, l1), grads = mx.value_and_grad(lasso)(params, inputs, targets)

</div>

</div>

Parameters<span class="colon">:</span>  
- **fun** (*Callable*) – A function which takes a variable number of <a
  href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.html#mlx.core.array"
  class="reference internal" title="mlx.core.array"><span
  class="pre"><code class="sourceCode python">array</code></span></a> or
  trees of <a
  href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.html#mlx.core.array"
  class="reference internal" title="mlx.core.array"><span
  class="pre"><code class="sourceCode python">array</code></span></a>
  and returns a scalar output <a
  href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.html#mlx.core.array"
  class="reference internal" title="mlx.core.array"><span
  class="pre"><code class="sourceCode python">array</code></span></a> or
  a tuple the first element of which should be a scalar <a
  href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.html#mlx.core.array"
  class="reference internal" title="mlx.core.array"><span
  class="pre"><code class="sourceCode python">array</code></span></a>.

- **argnums**
  (<a href="https://docs.python.org/3/library/functions.html#int"
  class="reference external" title="(in Python v3.13)"><em>int</em></a>
  *or* <a href="https://docs.python.org/3/library/stdtypes.html#list"
  class="reference external" title="(in Python v3.13)"><em>list</em></a>*(*<a href="https://docs.python.org/3/library/functions.html#int"
  class="reference external" title="(in Python v3.13)"><em>int</em></a>*),*
  *optional*) – Specify the index (or indices) of the positional
  arguments of <span class="pre">`fun`</span> to compute the gradient
  with respect to. If neither <span class="pre">`argnums`</span> nor
  <span class="pre">`argnames`</span> are provided
  <span class="pre">`argnums`</span> defaults to
  <span class="pre">`0`</span> indicating
  <span class="pre">`fun`</span>’s first argument.

- **argnames**
  (<a href="https://docs.python.org/3/library/stdtypes.html#str"
  class="reference external" title="(in Python v3.13)"><em>str</em></a>
  *or* <a href="https://docs.python.org/3/library/stdtypes.html#list"
  class="reference external" title="(in Python v3.13)"><em>list</em></a>*(*<a href="https://docs.python.org/3/library/stdtypes.html#str"
  class="reference external" title="(in Python v3.13)"><em>str</em></a>*),*
  *optional*) – Specify keyword arguments of
  <span class="pre">`fun`</span> to compute gradients with respect to.
  It defaults to \[\] so no gradients for keyword arguments by default.

Returns<span class="colon">:</span>  
A function which returns a tuple where the first element is the output
of fun and the second element is the gradients w.r.t. the loss.

Return type<span class="colon">:</span>  
*Callable*

</div>

<div class="prev-next-area">

<a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.grad.html"
class="left-prev" title="previous page"><em></em></a>

<div class="prev-next-info">

previous

mlx.core.grad

</div>

<a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.jvp.html"
class="right-next" title="next page"></a>

<div class="prev-next-info">

next

mlx.core.jvp

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
  href="https://ml-explore.github.io/mlx/build/html/#mlx.core.value_and_grad"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">value_and_grad()</code></span></a>

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
