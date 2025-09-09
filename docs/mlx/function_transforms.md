Curator's note: Prefer the human-authored MLX guides for clarity.
- ../docs_curated/README.md
- ../docs_curated/PYTORCH_DISSONANCE.md
- ../docs_curated/NUMPY_USERS.md
- ../docs_curated/COMMON_PITFALLS.md

<!--
Per-file analysis (function_transforms.md):
- Conceptual overview of grad/vmap/value_and_grad; includes interactive snippets.
- Users benefit from a minimal training-loop example and vmap axis hinting.
-->

## Curated Notes

- `mx.value_and_grad(fn)` is the workhorse for training: returns `(loss, grads)` in one pass.
- `mx.vmap(fn, in_axes=..., out_axes=...)` helps batch a pure function without writing loops; align axes explicitly.
- Compose transforms: `mx.vmap(mx.grad(fn))` is valid when `fn` is scalar‑valued per input.

### Examples

```python
import mlx.core as mx

# value_and_grad for training
def f(w, x):
    return mx.mean((x @ w) ** 2)
w = mx.random.normal((16,))
x = mx.random.normal((32, 16))
loss, grad = mx.value_and_grad(lambda w: f(w, x))(w)

# vmap over a scalar-valued function
def scalar_fn(a):
    return mx.sum(a * a)
batched = mx.vmap(scalar_fn)
A = mx.random.normal((8, 4))
vals = batched(A)  # shape (8,)

# Compose vmap and grad
df = mx.grad(scalar_fn)
df_batched = mx.vmap(df)
grads = df_batched(A)
```

## Function Basics (lazy, device‑agnostic)

- Define plain Python functions that operate on `mx.array` values; MLX records a graph lazily and executes on eval.
- No `device=` keyword on ops; control placement with default device/`stream`.

```python
import mlx.core as mx

def my_sum(x, y):
    return x + y

a = mx.array([1, 2, 3])
b = mx.array([4, 5, 6])
c = my_sum(a, b)
mx.eval(c)
print(c)  # array([5, 7, 9], dtype=int32)
```

## `mx.grad`

Returns a new function computing the gradient w.r.t. the first argument by default.

```python
import mlx.core as mx

def mse_loss(w, x, y):
    return mx.mean((w * x - y) ** 2)

grad_fn = mx.grad(mse_loss)
w = mx.array(1.0)
x = mx.array([0.5, -0.5])
y = mx.array([1.5, -1.5])
g = grad_fn(w, x, y)
mx.eval(g)
print(g)
```

## `mx.value_and_grad`

Computes value and gradient in one pass for efficiency.

```python
import mlx.core as mx

def mse_loss(w, x, y):
    return mx.mean((w * x - y) ** 2)

vag = mx.value_and_grad(mse_loss)
w = mx.array(1.0)
x = mx.array([0.5, -0.5])
y = mx.array([1.5, -1.5])
loss, grad_w = vag(w, x, y)
mx.eval(loss, grad_w)
print(loss, grad_w)
```

## `mx.vmap`

Vectorizes a function across batch dimensions without Python loops.

```python
import mlx.core as mx

def my_function(x):
    return x * 2

batched_x = mx.array([[1, 2], [3, 4]])
vf = mx.vmap(my_function)
out = vf(batched_x)
mx.eval(out)
print(out)
```

## `mx.compile`

JIT‑compiles a function; first call traces+compiles, subsequent calls with same signature are fast. See the Compile guide for benchmarks and caveats.

```python
import mlx.core as mx
import time

def fun(x, y):
    return mx.exp(-x) + y

x, y = mx.array(1.0), mx.array(2.0)
t0 = time.time(); r0 = fun(x, y); mx.eval(r0); t1 = time.time()
cf = mx.compile(fun)
t2 = time.time(); r1 = cf(x, y); mx.eval(r1); t3 = time.time()
t4 = time.time(); r2 = cf(x, y); mx.eval(r2); t5 = time.time()
print(f"regular={t1-t0:.6f}s first={t3-t2:.6f}s cached={t5-t4:.6f}s")
```

## `mx.custom_function` (advanced)

Wraps a function with custom differentiation rules (e.g., custom VJP/JVP) when the analytic reverse/forward is cheaper than the default. See Custom Extensions for end‑to‑end examples.


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
  href="https://ml-explore.github.io/mlx/build/html/_sources/usage/function_transforms.rst"
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

# Function Transforms

<div id="print-main-content">

<div id="jb-print-toc">

<div>

## Contents

</div>

- <a
  href="https://ml-explore.github.io/mlx/build/html/#automatic-differentiation"
  class="reference internal nav-link">Automatic Differentiation</a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#automatic-vectorization"
  class="reference internal nav-link">Automatic Vectorization</a>

</div>

</div>

</div>

<div id="searchbox">

</div>

<div id="function-transforms" class="section">

<span id="id1"></span>

# Function Transforms<a
href="https://ml-explore.github.io/mlx/build/html/#function-transforms"
class="headerlink" title="Link to this heading">#</a>

MLX uses composable function transformations for automatic
differentiation, vectorization, and compute graph optimizations. To see
the complete list of function transformations check-out the <a
href="https://ml-explore.github.io/mlx/build/html/python/transforms.html#transforms"
class="reference internal"><span class="std std-ref">API
documentation</span></a>.

The key idea behind composable function transformations is that every
transformation returns a function which can be further transformed.

Here is a simple example:

<div class="highlight-shell notranslate">

<div class="highlight">

    >>> dfdx = mx.grad(mx.sin)
    >>> dfdx(mx.array(mx.pi))
    array(-1, dtype=float32)
    >>> mx.cos(mx.array(mx.pi))
    array(-1, dtype=float32)

</div>

</div>

The output of <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.grad.html#mlx.core.grad"
class="reference internal" title="mlx.core.grad"><span class="pre"><code
class="sourceCode python">grad()</code></span></a> on <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.sin.html#mlx.core.sin"
class="reference internal" title="mlx.core.sin"><span class="pre"><code
class="sourceCode python">sin()</code></span></a> is simply another
function. In this case it is the gradient of the sine function which is
exactly the cosine function. To get the second derivative you can do:

<div class="highlight-shell notranslate">

<div class="highlight">

    >>> d2fdx2 = mx.grad(mx.grad(mx.sin))
    >>> d2fdx2(mx.array(mx.pi / 2))
    array(-1, dtype=float32)
    >>> mx.sin(mx.array(mx.pi / 2))
    array(1, dtype=float32)

</div>

</div>

Using <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.grad.html#mlx.core.grad"
class="reference internal" title="mlx.core.grad"><span class="pre"><code
class="sourceCode python">grad()</code></span></a> on the output of <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.grad.html#mlx.core.grad"
class="reference internal" title="mlx.core.grad"><span class="pre"><code
class="sourceCode python">grad()</code></span></a> is always ok. You
keep getting higher order derivatives.

Any of the MLX function transformations can be composed in any order to
any depth. See the following sections for more information on
<a href="https://ml-explore.github.io/mlx/build/html/#auto-diff"
class="reference internal"><span class="std std-ref">automatic
differentiation</span></a> and
<a href="https://ml-explore.github.io/mlx/build/html/#vmap"
class="reference internal"><span class="std std-ref">automatic
vectorization</span></a>. For more information on <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.compile.html#mlx.core.compile"
class="reference internal" title="mlx.core.compile"><span
class="pre"><code
class="sourceCode python"><span class="bu">compile</span>()</code></span></a>
see the <a
href="https://ml-explore.github.io/mlx/build/html/usage/compile.html#compile"
class="reference internal"><span class="std std-ref">compile
documentation</span></a>.

<div id="automatic-differentiation" class="section">

## Automatic Differentiation<a
href="https://ml-explore.github.io/mlx/build/html/#automatic-differentiation"
class="headerlink" title="Link to this heading">#</a>

Automatic differentiation in MLX works on functions rather than on
implicit graphs.

<div class="admonition note">

Note

If you are coming to MLX from PyTorch, you no longer need functions like
<span class="pre">`backward`</span>,
<span class="pre">`zero_grad`</span>, and
<span class="pre">`detach`</span>, or properties like
<span class="pre">`requires_grad`</span>.

</div>

The most basic example is taking the gradient of a scalar-valued
function as we saw above. You can use the <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.grad.html#mlx.core.grad"
class="reference internal" title="mlx.core.grad"><span class="pre"><code
class="sourceCode python">grad()</code></span></a> and <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.value_and_grad.html#mlx.core.value_and_grad"
class="reference internal" title="mlx.core.value_and_grad"><span
class="pre"><code
class="sourceCode python">value_and_grad()</code></span></a> function to
compute gradients of more complex functions. By default these functions
compute the gradient with respect to the first argument:

<div class="highlight-python notranslate">

<div class="highlight">

    def loss_fn(w, x, y):
       return mx.mean(mx.square(w * x - y))

    w = mx.array(1.0)
    x = mx.array([0.5, -0.5])
    y = mx.array([1.5, -1.5])

    # Computes the gradient of loss_fn with respect to w:
    grad_fn = mx.grad(loss_fn)
    dloss_dw = grad_fn(w, x, y)
    # Prints array(-1, dtype=float32)
    print(dloss_dw)

    # To get the gradient with respect to x we can do:
    grad_fn = mx.grad(loss_fn, argnums=1)
    dloss_dx = grad_fn(w, x, y)
    # Prints array([-1, 1], dtype=float32)
    print(dloss_dx)

</div>

</div>

One way to get the loss and gradient is to call
<span class="pre">`loss_fn`</span> followed by
<span class="pre">`grad_fn`</span>, but this can result in a lot of
redundant work. Instead, you should use <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.value_and_grad.html#mlx.core.value_and_grad"
class="reference internal" title="mlx.core.value_and_grad"><span
class="pre"><code
class="sourceCode python">value_and_grad()</code></span></a>. Continuing
the above example:

<div class="highlight-python notranslate">

<div class="highlight">

    # Computes the gradient of loss_fn with respect to w:
    loss_and_grad_fn = mx.value_and_grad(loss_fn)
    loss, dloss_dw = loss_and_grad_fn(w, x, y)

    # Prints array(1, dtype=float32)
    print(loss)

    # Prints array(-1, dtype=float32)
    print(dloss_dw)

</div>

</div>

You can also take the gradient with respect to arbitrarily nested Python
containers of arrays (specifically any of
<a href="https://docs.python.org/3/library/stdtypes.html#list"
class="reference external" title="(in Python v3.13)"><span
class="pre"><code
class="sourceCode python"><span class="bu">list</span></code></span></a>,
<a href="https://docs.python.org/3/library/stdtypes.html#tuple"
class="reference external" title="(in Python v3.13)"><span
class="pre"><code
class="sourceCode python"><span class="bu">tuple</span></code></span></a>,
or <a href="https://docs.python.org/3/library/stdtypes.html#dict"
class="reference external" title="(in Python v3.13)"><span
class="pre"><code
class="sourceCode python"><span class="bu">dict</span></code></span></a>).

Suppose we wanted a weight and a bias parameter in the above example. A
nice way to do that is the following:

<div class="highlight-python notranslate">

<div class="highlight">

    def loss_fn(params, x, y):
       w, b = params["weight"], params["bias"]
       h = w * x + b
       return mx.mean(mx.square(h - y))

    params = {"weight": mx.array(1.0), "bias": mx.array(0.0)}
    x = mx.array([0.5, -0.5])
    y = mx.array([1.5, -1.5])

    # Computes the gradient of loss_fn with respect to both the
    # weight and bias:
    grad_fn = mx.grad(loss_fn)
    grads = grad_fn(params, x, y)

    # Prints
    # {'weight': array(-1, dtype=float32), 'bias': array(0, dtype=float32)}
    print(grads)

</div>

</div>

Notice the tree structure of the parameters is preserved in the
gradients.

In some cases you may want to stop gradients from propagating through a
part of the function. You can use the <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.stop_gradient.html#mlx.core.stop_gradient"
class="reference internal" title="mlx.core.stop_gradient"><span
class="pre"><code
class="sourceCode python">stop_gradient()</code></span></a> for that.

</div>

<div id="automatic-vectorization" class="section">

## Automatic Vectorization<a
href="https://ml-explore.github.io/mlx/build/html/#automatic-vectorization"
class="headerlink" title="Link to this heading">#</a>

Use <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.vmap.html#mlx.core.vmap"
class="reference internal" title="mlx.core.vmap"><span class="pre"><code
class="sourceCode python">vmap()</code></span></a> to automate
vectorizing complex functions. Here we’ll go through a basic and
contrived example for the sake of clarity, but <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.vmap.html#mlx.core.vmap"
class="reference internal" title="mlx.core.vmap"><span class="pre"><code
class="sourceCode python">vmap()</code></span></a> can be quite powerful
for more complex functions which are difficult to optimize by hand.

<div class="admonition warning">

Warning

Some operations are not yet supported with <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.vmap.html#mlx.core.vmap"
class="reference internal" title="mlx.core.vmap"><span class="pre"><code
class="sourceCode python">vmap()</code></span></a>. If you encounter an
error like:
<span class="pre">`ValueError:`</span>` `<span class="pre">`Primitive's`</span>` `<span class="pre">`vmap`</span>` `<span class="pre">`not`</span>` `<span class="pre">`implemented.`</span>
file an <a href="https://github.com/ml-explore/mlx/issues"
class="reference external">issue</a> and include your function. We will
prioritize including it.

</div>

A naive way to add the elements from two sets of vectors is with a loop:

<div class="highlight-python notranslate">

<div class="highlight">

    xs = mx.random.uniform(shape=(4096, 100))
    ys = mx.random.uniform(shape=(100, 4096))

    def naive_add(xs, ys):
        return [xs[i] + ys[:, i] for i in range(xs.shape[0])]

</div>

</div>

Instead you can use <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.vmap.html#mlx.core.vmap"
class="reference internal" title="mlx.core.vmap"><span class="pre"><code
class="sourceCode python">vmap()</code></span></a> to automatically
vectorize the addition:

<div class="highlight-python notranslate">

<div class="highlight">

    # Vectorize over the second dimension of x and the
    # first dimension of y
    vmap_add = mx.vmap(lambda x, y: x + y, in_axes=(0, 1))

</div>

</div>

The <span class="pre">`in_axes`</span> parameter can be used to specify
which dimensions of the corresponding input to vectorize over.
Similarly, use <span class="pre">`out_axes`</span> to specify where the
vectorized axes should be in the outputs.

Let’s time these two different versions:

<div class="highlight-python notranslate">

<div class="highlight">

    import timeit

    print(timeit.timeit(lambda: mx.eval(naive_add(xs, ys)), number=100))
    print(timeit.timeit(lambda: mx.eval(vmap_add(xs, ys)), number=100))

</div>

</div>

On an M1 Max the naive version takes in total
<span class="pre">`5.639`</span> seconds whereas the vectorized version
takes only <span class="pre">`0.024`</span> seconds, more than 200 times
faster.

Of course, this operation is quite contrived. A better approach is to
simply do
<span class="pre">`xs`</span>` `<span class="pre">`+`</span>` `<span class="pre">`ys.T`</span>,
but for more complex functions <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.vmap.html#mlx.core.vmap"
class="reference internal" title="mlx.core.vmap"><span class="pre"><code
class="sourceCode python">vmap()</code></span></a> can be quite handy.

</div>

</div>

<div class="prev-next-area">

<a
href="https://ml-explore.github.io/mlx/build/html/usage/saving_and_loading.html"
class="left-prev" title="previous page"><em></em></a>

<div class="prev-next-info">

previous

Saving and Loading Arrays

</div>

<a href="https://ml-explore.github.io/mlx/build/html/usage/compile.html"
class="right-next" title="next page"></a>

<div class="prev-next-info">

next

Compilation

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
  href="https://ml-explore.github.io/mlx/build/html/#automatic-differentiation"
  class="reference internal nav-link">Automatic Differentiation</a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#automatic-vectorization"
  class="reference internal nav-link">Automatic Vectorization</a>

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
