Curator's note: Prefer the human-authored MLX guides for clarity.
- ../docs_curated/README.md
- ../docs_curated/PYTORCH_DISSONANCE.md
- ../docs_curated/NUMPY_USERS.md
- ../docs_curated/COMMON_PITFALLS.md

<!--
Per-file analysis (compile.md):
- Sphinx-converted overview of MLX compile; includes table-of-contents fragments and theme markup.
- Key user pitfalls: non-pure functions, shape changes across steps, and hidden Python-side mutation.
-->

## Curated Notes

- Keep compiled functions pure: no hidden mutation of globals or captured state.
- Stabilize shapes for best results; frequent shape changes reduce compile benefits.
- Validate numerics after enabling compile; small differences can appear due to fused kernels.
- Toggle quickly during bring-up: `mx.enable_compile()` / `mx.disable_compile()`.

## How It Works

- Function transform: `mx.compile(fn)` returns a compiled version of `fn` (you can also use the `@mx.compile` decorator).
- First-call compile: the first call traces and optimizes the graph, so it is slower; results are cached.
- Caching: subsequent calls are fast as long as input count, dtypes, ranks, and shapes match the cached signature.
- Recompilation triggers: any change to
  - number of inputs
  - dtype of an input
  - rank (ndim) of an input
  - shape of an input
  will cause a new compile.

## Basic Example (timed)

```python
import time
import mlx.core as mx

def run():
    x = mx.random.normal((2048, 2048))
    y = mx.random.normal((2048, 2048))
    return mx.sum(x @ y)

mx.disable_compile()
t0 = time.time(); _ = run(); mx.eval(_); t1 = time.time()

mx.enable_compile()
t2 = time.time(); _ = run(); mx.eval(_); t3 = time.time()

print("no-compile:", t1 - t0)
print("compile:  ", t3 - t2)
```

## Basic Example (math function)

```python
import mlx.core as mx
import time

def fun(x, y):
    return mx.exp(-x) + y

x = mx.array(1.0)
y = mx.array(2.0)

t0 = time.time(); r0 = fun(x, y); mx.eval(r0); t1 = time.time()
compiled_fun = mx.compile(fun)
t2 = time.time(); r1 = compiled_fun(x, y); mx.eval(r1); t3 = time.time()
t4 = time.time(); r2 = compiled_fun(x, y); mx.eval(r2); t5 = time.time()

print(f"Regular: {t1 - t0:.6f}s | First compiled: {t3 - t2:.6f}s | Cached: {t5 - t4:.6f}s")
print(r0, r2)
```

## Advanced Example (training step)

```python
import mlx.core as mx
import mlx.nn as nn
from mlx.optimizers import SGD

class MLP(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.l1 = nn.Linear(d_in, 256)
        self.l2 = nn.Linear(256, d_out)
    def __call__(self, x):
        x = nn.relu(self.l1(x))
        return self.l2(x)

model = MLP(10, 5)
params = model.parameters()
opt = SGD(1e-2)

def loss_fn(p, x, y):
    logits = model.apply(p, x)
    return mx.mean(nn.losses.cross_entropy(logits, y))

# Compile value_and_grad for the loss function
loss_and_grad = mx.compile(mx.value_and_grad(loss_fn))

x = mx.random.uniform(shape=(32, 10))
y = mx.random.randint(0, 5, (32,))

for step in range(5):
    loss, grads = loss_and_grad(params, x, y)
    params = opt.update(params, grads)
    mx.eval(loss)  # ensure compute
    if step == 0:
        print("Initial loss:", float(loss))
print("Final loss:", float(loss))
```

## Caveats and Best Practices

- Pure functions: compile only side‑effect‑free functions; avoid I/O (e.g., `print`) in compiled regions.
- Debugging: first call compiles; disable with `mx.disable_compile()` or `MLX_DISABLE_COMPILE=1` when needed.
- Control flow: input‑dependent branching can force recompiles; prefer shape‑stable, data‑independent control flow in compiled regions.


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
  href="https://ml-explore.github.io/mlx/build/html/_sources/usage/compile.rst"
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

# Compilation

<div id="print-main-content">

<div id="jb-print-toc">

<div>

## Contents

</div>

- <a href="https://ml-explore.github.io/mlx/build/html/#basics-of-compile"
  class="reference internal nav-link">Basics of Compile</a>
- <a href="https://ml-explore.github.io/mlx/build/html/#example-speedup"
  class="reference internal nav-link">Example Speedup</a>
- <a href="https://ml-explore.github.io/mlx/build/html/#debugging"
  class="reference internal nav-link">Debugging</a>
- <a href="https://ml-explore.github.io/mlx/build/html/#pure-functions"
  class="reference internal nav-link">Pure Functions</a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#compiling-training-graphs"
  class="reference internal nav-link">Compiling Training Graphs</a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#transformations-with-compile"
  class="reference internal nav-link">Transformations with Compile</a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#shapeless-compilation"
  class="reference internal nav-link">Shapeless Compilation</a>

</div>

</div>

</div>

<div id="searchbox">

</div>

<div id="compilation" class="section">

<span id="compile"></span>

# Compilation<a href="https://ml-explore.github.io/mlx/build/html/#compilation"
class="headerlink" title="Link to this heading">#</a>

MLX has a <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.compile.html#mlx.core.compile"
class="reference internal" title="mlx.core.compile"><span
class="pre"><code
class="sourceCode python"><span class="bu">compile</span>()</code></span></a>
function transformation which compiles computation graphs. Function
compilation results in smaller graphs by merging common work and fusing
certain operations. In many cases this can lead to big improvements in
run-time and memory use.

Getting started with <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.compile.html#mlx.core.compile"
class="reference internal" title="mlx.core.compile"><span
class="pre"><code
class="sourceCode python"><span class="bu">compile</span>()</code></span></a>
is simple, but there are some edge cases that are good to be aware of
for more complex graphs and advanced usage.

<div id="basics-of-compile" class="section">

## Basics of Compile<a href="https://ml-explore.github.io/mlx/build/html/#basics-of-compile"
class="headerlink" title="Link to this heading">#</a>

Let’s start with a simple example:

<div class="highlight-python notranslate">

<div class="highlight">

    def fun(x, y):
        return mx.exp(-x) + y

    x = mx.array(1.0)
    y = mx.array(2.0)

    # Regular call, no compilation
    # Prints: array(2.36788, dtype=float32)
    print(fun(x, y))

    # Compile the function
    compiled_fun = mx.compile(fun)

    # Prints: array(2.36788, dtype=float32)
    print(compiled_fun(x, y))

</div>

</div>

The output of both the regular function and the compiled function is the
same up to numerical precision.

The first time you call a compiled function, MLX will build the compute
graph, optimize it, and generate and compile code. This can be
relatively slow. However, MLX will cache compiled functions, so calling
a compiled function multiple times will not initiate a new compilation.
This means you should typically compile functions that you plan to use
more than once.

<div class="highlight-python notranslate">

<div class="highlight">

    def fun(x, y):
        return mx.exp(-x) + y

    x = mx.array(1.0)
    y = mx.array(2.0)

    compiled_fun = mx.compile(fun)

    # Compiled here
    compiled_fun(x, y)

    # Not compiled again
    compiled_fun(x, y)

    # Not compiled again
    mx.compile(fun)(x, y)

</div>

</div>

There are some important cases to be aware of that can cause a function
to be recompiled:

- Changing the shape or number of dimensions

- Changing the type of any of the inputs

- Changing the number of inputs to the function

In certain cases only some of the compilation stack will be rerun (for
example when changing the shapes) and in other cases the full
compilation stack will be rerun (for example when changing the types).
In general you should avoid compiling functions too frequently.

Another idiom to watch out for is compiling functions which get created
and destroyed frequently. This can happen, for example, when compiling
an anonymous function in a loop:

<div class="highlight-python notranslate">

<div class="highlight">

    a = mx.array(1.0)
    # Don't do this, compiles lambda at each iteration
    for _ in range(5):
        mx.compile(lambda x: mx.exp(mx.abs(x)))(a)

</div>

</div>

</div>

<div id="example-speedup" class="section">

## Example Speedup<a href="https://ml-explore.github.io/mlx/build/html/#example-speedup"
class="headerlink" title="Link to this heading">#</a>

The <a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary_functions/mlx.nn.gelu.html#mlx.nn.gelu"
class="reference internal" title="mlx.nn.gelu"><span class="pre"><code
class="sourceCode python">mlx.nn.gelu()</code></span></a> is a nonlinear
activation function commonly used with Transformer-based models. The
implementation involves several unary and binary element-wise
operations:

<div class="highlight-python notranslate">

<div class="highlight">

    def gelu(x):
        return x * (1 + mx.erf(x / math.sqrt(2))) / 2

</div>

</div>

If you use this function with small arrays, it will be overhead bound.
If you use it with large arrays it will be memory bandwidth bound.
However, all of the operations in the <span class="pre">`gelu`</span>
are fusible into a single kernel with <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.compile.html#mlx.core.compile"
class="reference internal" title="mlx.core.compile"><span
class="pre"><code
class="sourceCode python"><span class="bu">compile</span>()</code></span></a>.
This can speedup both cases considerably.

Let’s compare the runtime of the regular function versus the compiled
function. We’ll use the following timing helper which does a warm up and
handles synchronization:

<div class="highlight-python notranslate">

<div class="highlight">

    import time

    def timeit(fun, x):
        # warm up
        for _ in range(10):
            mx.eval(fun(x))

        tic = time.perf_counter()
        for _ in range(100):
            mx.eval(fun(x))
        toc = time.perf_counter()
        tpi = 1e3 * (toc - tic) / 100
        print(f"Time per iteration {tpi:.3f} (ms)")

</div>

</div>

Now make an array, and benchmark both functions:

<div class="highlight-python notranslate">

<div class="highlight">

    x = mx.random.uniform(shape=(32, 1000, 4096))
    timeit(nn.gelu, x)
    timeit(mx.compile(nn.gelu), x)

</div>

</div>

On an M1 Max the times are 15.5 and 3.1 milliseconds. The compiled
<span class="pre">`gelu`</span> is five times faster.

</div>

<div id="debugging" class="section">

## Debugging<a href="https://ml-explore.github.io/mlx/build/html/#debugging"
class="headerlink" title="Link to this heading">#</a>

When a compiled function is first called, it is traced with placeholder
inputs. This means you can’t evaluate arrays (for example to print their
contents) inside compiled functions.

<div class="highlight-python notranslate">

<div class="highlight">

    @mx.compile
    def fun(x):
        z = -x
        print(z)  # Crash
        return mx.exp(z)

    fun(mx.array(5.0))

</div>

</div>

For debugging, inspecting arrays can be helpful. One way to do that is
to globally disable compilation using the <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.disable_compile.html#mlx.core.disable_compile"
class="reference internal" title="mlx.core.disable_compile"><span
class="pre"><code
class="sourceCode python">disable_compile()</code></span></a> function
or <span class="pre">`MLX_DISABLE_COMPILE`</span> flag. For example the
following is okay even though <span class="pre">`fun`</span> is
compiled:

<div class="highlight-python notranslate">

<div class="highlight">

    @mx.compile
    def fun(x):
        z = -x
        print(z) # Okay
        return mx.exp(z)

    mx.disable_compile()
    fun(mx.array(5.0))

</div>

</div>

</div>

<div id="pure-functions" class="section">

## Pure Functions<a href="https://ml-explore.github.io/mlx/build/html/#pure-functions"
class="headerlink" title="Link to this heading">#</a>

Compiled functions are intended to be *pure*; that is they should not
have side effects. For example:

<div class="highlight-python notranslate">

<div class="highlight">

    state = []

    @mx.compile
    def fun(x, y):
        z = x + y
        state.append(z)
        return mx.exp(z)

    fun(mx.array(1.0), mx.array(2.0))
    # Crash!
    print(state)

</div>

</div>

After the first call of <span class="pre">`fun`</span>, the
<span class="pre">`state`</span> list will hold a placeholder array. The
placeholder does not have any data; it is only used to build the
computation graph. Printing such an array results in a crash.

You have two options to deal with this. The first option is to simply
return <span class="pre">`state`</span> as an output:

<div class="highlight-python notranslate">

<div class="highlight">

    state = []

    @mx.compile
    def fun(x, y):
       z = x + y
       state.append(z)
       return mx.exp(z), state

     _, state = fun(mx.array(1.0), mx.array(2.0))
     # Prints [array(3, dtype=float32)]
     print(state)

</div>

</div>

In some cases returning updated state can be pretty inconvenient. Hence,
<a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.compile.html#mlx.core.compile"
class="reference internal" title="mlx.core.compile"><span
class="pre"><code
class="sourceCode python"><span class="bu">compile</span>()</code></span></a>
has a parameter to capture implicit outputs:

<div class="highlight-python notranslate">

<div class="highlight">

    from functools import partial

    state = []

    # Tell compile to capture state as an output
    @partial(mx.compile, outputs=state)
    def fun(x, y):
        z = x + y
        state.append(z)
        return mx.exp(z), state

    fun(mx.array(1.0), mx.array(2.0))
    # Prints [array(3, dtype=float32)]
    print(state)

</div>

</div>

This is particularly useful for compiling a function which includes an
update to a container of arrays, as is commonly done when training the
parameters of a <a
href="https://ml-explore.github.io/mlx/build/html/python/nn/module.html#mlx.nn.Module"
class="reference internal" title="mlx.nn.Module"><span class="pre"><code
class="sourceCode python">mlx.nn.Module</code></span></a>.

Compiled functions will also treat any inputs not in the parameter list
as constants. For example:

<div class="highlight-python notranslate">

<div class="highlight">

    state = [mx.array(1.0)]

    @mx.compile
    def fun(x):
        return x + state[0]

    # Prints array(2, dtype=float32)
    print(fun(mx.array(1.0)))

    # Update state
    state[0] = mx.array(5.0)

    # Still prints array(2, dtype=float32)
    print(fun(mx.array(1.0)))

</div>

</div>

In order to have the change of state reflected in the outputs of
<span class="pre">`fun`</span> you again have two options. The first
option is to simply pass <span class="pre">`state`</span> as input to
the function. In some cases this can be pretty inconvenient. Hence, <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.compile.html#mlx.core.compile"
class="reference internal" title="mlx.core.compile"><span
class="pre"><code
class="sourceCode python"><span class="bu">compile</span>()</code></span></a>
also has a parameter to capture implicit inputs:

<div class="highlight-python notranslate">

<div class="highlight">

    from functools import partial
    state = [mx.array(1.0)]

    # Tell compile to capture state as an input
    @partial(mx.compile, inputs=state)
    def fun(x):
        return x + state[0]

    # Prints array(2, dtype=float32)
    print(fun(mx.array(1.0)))

    # Update state
    state[0] = mx.array(5.0)

    # Prints array(6, dtype=float32)
    print(fun(mx.array(1.0)))

</div>

</div>

</div>

<div id="compiling-training-graphs" class="section">

## Compiling Training Graphs<a
href="https://ml-explore.github.io/mlx/build/html/#compiling-training-graphs"
class="headerlink" title="Link to this heading">#</a>

This section will step through how to use <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.compile.html#mlx.core.compile"
class="reference internal" title="mlx.core.compile"><span
class="pre"><code
class="sourceCode python"><span class="bu">compile</span>()</code></span></a>
with a simple example of a common setup: training a model with <a
href="https://ml-explore.github.io/mlx/build/html/python/nn/module.html#mlx.nn.Module"
class="reference internal" title="mlx.nn.Module"><span class="pre"><code
class="sourceCode python">mlx.nn.Module</code></span></a> using an <a
href="https://ml-explore.github.io/mlx/build/html/python/optimizers/optimizer.html#mlx.optimizers.Optimizer"
class="reference internal" title="mlx.optimizers.Optimizer"><span
class="pre"><code
class="sourceCode python">mlx.optimizers.Optimizer</code></span></a>
with state. We will show how to compile the full forward, backward, and
update with <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.compile.html#mlx.core.compile"
class="reference internal" title="mlx.core.compile"><span
class="pre"><code
class="sourceCode python"><span class="bu">compile</span>()</code></span></a>.

To start, here is the simple example without any compilation:

<div class="highlight-python notranslate">

<div class="highlight">

    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim

    # 4 examples with 10 features each
    x = mx.random.uniform(shape=(4, 10))

    # 0, 1 targets
    y = mx.array([0, 1, 0, 1])

    # Simple linear model
    model = nn.Linear(10, 1)

    # SGD with momentum
    optimizer = optim.SGD(learning_rate=0.1, momentum=0.8)

    def loss_fn(model, x, y):
        logits = model(x).squeeze()
        return nn.losses.binary_cross_entropy(logits, y)

    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

    # Perform 10 steps of gradient descent
    for it in range(10):
        loss, grads = loss_and_grad_fn(model, x, y)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

</div>

</div>

To compile the update we can put it all in a function and compile it
with the appropriate input and output captures. Here’s the same example
but compiled:

<div class="highlight-python notranslate">

<div class="highlight">

    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from functools import partial

    # 4 examples with 10 features each
    x = mx.random.uniform(shape=(4, 10))

    # 0, 1 targets
    y = mx.array([0, 1, 0, 1])

    # Simple linear model
    model = nn.Linear(10, 1)

    # SGD with momentum
    optimizer = optim.SGD(learning_rate=0.1, momentum=0.8)

    def loss_fn(model, x, y):
        logits = model(x).squeeze()
        return nn.losses.binary_cross_entropy(logits, y)

    # The state that will be captured as input and output
    state = [model.state, optimizer.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def step(x, y):
        loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
        loss, grads = loss_and_grad_fn(model, x, y)
        optimizer.update(model, grads)
        return loss

    # Perform 10 steps of gradient descent
    for it in range(10):
        loss = step(x, y)
        # Evaluate the model and optimizer state
        mx.eval(state)
        print(loss)

</div>

</div>

<div class="admonition note">

Note

If you are using a module which performs random sampling such as <a
href="https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Dropout.html#mlx.nn.Dropout"
class="reference internal" title="mlx.nn.Dropout"><span
class="pre"><code
class="sourceCode python">mlx.nn.Dropout()</code></span></a>, make sure
you also include <span class="pre">`mx.random.state`</span> in the
<span class="pre">`state`</span> captured by <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.compile.html#mlx.core.compile"
class="reference internal" title="mlx.core.compile"><span
class="pre"><code
class="sourceCode python"><span class="bu">compile</span>()</code></span></a>,
i.e.
<span class="pre">`state`</span>` `<span class="pre">`=`</span>` `<span class="pre">`[model.state,`</span>` `<span class="pre">`optimizer.state,`</span>` `<span class="pre">`mx.random.state]`</span>.

</div>

<div class="admonition note">

Note

For more examples of compiling full training graphs checkout the
<a href="https://github.com/ml-explore/mlx-examples"
class="reference external">MLX Examples</a> GitHub repo.

</div>

</div>

<div id="transformations-with-compile" class="section">

## Transformations with Compile<a
href="https://ml-explore.github.io/mlx/build/html/#transformations-with-compile"
class="headerlink" title="Link to this heading">#</a>

In MLX function transformations are composable. You can apply any
function transformation to the output of any other function
transformation. For more on this, see the documentation on <a
href="https://ml-explore.github.io/mlx/build/html/usage/function_transforms.html#function-transforms"
class="reference internal"><span class="std std-ref">function
transforms</span></a>.

Compiling transformed functions works just as expected:

<div class="highlight-python notranslate">

<div class="highlight">

    grad_fn = mx.grad(mx.exp)

    compiled_grad_fn = mx.compile(grad_fn)

    # Prints: array(2.71828, dtype=float32)
    print(grad_fn(mx.array(1.0)))

    # Also prints: array(2.71828, dtype=float32)
    print(compiled_grad_fn(mx.array(1.0)))

</div>

</div>

<div class="admonition note">

Note

In order to compile as much as possible, a transformation of a compiled
function will not by default be compiled. To compile the transformed
function simply pass it through <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.compile.html#mlx.core.compile"
class="reference internal" title="mlx.core.compile"><span
class="pre"><code
class="sourceCode python"><span class="bu">compile</span>()</code></span></a>.

</div>

You can also compile functions which themselves call compiled functions.
A good practice is to compile the outer most function to give <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.compile.html#mlx.core.compile"
class="reference internal" title="mlx.core.compile"><span
class="pre"><code
class="sourceCode python"><span class="bu">compile</span>()</code></span></a>
the most opportunity to optimize the computation graph:

<div class="highlight-python notranslate">

<div class="highlight">

    @mx.compile
    def inner(x):
        return mx.exp(-mx.abs(x))

    def outer(x):
        inner(inner(x))

    # Compiling the outer function is good to do as it will likely
    # be faster even though the inner functions are compiled
    fun = mx.compile(outer)

</div>

</div>

</div>

<div id="shapeless-compilation" class="section">

<span id="shapeless-compile"></span>

## Shapeless Compilation<a
href="https://ml-explore.github.io/mlx/build/html/#shapeless-compilation"
class="headerlink" title="Link to this heading">#</a>

When the shape of an input to a compiled function changes, the function
is recompiled. You can compile a function once and run it on inputs with
variable shapes by specifying <span class="pre">`shapeless=True`</span>
to <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.compile.html#mlx.core.compile"
class="reference internal" title="mlx.core.compile"><span
class="pre"><code
class="sourceCode python"><span class="bu">compile</span>()</code></span></a>.
In this case changes to the shapes of the inputs do not cause the
function to be recompiled.

<div class="highlight-python notranslate">

<div class="highlight">

    def fun(x, y):
        return mx.abs(x + y)

    compiled_fun = mx.compile(fun, shapeless=True)

    x = mx.array(1.0)
    y = mx.array(-2.0)

    # Firt call compiles the function
    print(compiled_fun(x, y))

    # Second call with different shapes
    # does not recompile the function
    x = mx.array([1.0, -6.0])
    y = mx.array([-2.0, 3.0])
    print(compiled_fun(x, y))

</div>

</div>

Use shapeless compilations carefully. Since compilation is not triggered
when shapes change, any graphs which are conditional on the input shapes
will not work as expected. Shape-dependent computations are common and
sometimes subtle to detect. For example:

<div class="highlight-python notranslate">

<div class="highlight">

    def fun(x):
        return x.reshape(x.shape[0] * x.shape[1], -1)

    compiled_fun = mx.compile(fun, shapeless=True)

    x = mx.random.uniform(shape=(2, 3, 4))

    out = compiled_fun(x)

    x = mx.random.uniform(shape=(5, 5, 3))

    # Error, can't reshape (5, 5, 3) to (6, -1)
    out = compiled_fun(x)

</div>

</div>

The second call to the <span class="pre">`compiled_fun`</span> fails
because of the call to <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.reshape.html#mlx.core.reshape"
class="reference internal" title="mlx.core.reshape"><span
class="pre"><code class="sourceCode python">reshape()</code></span></a>
which uses the static shape of <span class="pre">`x`</span> in the first
call. We can fix this by using <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.flatten.html#mlx.core.flatten"
class="reference internal" title="mlx.core.flatten"><span
class="pre"><code class="sourceCode python">flatten()</code></span></a>
to avoid hardcoding the shape of <span class="pre">`x`</span>:

<div class="highlight-python notranslate">

<div class="highlight">

    def fun(x):
        return x.flatten(0, 1)

    compiled_fun = mx.compile(fun, shapeless=True)

    x = mx.random.uniform(shape=(2, 3, 4))

    out = compiled_fun(x)

    x = mx.random.uniform(shape=(5, 5, 3))

    # Ok
    out = compiled_fun(x)

</div>

</div>

</div>

</div>

<div class="prev-next-area">

<a
href="https://ml-explore.github.io/mlx/build/html/usage/function_transforms.html"
class="left-prev" title="previous page"><em></em></a>

<div class="prev-next-info">

previous

Function Transforms

</div>

<a href="https://ml-explore.github.io/mlx/build/html/usage/numpy.html"
class="right-next" title="next page"></a>

<div class="prev-next-info">

next

Conversion to NumPy and Other Frameworks

</div>

</div>

</div>

<div class="bd-sidebar-secondary bd-toc">

<div class="sidebar-secondary-items sidebar-secondary__inner">

<div class="sidebar-secondary-item">

<div class="page-toc tocsection onthispage">

Contents

</div>

- <a href="https://ml-explore.github.io/mlx/build/html/#basics-of-compile"
  class="reference internal nav-link">Basics of Compile</a>
- <a href="https://ml-explore.github.io/mlx/build/html/#example-speedup"
  class="reference internal nav-link">Example Speedup</a>
- <a href="https://ml-explore.github.io/mlx/build/html/#debugging"
  class="reference internal nav-link">Debugging</a>
- <a href="https://ml-explore.github.io/mlx/build/html/#pure-functions"
  class="reference internal nav-link">Pure Functions</a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#compiling-training-graphs"
  class="reference internal nav-link">Compiling Training Graphs</a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#transformations-with-compile"
  class="reference internal nav-link">Transformations with Compile</a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#shapeless-compilation"
  class="reference internal nav-link">Shapeless Compilation</a>

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
