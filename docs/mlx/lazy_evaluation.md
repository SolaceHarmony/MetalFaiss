Curator's note: Prefer the human-authored MLX guides for clarity.
- ../docs_curated/README.md
- ../docs_curated/PYTORCH_DISSONANCE.md
- ../docs_curated/NUMPY_USERS.md
- ../docs_curated/COMMON_PITFALLS.md

<!--
Per-file analysis (lazy_evaluation.md):
- Explains MLX’s lazy execution and eval points.
- Add practical guidance on forcing eval and debugging graph issues.
-->

## Curated Notes

- Force materialization when needed: `mx.eval(x)` or by calling `.item()` on scalars.
- When profiling, insert evals at boundaries to isolate segments; remove them for final performance.
- Be cautious with side‑effects depending on timing; rely on pure functions to avoid surprises.
- Framework boundaries: converting to NumPy and mutating via a view is out‑of‑graph and breaks lazy optimizations/gradients. Prefer to keep compute in MLX; if interop is required, copy at the boundary (`np.array(x, copy=True)` and `mx.array(np_arr.copy())`).


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
  href="https://ml-explore.github.io/mlx/build/html/_sources/usage/lazy_evaluation.rst"
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

# Lazy Evaluation

<div id="print-main-content">

<div id="jb-print-toc">

<div>

## Contents

</div>

- <a
  href="https://ml-explore.github.io/mlx/build/html/#why-lazy-evaluation"
  class="reference internal nav-link">Why Lazy Evaluation</a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/#transforming-compute-graphs"
    class="reference internal nav-link">Transforming Compute Graphs</a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/#only-compute-what-you-use"
    class="reference internal nav-link">Only Compute What You Use</a>
- <a href="https://ml-explore.github.io/mlx/build/html/#when-to-evaluate"
  class="reference internal nav-link">When to Evaluate</a>

</div>

</div>

</div>

<div id="searchbox">

</div>

<div id="lazy-evaluation" class="section">

<span id="lazy-eval"></span>

# Lazy Evaluation<a href="https://ml-explore.github.io/mlx/build/html/#lazy-evaluation"
class="headerlink" title="Link to this heading">#</a>

<div id="why-lazy-evaluation" class="section">

## Why Lazy Evaluation<a
href="https://ml-explore.github.io/mlx/build/html/#why-lazy-evaluation"
class="headerlink" title="Link to this heading">#</a>

When you perform operations in MLX, no computation actually happens.
Instead a compute graph is recorded. The actual computation only happens
if an <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.eval.html#mlx.core.eval"
class="reference internal" title="mlx.core.eval"><span class="pre"><code
class="sourceCode python"><span class="bu">eval</span>()</code></span></a>
is performed.

MLX uses lazy evaluation because it has some nice features, some of
which we describe below.

<div id="transforming-compute-graphs" class="section">

### Transforming Compute Graphs<a
href="https://ml-explore.github.io/mlx/build/html/#transforming-compute-graphs"
class="headerlink" title="Link to this heading">#</a>

Lazy evaluation lets us record a compute graph without actually doing
any computations. This is useful for function transformations like <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.grad.html#mlx.core.grad"
class="reference internal" title="mlx.core.grad"><span class="pre"><code
class="sourceCode python">grad()</code></span></a> and <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.vmap.html#mlx.core.vmap"
class="reference internal" title="mlx.core.vmap"><span class="pre"><code
class="sourceCode python">vmap()</code></span></a> and graph
optimizations.

Currently, MLX does not compile and rerun compute graphs. They are all
generated dynamically. However, lazy evaluation makes it much easier to
integrate compilation for future performance enhancements.

</div>

<div id="only-compute-what-you-use" class="section">

### Only Compute What You Use<a
href="https://ml-explore.github.io/mlx/build/html/#only-compute-what-you-use"
class="headerlink" title="Link to this heading">#</a>

In MLX you do not need to worry as much about computing outputs that are
never used. For example:

<div class="highlight-python notranslate">

<div class="highlight">

    def fun(x):
        a = fun1(x)
        b = expensive_fun(a)
        return a, b

    y, _ = fun(x)

</div>

</div>

Here, we never actually compute the output of
<span class="pre">`expensive_fun`</span>. Use this pattern with care
though, as the graph of <span class="pre">`expensive_fun`</span> is
still built, and that has some cost associated to it.

Similarly, lazy evaluation can be beneficial for saving memory while
keeping code simple. Say you have a very large model
<span class="pre">`Model`</span> derived from <a
href="https://ml-explore.github.io/mlx/build/html/python/nn/module.html#mlx.nn.Module"
class="reference internal" title="mlx.nn.Module"><span class="pre"><code
class="sourceCode python">mlx.nn.Module</code></span></a>. You can
instantiate this model with
<span class="pre">`model`</span>` `<span class="pre">`=`</span>` `<span class="pre">`Model()`</span>.
Typically, this will initialize all of the weights as
<span class="pre">`float32`</span>, but the initialization does not
actually compute anything until you perform an <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.eval.html#mlx.core.eval"
class="reference internal" title="mlx.core.eval"><span class="pre"><code
class="sourceCode python"><span class="bu">eval</span>()</code></span></a>.
If you update the model with <span class="pre">`float16`</span> weights,
your maximum consumed memory will be half that required if eager
computation was used instead.

This pattern is simple to do in MLX thanks to lazy computation:

<div class="highlight-python notranslate">

<div class="highlight">

    model = Model() # no memory used yet
    model.load_weights("weights_fp16.safetensors")

</div>

</div>

</div>

</div>

<div id="when-to-evaluate" class="section">

## When to Evaluate<a href="https://ml-explore.github.io/mlx/build/html/#when-to-evaluate"
class="headerlink" title="Link to this heading">#</a>

A common question is when to use <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.eval.html#mlx.core.eval"
class="reference internal" title="mlx.core.eval"><span class="pre"><code
class="sourceCode python"><span class="bu">eval</span>()</code></span></a>.
The trade-off is between letting graphs get too large and not batching
enough useful work.

For example:

<div class="highlight-python notranslate">

<div class="highlight">

    for _ in range(100):
         a = a + b
         mx.eval(a)
         b = b * 2
         mx.eval(b)

</div>

</div>

This is a bad idea because there is some fixed overhead with each graph
evaluation. On the other hand, there is some slight overhead which grows
with the compute graph size, so extremely large graphs (while
computationally correct) can be costly.

Luckily, a wide range of compute graph sizes work pretty well with MLX:
anything from a few tens of operations to many thousands of operations
per evaluation should be okay.

Most numerical computations have an iterative outer loop (e.g. the
iteration in stochastic gradient descent). A natural and usually
efficient place to use <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.eval.html#mlx.core.eval"
class="reference internal" title="mlx.core.eval"><span class="pre"><code
class="sourceCode python"><span class="bu">eval</span>()</code></span></a>
is at each iteration of this outer loop.

Here is a concrete example:

<div class="highlight-python notranslate">

<div class="highlight">

    for batch in dataset:

        # Nothing has been evaluated yet
        loss, grad = value_and_grad_fn(model, batch)

        # Still nothing has been evaluated
        optimizer.update(model, grad)

        # Evaluate the loss and the new parameters which will
        # run the full gradient computation and optimizer update
        mx.eval(loss, model.parameters())

</div>

</div>

An important behavior to be aware of is when the graph will be
implicitly evaluated. Anytime you <span class="pre">`print`</span> an
array, convert it to an <a
href="https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray"
class="reference external" title="(in NumPy v2.2)"><span
class="pre"><code
class="sourceCode python">numpy.ndarray</code></span></a>, or otherwise
access its memory via
<a href="https://docs.python.org/3/library/stdtypes.html#memoryview"
class="reference external" title="(in Python v3.13)"><span
class="pre"><code
class="sourceCode python"><span class="bu">memoryview</span></code></span></a>,
the graph will be evaluated. Saving arrays via <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.save.html#mlx.core.save"
class="reference internal" title="mlx.core.save"><span class="pre"><code
class="sourceCode python">save()</code></span></a> (or any other MLX
saving functions) will also evaluate the array.

Calling <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.item.html#mlx.core.array.item"
class="reference internal" title="mlx.core.array.item"><span
class="pre"><code
class="sourceCode python">array.item()</code></span></a> on a scalar
array will also evaluate it. In the example above, printing the loss
(<span class="pre">`print(loss)`</span>) or adding the loss scalar to a
list (<span class="pre">`losses.append(loss.item())`</span>) would cause
a graph evaluation. If these lines are before
<span class="pre">`mx.eval(loss,`</span>` `<span class="pre">`model.parameters())`</span>
then this will be a partial evaluation, computing only the forward pass.

Also, calling <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.eval.html#mlx.core.eval"
class="reference internal" title="mlx.core.eval"><span class="pre"><code
class="sourceCode python"><span class="bu">eval</span>()</code></span></a>
on an array or set of arrays multiple times is perfectly fine. This is
effectively a no-op.

<div class="admonition warning">

Warning

Using scalar arrays for control-flow will cause an evaluation.

</div>

Here is an example:

<div class="highlight-python notranslate">

<div class="highlight">

    def fun(x):
        h, y = first_layer(x)
        if y > 0:  # An evaluation is done here!
            z  = second_layer_a(h)
        else:
            z  = second_layer_b(h)
        return z

</div>

</div>

Using arrays for control flow should be done with care. The above
example works and can even be used with gradient transformations.
However, this can be very inefficient if evaluations are done too
frequently.

</div>

</div>

<div class="prev-next-area">

<a
href="https://ml-explore.github.io/mlx/build/html/usage/quick_start.html"
class="left-prev" title="previous page"><em></em></a>

<div class="prev-next-info">

previous

Quick Start Guide

</div>

<a
href="https://ml-explore.github.io/mlx/build/html/usage/unified_memory.html"
class="right-next" title="next page"></a>

<div class="prev-next-info">

next

Unified Memory

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
  href="https://ml-explore.github.io/mlx/build/html/#why-lazy-evaluation"
  class="reference internal nav-link">Why Lazy Evaluation</a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/#transforming-compute-graphs"
    class="reference internal nav-link">Transforming Compute Graphs</a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/#only-compute-what-you-use"
    class="reference internal nav-link">Only Compute What You Use</a>
- <a href="https://ml-explore.github.io/mlx/build/html/#when-to-evaluate"
  class="reference internal nav-link">When to Evaluate</a>

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
