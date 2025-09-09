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
  href="https://ml-explore.github.io/mlx/build/html/_sources/python/_autosummary/mlx.core.compile.rst"
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

# mlx.core.compile

<div id="print-main-content">

<div id="jb-print-toc">

<div>

## Contents

</div>

- <a href="https://ml-explore.github.io/mlx/build/html/#mlx.core.compile"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">compile()</code></span></a>

</div>

</div>

</div>

<div id="searchbox">

</div>

<div id="mlx-core-compile" class="section">

# mlx.core.compile<a href="https://ml-explore.github.io/mlx/build/html/#mlx-core-compile"
class="headerlink" title="Link to this heading">#</a>

<span class="sig-name descname"><span class="pre">compile</span></span><span class="sig-paren">(</span>*<span class="n"><span class="pre">fun</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><span class="pre">Callable</span></span>*, *<span class="n"><span class="pre">inputs</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><a href="https://docs.python.org/3/library/functions.html#object"
class="reference external" title="(in Python v3.13)"><span
class="pre">object</span></a><span class="w"> </span><span class="p"><span class="pre">\|</span></span><span class="w"> </span><a href="https://docs.python.org/3/library/constants.html#None"
class="reference external" title="(in Python v3.13)"><span
class="pre">None</span></a></span><span class="w"> </span><span class="o"><span class="pre">=</span></span><span class="w"> </span><span class="default_value"><span class="pre">None</span></span>*, *<span class="n"><span class="pre">outputs</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><a href="https://docs.python.org/3/library/functions.html#object"
class="reference external" title="(in Python v3.13)"><span
class="pre">object</span></a><span class="w"> </span><span class="p"><span class="pre">\|</span></span><span class="w"> </span><a href="https://docs.python.org/3/library/constants.html#None"
class="reference external" title="(in Python v3.13)"><span
class="pre">None</span></a></span><span class="w"> </span><span class="o"><span class="pre">=</span></span><span class="w"> </span><span class="default_value"><span class="pre">None</span></span>*, *<span class="n"><span class="pre">shapeless</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><a href="https://docs.python.org/3/library/functions.html#bool"
class="reference external" title="(in Python v3.13)"><span
class="pre">bool</span></a></span><span class="w"> </span><span class="o"><span class="pre">=</span></span><span class="w"> </span><span class="default_value"><span class="pre">False</span></span>*<span class="sig-paren">)</span> <span class="sig-return"><span class="sig-return-icon">→</span> <span class="sig-return-typehint"><span class="pre">Callable</span></span></span><a href="https://ml-explore.github.io/mlx/build/html/#mlx.core.compile"
class="headerlink" title="Link to this definition">#</a>  
Returns a compiled function which produces the same output as
<span class="pre">`fun`</span>.

Parameters<span class="colon">:</span>  
- **fun** (*Callable*) – A function which takes a variable number of <a
  href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.html#mlx.core.array"
  class="reference internal" title="mlx.core.array"><span
  class="pre"><code class="sourceCode python">array</code></span></a> or
  trees of <a
  href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.html#mlx.core.array"
  class="reference internal" title="mlx.core.array"><span
  class="pre"><code class="sourceCode python">array</code></span></a>
  and returns a variable number of <a
  href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.html#mlx.core.array"
  class="reference internal" title="mlx.core.array"><span
  class="pre"><code class="sourceCode python">array</code></span></a> or
  trees of <a
  href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.html#mlx.core.array"
  class="reference internal" title="mlx.core.array"><span
  class="pre"><code class="sourceCode python">array</code></span></a>.

- **inputs**
  (<a href="https://docs.python.org/3/library/stdtypes.html#list"
  class="reference external" title="(in Python v3.13)"><em>list</em></a>
  *or* <a href="https://docs.python.org/3/library/stdtypes.html#dict"
  class="reference external" title="(in Python v3.13)"><em>dict</em></a>*,*
  *optional*) – These inputs will be captured during the function
  compilation along with the inputs to <span class="pre">`fun`</span>.
  The <span class="pre">`inputs`</span> can be a
  <a href="https://docs.python.org/3/library/stdtypes.html#list"
  class="reference external" title="(in Python v3.13)"><span
  class="pre"><code
  class="sourceCode python"><span class="bu">list</span></code></span></a>
  or a <a href="https://docs.python.org/3/library/stdtypes.html#dict"
  class="reference external" title="(in Python v3.13)"><span
  class="pre"><code
  class="sourceCode python"><span class="bu">dict</span></code></span></a>
  containing arbitrarily nested lists, dictionaries, or arrays. Leaf
  nodes that are not <a
  href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.html#mlx.core.array"
  class="reference internal" title="mlx.core.array"><span
  class="pre"><code class="sourceCode python">array</code></span></a>
  are ignored. Default: <span class="pre">`None`</span>

- **outputs**
  (<a href="https://docs.python.org/3/library/stdtypes.html#list"
  class="reference external" title="(in Python v3.13)"><em>list</em></a>
  *or* <a href="https://docs.python.org/3/library/stdtypes.html#dict"
  class="reference external" title="(in Python v3.13)"><em>dict</em></a>*,*
  *optional*) – These outputs will be captured and updated in a compiled
  function. The <span class="pre">`outputs`</span> can be a
  <a href="https://docs.python.org/3/library/stdtypes.html#list"
  class="reference external" title="(in Python v3.13)"><span
  class="pre"><code
  class="sourceCode python"><span class="bu">list</span></code></span></a>
  or a <a href="https://docs.python.org/3/library/stdtypes.html#dict"
  class="reference external" title="(in Python v3.13)"><span
  class="pre"><code
  class="sourceCode python"><span class="bu">dict</span></code></span></a>
  containing arbitrarily nested lists, dictionaries, or arrays. Leaf
  nodes that are not <a
  href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.html#mlx.core.array"
  class="reference internal" title="mlx.core.array"><span
  class="pre"><code class="sourceCode python">array</code></span></a>
  are ignored. Default: <span class="pre">`None`</span>

- **shapeless**
  (<a href="https://docs.python.org/3/library/functions.html#bool"
  class="reference external" title="(in Python v3.13)"><em>bool</em></a>*,*
  *optional*) – A function compiled with the
  <span class="pre">`shapeless`</span> option enabled will not be
  recompiled when the input shape changes. Not all functions can be
  compiled with <span class="pre">`shapeless`</span> enabled. Attempting
  to compile such functions with shapeless enabled will throw. Note,
  changing the number of dimensions or type of any input will result in
  a recompilation even with <span class="pre">`shapeless`</span> set to
  <span class="pre">`True`</span>. Default:
  <span class="pre">`False`</span>

Returns<span class="colon">:</span>  
A compiled function which has the same input arguments as
<span class="pre">`fun`</span> and returns the the same output(s).

Return type<span class="colon">:</span>  
*Callable*

## MetalFaiss Notes — Real‑World Use and Benchmarks

- Where compile helped most in practice:
  - SVD MLX step (Z = Aᵀ(A·V) then QR): compiling the one‑step function reduced repeated‑call overhead and fused MLX graphs. On a 512×256 matrix with k=32 and 3 iterations, we measured ~1.6× speedup (non‑compiled ≈0.036s → compiled ≈0.023s median).
  - Simple elementwise chains (e.g., GELU) fuse into one kernel; MLX docs show up to ~5× on large arrays. See also: function_transforms.md (value_and_grad examples) and compile.md.
  - Kernel orchestration (custom Metal kernels): compiling a thin wrapper around gemm_av→gemm_at_b is near parity (kernels dominate runtime), but still trims Python overhead when many bands or small kernels are dispatched.

- Patterns we use:
  - Compile outer steps with stable shapes/dtypes; cache by (m,n,k,dtype,device). Avoid building/destroying compiled lambdas in loops.
  - Keep code MLX‑pure: use `mx.square/mx.divide/mx.multiply/mx.where`, no `.item()/.numpy()/.tolist()`; constants as MLX scalars.
  - Mixed CPU/GPU inside a compiled function works on Apple unified memory; assign ops to `stream=mx.cpu` or `mx.gpu`. MLX schedules cross‑device deps. See unified‑memory guidance in compile.md and transforms.md.

### Cross‑References

- See also:
  - compile.md (Basics, shapeless mode, pitfalls)
  - function_transforms.md (compose compile with grad/value_and_grad)
  - mlx.core.enable_compile.md / mlx.core.disable_compile.md
  - ../guides/MetalFaiss-Compile-Guide.md (project‑specific patterns, do/don’t list, and code snippets)


</div>

<div class="prev-next-area">

<a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.async_eval.html"
class="left-prev" title="previous page"><em></em></a>

<div class="prev-next-info">

previous

mlx.core.async_eval

</div>

<a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.custom_function.html"
class="right-next" title="next page"></a>

<div class="prev-next-info">

next

mlx.core.custom_function

</div>

</div>

</div>

<div class="bd-sidebar-secondary bd-toc">

<div class="sidebar-secondary-items sidebar-secondary__inner">

<div class="sidebar-secondary-item">

<div class="page-toc tocsection onthispage">

Contents

</div>

- <a href="https://ml-explore.github.io/mlx/build/html/#mlx.core.compile"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">compile()</code></span></a>

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
