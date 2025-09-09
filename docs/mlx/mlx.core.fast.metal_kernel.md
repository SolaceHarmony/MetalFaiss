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
  href="https://ml-explore.github.io/mlx/build/html/_sources/python/_autosummary/mlx.core.fast.metal_kernel.rst"
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

# mlx.core.fast.metal_kernel

<div id="print-main-content">

<div id="jb-print-toc">

<div>

## Contents

</div>

- <a
  href="https://ml-explore.github.io/mlx/build/html/#mlx.core.fast.metal_kernel"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">metal_kernel()</code></span></a>

</div>

</div>

</div>

<div id="searchbox">

</div>

<div id="mlx-core-fast-metal-kernel" class="section">

# mlx.core.fast.metal_kernel<a
href="https://ml-explore.github.io/mlx/build/html/#mlx-core-fast-metal-kernel"
class="headerlink" title="Link to this heading">#</a>

<span class="sig-name descname"><span class="pre">metal_kernel</span></span><span class="sig-paren">(</span>*<span class="n"><span class="pre">name</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><a href="https://docs.python.org/3/library/stdtypes.html#str"
class="reference external" title="(in Python v3.13)"><span
class="pre">str</span></a></span>*, *<span class="n"><span class="pre">input_names</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><a
href="https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence"
class="reference external" title="(in Python v3.13)"><span
class="pre">Sequence</span></a><span class="p"><span class="pre">\[</span></span><a href="https://docs.python.org/3/library/stdtypes.html#str"
class="reference external" title="(in Python v3.13)"><span
class="pre">str</span></a><span class="p"><span class="pre">\]</span></span></span>*, *<span class="n"><span class="pre">output_names</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><a
href="https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence"
class="reference external" title="(in Python v3.13)"><span
class="pre">Sequence</span></a><span class="p"><span class="pre">\[</span></span><a href="https://docs.python.org/3/library/stdtypes.html#str"
class="reference external" title="(in Python v3.13)"><span
class="pre">str</span></a><span class="p"><span class="pre">\]</span></span></span>*, *<span class="n"><span class="pre">source</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><a href="https://docs.python.org/3/library/stdtypes.html#str"
class="reference external" title="(in Python v3.13)"><span
class="pre">str</span></a></span>*, *<span class="n"><span class="pre">header</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><a href="https://docs.python.org/3/library/stdtypes.html#str"
class="reference external" title="(in Python v3.13)"><span
class="pre">str</span></a></span><span class="w"> </span><span class="o"><span class="pre">=</span></span><span class="w"> </span><span class="default_value"><span class="pre">''</span></span>*, *<span class="n"><span class="pre">ensure_row_contiguous</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><a href="https://docs.python.org/3/library/functions.html#bool"
class="reference external" title="(in Python v3.13)"><span
class="pre">bool</span></a></span><span class="w"> </span><span class="o"><span class="pre">=</span></span><span class="w"> </span><span class="default_value"><span class="pre">True</span></span>*, *<span class="n"><span class="pre">atomic_outputs</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><a href="https://docs.python.org/3/library/functions.html#bool"
class="reference external" title="(in Python v3.13)"><span
class="pre">bool</span></a></span><span class="w"> </span><span class="o"><span class="pre">=</span></span><span class="w"> </span><span class="default_value"><span class="pre">False</span></span>*<span class="sig-paren">)</span> <span class="sig-return"><span class="sig-return-icon">→</span> <span class="sig-return-typehint"><a href="https://docs.python.org/3/library/functions.html#object"
class="reference external" title="(in Python v3.13)"><span
class="pre">object</span></a></span></span><a
href="https://ml-explore.github.io/mlx/build/html/#mlx.core.fast.metal_kernel"
class="headerlink" title="Link to this definition">#</a>  
A jit-compiled custom Metal kernel defined from a source string.

Full documentation: <a
href="https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html#custom-metal-kernels"
class="reference internal"><span class="std std-ref">Custom Metal
Kernels</span></a>.

Parameters<span class="colon">:</span>  
- **name**
  (<a href="https://docs.python.org/3/library/stdtypes.html#str"
  class="reference external" title="(in Python v3.13)"><em>str</em></a>)
  – Name for the kernel.

- **input_names**
  (*List\[*<a href="https://docs.python.org/3/library/stdtypes.html#str"
  class="reference external" title="(in Python v3.13)"><em>str</em></a>*\]*)
  – The parameter names of the inputs in the function signature.

- **output_names**
  (*List\[*<a href="https://docs.python.org/3/library/stdtypes.html#str"
  class="reference external" title="(in Python v3.13)"><em>str</em></a>*\]*)
  – The parameter names of the outputs in the function signature.

- **source**
  (<a href="https://docs.python.org/3/library/stdtypes.html#str"
  class="reference external" title="(in Python v3.13)"><em>str</em></a>)
  – Source code. This is the body of a function in Metal, the function
  signature will be automatically generated.

- **header**
  (<a href="https://docs.python.org/3/library/stdtypes.html#str"
  class="reference external" title="(in Python v3.13)"><em>str</em></a>)
  – Header source code to include before the main function. Useful for
  helper functions or includes that should live outside of the main
  function body.

- **ensure_row_contiguous**
  (<a href="https://docs.python.org/3/library/functions.html#bool"
  class="reference external" title="(in Python v3.13)"><em>bool</em></a>)
  – Whether to ensure the inputs are row contiguous before the kernel
  runs. Default: <span class="pre">`True`</span>.

- **atomic_outputs**
  (<a href="https://docs.python.org/3/library/functions.html#bool"
  class="reference external" title="(in Python v3.13)"><em>bool</em></a>)
  – Whether to use atomic outputs in the function signature e.g.
  <span class="pre">`device`</span>` `<span class="pre">`atomic<float>`</span>.
  Default: <span class="pre">`False`</span>.

Returns<span class="colon">:</span>  
Callable <span class="pre">`metal_kernel`</span>.

Example

<div class="highlight-python notranslate">

<div class="highlight">

    def exp_elementwise(a: mx.array):
        source = '''
            uint elem = thread_position_in_grid.x;
            T tmp = inp[elem];
            out[elem] = metal::exp(tmp);
        '''

        kernel = mx.fast.metal_kernel(
            name="myexp",
            input_names=["inp"],
            output_names=["out"],
            source=source
        )
        outputs = kernel(
            inputs=[a],
            template=[("T", mx.float32)],
            grid=(a.size, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[a.shape],
            output_dtypes=[a.dtype],
            verbose=True,
        )
        return outputs[0]

    a = mx.random.normal(shape=(4, 16)).astype(mx.float16)
    b = exp_elementwise(a)
    assert mx.allclose(b, mx.exp(a))

</div>

</div>

</div>

<div class="prev-next-area">

<a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fast.scaled_dot_product_attention.html"
class="left-prev" title="previous page"><em></em></a>

<div class="prev-next-info">

previous

mlx.core.fast.scaled_dot_product_attention

</div>

<a href="https://ml-explore.github.io/mlx/build/html/python/fft.html"
class="right-next" title="next page"></a>

<div class="prev-next-info">

next

FFT

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
  href="https://ml-explore.github.io/mlx/build/html/#mlx.core.fast.metal_kernel"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">metal_kernel()</code></span></a>

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
