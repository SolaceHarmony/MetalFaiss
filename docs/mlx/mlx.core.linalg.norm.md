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
  href="https://ml-explore.github.io/mlx/build/html/_sources/python/_autosummary/mlx.core.linalg.norm.rst"
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

# mlx.core.linalg.norm

<div id="print-main-content">

<div id="jb-print-toc">

<div>

## Contents

</div>

- <a
  href="https://ml-explore.github.io/mlx/build/html/#mlx.core.linalg.norm"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">norm()</code></span></a>

</div>

</div>

</div>

<div id="searchbox">

</div>

<div id="mlx-core-linalg-norm" class="section">

# mlx.core.linalg.norm<a
href="https://ml-explore.github.io/mlx/build/html/#mlx-core-linalg-norm"
class="headerlink" title="Link to this heading">#</a>

<span class="sig-name descname"><span class="pre">norm</span></span><span class="sig-paren">(</span>*<span class="n"><span class="pre">a</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.html#mlx.core.array"
class="reference internal" title="mlx.core.array"><span
class="pre">array</span></a></span>*, *<span class="o"><span class="pre">/</span></span>*, *<span class="n"><span class="pre">ord</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><a href="https://docs.python.org/3/library/constants.html#None"
class="reference external" title="(in Python v3.13)"><span
class="pre">None</span></a><span class="w"> </span><span class="p"><span class="pre">\|</span></span><span class="w"> </span><a href="https://docs.python.org/3/library/functions.html#int"
class="reference external" title="(in Python v3.13)"><span
class="pre">int</span></a><span class="w"> </span><span class="p"><span class="pre">\|</span></span><span class="w"> </span><a href="https://docs.python.org/3/library/functions.html#float"
class="reference external" title="(in Python v3.13)"><span
class="pre">float</span></a><span class="w"> </span><span class="p"><span class="pre">\|</span></span><span class="w"> </span><a href="https://docs.python.org/3/library/stdtypes.html#str"
class="reference external" title="(in Python v3.13)"><span
class="pre">str</span></a></span><span class="w"> </span><span class="o"><span class="pre">=</span></span><span class="w"> </span><span class="default_value"><span class="pre">None</span></span>*, *<span class="n"><span class="pre">axis</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><a href="https://docs.python.org/3/library/constants.html#None"
class="reference external" title="(in Python v3.13)"><span
class="pre">None</span></a><span class="w"> </span><span class="p"><span class="pre">\|</span></span><span class="w"> </span><a href="https://docs.python.org/3/library/functions.html#int"
class="reference external" title="(in Python v3.13)"><span
class="pre">int</span></a><span class="w"> </span><span class="p"><span class="pre">\|</span></span><span class="w"> </span><a href="https://docs.python.org/3/library/stdtypes.html#list"
class="reference external" title="(in Python v3.13)"><span
class="pre">list</span></a><span class="p"><span class="pre">\[</span></span><a href="https://docs.python.org/3/library/functions.html#int"
class="reference external" title="(in Python v3.13)"><span
class="pre">int</span></a><span class="p"><span class="pre">\]</span></span></span><span class="w"> </span><span class="o"><span class="pre">=</span></span><span class="w"> </span><span class="default_value"><span class="pre">None</span></span>*, *<span class="n"><span class="pre">keepdims</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><a href="https://docs.python.org/3/library/functions.html#bool"
class="reference external" title="(in Python v3.13)"><span
class="pre">bool</span></a></span><span class="w"> </span><span class="o"><span class="pre">=</span></span><span class="w"> </span><span class="default_value"><span class="pre">False</span></span>*, *<span class="o"><span class="pre">\*</span></span>*, *<span class="n"><span class="pre">stream</span></span><span class="p"><span class="pre">:</span></span><span class="w"> </span><span class="n"><a href="https://docs.python.org/3/library/constants.html#None"
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
class="pre">array</span></a></span></span><a
href="https://ml-explore.github.io/mlx/build/html/#mlx.core.linalg.norm"
class="headerlink" title="Link to this definition">#</a>  
Matrix or vector norm.

This function computes vector or matrix norms depending on the value of
the <span class="pre">`ord`</span> and <span class="pre">`axis`</span>
parameters.

Parameters<span class="colon">:</span>  
- **a** (<a
  href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.html#mlx.core.array"
  class="reference internal" title="mlx.core.array"><em>array</em></a>)
  – Input array. If <span class="pre">`axis`</span> is
  <span class="pre">`None`</span>, <span class="pre">`a`</span> must be
  1-D or 2-D, unless <span class="pre">`ord`</span> is
  <span class="pre">`None`</span>. If both
  <span class="pre">`axis`</span> and <span class="pre">`ord`</span> are
  <span class="pre">`None`</span>, the 2-norm of
  <span class="pre">`a.flatten`</span> will be returned.

- **ord**
  (<a href="https://docs.python.org/3/library/functions.html#int"
  class="reference external" title="(in Python v3.13)"><em>int</em></a>*,*
  <a href="https://docs.python.org/3/library/functions.html#float"
  class="reference external" title="(in Python v3.13)"><em>float</em></a>
  *or* <a href="https://docs.python.org/3/library/stdtypes.html#str"
  class="reference external" title="(in Python v3.13)"><em>str</em></a>*,*
  *optional*) – Order of the norm (see table under
  <span class="pre">`Notes`</span>). If <span class="pre">`None`</span>,
  the 2-norm (or Frobenius norm for matrices) will be computed along the
  given <span class="pre">`axis`</span>. Default:
  <span class="pre">`None`</span>.

- **axis**
  (<a href="https://docs.python.org/3/library/functions.html#int"
  class="reference external" title="(in Python v3.13)"><em>int</em></a>
  *or* <a href="https://docs.python.org/3/library/stdtypes.html#list"
  class="reference external" title="(in Python v3.13)"><em>list</em></a>*(*<a href="https://docs.python.org/3/library/functions.html#int"
  class="reference external" title="(in Python v3.13)"><em>int</em></a>*),*
  *optional*) – If <span class="pre">`axis`</span> is an integer, it
  specifies the axis of <span class="pre">`a`</span> along which to
  compute the vector norms. If <span class="pre">`axis`</span> is a
  2-tuple, it specifies the axes that hold 2-D matrices, and the matrix
  norms of these matrices are computed. If axis is
  <span class="pre">`None`</span> then either a vector norm (when
  <span class="pre">`a`</span> is 1-D) or a matrix norm (when
  <span class="pre">`a`</span> is 2-D) is returned. Default:
  <span class="pre">`None`</span>.

- **keepdims**
  (<a href="https://docs.python.org/3/library/functions.html#bool"
  class="reference external" title="(in Python v3.13)"><em>bool</em></a>*,*
  *optional*) – If <span class="pre">`True`</span>, the axes which are
  normed over are left in the result as dimensions with size one.
  Default <span class="pre">`False`</span>.

Returns<span class="colon">:</span>  
The output containing the norm(s).

Return type<span class="colon">:</span>  
<a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.html#mlx.core.array"
class="reference internal" title="mlx.core.array"><em>array</em></a>

Notes

For values of
<span class="pre">`ord`</span>` `<span class="pre">`<`</span>` `<span class="pre">`1`</span>,
the result is, strictly speaking, not a mathematical norm, but it may
still be useful for various numerical purposes.

The following norms can be calculated:

<div class="pst-scrollable-table-container">

| ord   | norm for matrices            | norm for vectors               |
|-------|------------------------------|--------------------------------|
| None  | Frobenius norm               | 2-norm                         |
| ‘fro’ | Frobenius norm               | –                              |
| ‘nuc’ | nuclear norm                 | –                              |
| inf   | max(sum(abs(x), axis=1))     | max(abs(x))                    |
| -inf  | min(sum(abs(x), axis=1))     | min(abs(x))                    |
| 0     | –                            | sum(x != 0)                    |
| 1     | max(sum(abs(x), axis=0))     | as below                       |
| -1    | min(sum(abs(x), axis=0))     | as below                       |
| 2     | 2-norm (largest sing. value) | as below                       |
| -2    | smallest singular value      | as below                       |
| other | –                            | sum(abs(x)\*\*ord)\*\*(1./ord) |

</div>

The Frobenius norm is given by [^1]:

> <div>
>
> <span class="math notranslate nohighlight">\\\|\|A\|\|\_F =
> \[\sum\_{i,j} abs(a\_{i,j})^2\]^{1/2}\\</span>
>
> </div>

The nuclear norm is the sum of the singular values.

Both the Frobenius and nuclear norm orders are only defined for matrices
and raise a <span class="pre">`ValueError`</span> when
<span class="pre">`a.ndim`</span>` `<span class="pre">`!=`</span>` `<span class="pre">`2`</span>.

References

<span class="label"><span class="fn-bracket">\[</span><a href="https://ml-explore.github.io/mlx/build/html/#id1"
role="doc-backlink">1</a><span class="fn-bracket">\]</span></span>

G. H. Golub and C. F. Van Loan, *Matrix Computations*, Baltimore, MD,
Johns Hopkins University Press, 1985, pg. 15

Examples

<div class="doctest highlight-default notranslate">

<div class="highlight">

    >>> import mlx.core as mx
    >>> from mlx.core import linalg as la
    >>> a = mx.arange(9) - 4
    >>> a
    array([-4, -3, -2, ..., 2, 3, 4], dtype=int32)
    >>> b = a.reshape((3,3))
    >>> b
    array([[-4, -3, -2],
           [-1,  0,  1],
           [ 2,  3,  4]], dtype=int32)
    >>> la.norm(a)
    array(7.74597, dtype=float32)
    >>> la.norm(b)
    array(7.74597, dtype=float32)
    >>> la.norm(b, 'fro')
    array(7.74597, dtype=float32)
    >>> la.norm(a, float("inf"))
    array(4, dtype=float32)
    >>> la.norm(b, float("inf"))
    array(9, dtype=float32)
    >>> la.norm(a, -float("inf"))
    array(0, dtype=float32)
    >>> la.norm(b, -float("inf"))
    array(2, dtype=float32)
    >>> la.norm(a, 1)
    array(20, dtype=float32)
    >>> la.norm(b, 1)
    array(7, dtype=float32)
    >>> la.norm(a, -1)
    array(0, dtype=float32)
    >>> la.norm(b, -1)
    array(6, dtype=float32)
    >>> la.norm(a, 2)
    array(7.74597, dtype=float32)
    >>> la.norm(a, 3)
    array(5.84804, dtype=float32)
    >>> la.norm(a, -3)
    array(0, dtype=float32)
    >>> c = mx.array([[ 1, 2, 3],
    ...               [-1, 1, 4]])
    >>> la.norm(c, axis=0)
    array([1.41421, 2.23607, 5], dtype=float32)
    >>> la.norm(c, axis=1)
    array([3.74166, 4.24264], dtype=float32)
    >>> la.norm(c, ord=1, axis=1)
    array([6, 6], dtype=float32)
    >>> m = mx.arange(8).reshape(2,2,2)
    >>> la.norm(m, axis=(1,2))
    array([3.74166, 11.225], dtype=float32)
    >>> la.norm(m[0, :, :]), LA.norm(m[1, :, :])
    (array(3.74166, dtype=float32), array(11.225, dtype=float32))

</div>

</div>

</div>

<div class="prev-next-area">

<a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.linalg.tri_inv.html"
class="left-prev" title="previous page"><em></em></a>

<div class="prev-next-info">

previous

mlx.core.linalg.tri_inv

</div>

<a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.linalg.cholesky.html"
class="right-next" title="next page"></a>

<div class="prev-next-info">

next

mlx.core.linalg.cholesky

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
  href="https://ml-explore.github.io/mlx/build/html/#mlx.core.linalg.norm"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">norm()</code></span></a>

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

[^1]:
