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
  href="https://ml-explore.github.io/mlx/build/html/_sources/cpp/ops.rst"
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

# Operations

<div id="print-main-content">

<div id="jb-print-toc">

<div>

## Contents

</div>

- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv46arangeddd5Dtype14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">arange()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv46arangeddd14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">arange()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv46arangedd5Dtype14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">arange()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv46arangedd14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">arange()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv46aranged5Dtype14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">arange()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv46aranged14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">arange()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv46arangeiii14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">arange()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv46arangeii14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">arange()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv46arangei14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">arange()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv48linspaceddi5Dtype14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">linspace()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv46astype5array5Dtype14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">astype()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv410as_strided5array5Shape7Strides6size_t14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">as_strided()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv44copy5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">copy()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv44full5Shape5array5Dtype14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">full()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv44full5Shape5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">full()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv4I0E4full5array5Shape1T5Dtype14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">full()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv4I0E4full5array5Shape1T14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">full()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv45zerosRK5Shape5Dtype14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">zeros()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv45zerosRK5Shape14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">zeros()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv410zeros_likeRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">zeros_like()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv44onesRK5Shape5Dtype14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">ones()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv44onesRK5Shape14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">ones()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv49ones_likeRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">ones_like()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv43eyeiii5Dtype14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">eye()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv43eyei5Dtype14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">eye()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv43eyeii14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">eye()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv43eyeiii14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">eye()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv43eyei14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">eye()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv48identityi5Dtype14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">identity()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv48identityi14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">identity()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv43triiii5Dtype14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">tri()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv43trii5Dtype14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">tri()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv44tril5arrayi14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">tril()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv44triu5arrayi14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">triu()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv47reshapeRK5array5Shape14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">reshape()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv49unflattenRK5arrayi5Shape14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">unflatten()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv47flattenRK5arrayii14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">flatten()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv47flattenRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">flatten()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv418hadamard_transformRK5arrayNSt8optionalIfEE14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">hadamard_transform()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv47squeezeRK5arrayRKNSt6vectorIiEE14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">squeeze()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv47squeezeRK5arrayi14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">squeeze()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv47squeezeRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">squeeze()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv411expand_dimsRK5arrayRKNSt6vectorIiEE14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">expand_dims()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv411expand_dimsRK5arrayi14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">expand_dims()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv45sliceRK5array5Shape5Shape5Shape14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">slice()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv45sliceRK5arrayNSt16initializer_listIiEE5Shape5Shape14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">slice()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv45sliceRK5array5Shape5Shape14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">slice()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv45sliceRK5arrayRK5arrayNSt6vectorIiEE5Shape14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">slice()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv412slice_updateRK5arrayRK5array5Shape5Shape5Shape14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">slice_update()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv412slice_updateRK5arrayRK5array5Shape5Shape14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">slice_update()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv412slice_updateRK5arrayRK5arrayRK5arrayNSt6vectorIiEE14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">slice_update()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv45splitRK5arrayii14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">split()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv45splitRK5arrayi14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">split()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv45splitRK5arrayRK5Shapei14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">split()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv45splitRK5arrayRK5Shape14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">split()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv48meshgridRKNSt6vectorI5arrayEEbRKNSt6stringE14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">meshgrid()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv44clipRK5arrayRKNSt8optionalI5arrayEERKNSt8optionalI5arrayEE14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">clip()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv411concatenateNSt6vectorI5arrayEEi14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">concatenate()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv411concatenateNSt6vectorI5arrayEE14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">concatenate()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv45stackRKNSt6vectorI5arrayEEi14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">stack()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv45stackRKNSt6vectorI5arrayEE14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">stack()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv46repeatRK5arrayii14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">repeat()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv46repeatRK5arrayi14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">repeat()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv44tileRK5arrayNSt6vectorIiEE14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">tile()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv49transposeRK5arrayNSt6vectorIiEE14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">transpose()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv49transposeRK5arrayNSt16initializer_listIiEE14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">transpose()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv48swapaxesRK5arrayii14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">swapaxes()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv48moveaxisRK5arrayii14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">moveaxis()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv43padRK5arrayRKNSt6vectorIiEERK5ShapeRK5ShapeRK5arrayRKNSt6stringE14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">pad()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv43padRK5arrayRKNSt6vectorINSt4pairIiiEEEERK5arrayRKNSt6stringE14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">pad()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv43padRK5arrayRKNSt4pairIiiEERK5arrayRKNSt6stringE14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">pad()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv43padRK5arrayiRK5arrayRKNSt6stringE14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">pad()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv49transposeRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">transpose()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv412broadcast_toRK5arrayRK5Shape14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">broadcast_to()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv416broadcast_arraysRKNSt6vectorI5arrayEE14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">broadcast_arrays()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv45equalRK5arrayRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">equal()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv4eqRK5arrayRK5array"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">operator==()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv4I0Eeq5array1TRK5array"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">operator==()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv4I0Eeq5arrayRK5array1T"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">operator==()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv49not_equalRK5arrayRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">not_equal()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv4neRK5arrayRK5array"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">operator!=()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv4I0Ene5array1TRK5array"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">operator!=()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv4I0Ene5arrayRK5array1T"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">operator!=()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv47greaterRK5arrayRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">greater()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv4gtRK5arrayRK5array"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">operator&gt;()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv4I0Egt5array1TRK5array"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">operator&gt;()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv4I0Egt5arrayRK5array1T"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">operator&gt;()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv413greater_equalRK5arrayRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">greater_equal()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv4geRK5arrayRK5array"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">operator&gt;=()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv4I0Ege5array1TRK5array"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">operator&gt;=()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv4I0Ege5arrayRK5array1T"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">operator&gt;=()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv44lessRK5arrayRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">less()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv4ltRK5arrayRK5array"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">operator&lt;()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv4I0Elt5array1TRK5array"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">operator&lt;()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv4I0Elt5arrayRK5array1T"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">operator&lt;()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv410less_equalRK5arrayRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">less_equal()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv4leRK5arrayRK5array"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">operator&lt;=()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv4I0Ele5array1TRK5array"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">operator&lt;=()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv4I0Ele5arrayRK5array1T"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">operator&lt;=()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv411array_equalRK5arrayRK5arrayb14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">array_equal()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv411array_equalRK5arrayRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">array_equal()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv45isnanRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">isnan()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv45isinfRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">isinf()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv48isfiniteRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">isfinite()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv48isposinfRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">isposinf()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv48isneginfRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">isneginf()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv45whereRK5arrayRK5arrayRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">where()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv410nan_to_numRK5arrayfKNSt8optionalIfEEKNSt8optionalIfEE14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">nan_to_num()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv43allRK5arrayb14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">all()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv43allRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">all()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv48allcloseRK5arrayRK5arrayddb14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">allclose()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv47iscloseRK5arrayRK5arrayddb14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">isclose()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv43allRK5arrayRKNSt6vectorIiEEb14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">all()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv43allRK5arrayib14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">all()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv43anyRK5arrayb14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">any()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv43anyRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">any()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv43anyRK5arrayRKNSt6vectorIiEEb14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">any()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv43anyRK5arrayib14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">any()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv43sumRK5arrayb14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">sum()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv43sumRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">sum()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv43sumRK5arrayRKNSt6vectorIiEEb14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">sum()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv43sumRK5arrayib14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">sum()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv44meanRK5arrayb14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">mean()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv44meanRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">mean()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv44meanRK5arrayRKNSt6vectorIiEEb14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">mean()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv44meanRK5arrayib14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">mean()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv43varRK5arraybi14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">var()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv43varRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">var()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv43varRK5arrayRKNSt6vectorIiEEbi14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">var()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv43varRK5arrayibi14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">var()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv4StRK5arraybi14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">std()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv4StRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">std()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv4StRK5arrayRKNSt6vectorIiEEbi14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">std()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv4StRK5arrayibi14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">std()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv44prodRK5arrayb14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">prod()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv44prodRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">prod()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv44prodRK5arrayRKNSt6vectorIiEEb14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">prod()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv44prodRK5arrayib14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">prod()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv43maxRK5arrayb14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">max()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv43maxRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">max()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv43maxRK5arrayRKNSt6vectorIiEEb14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">max()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv43maxRK5arrayib14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">max()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv43minRK5arrayb14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">min()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv43minRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">min()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv43minRK5arrayRKNSt6vectorIiEEb14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">min()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv43minRK5arrayib14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">min()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv46argminRK5arrayb14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">argmin()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv46argminRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">argmin()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv46argminRK5arrayib14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">argmin()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv46argmaxRK5arrayb14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">argmax()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv46argmaxRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">argmax()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv46argmaxRK5arrayib14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">argmax()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv44sortRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">sort()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv44sortRK5arrayi14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">sort()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv47argsortRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">argsort()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv47argsortRK5arrayi14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">argsort()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv49partitionRK5arrayi14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">partition()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv49partitionRK5arrayii14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">partition()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv412argpartitionRK5arrayi14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">argpartition()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv412argpartitionRK5arrayii14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">argpartition()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv44topkRK5arrayi14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">topk()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv44topkRK5arrayii14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">topk()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv412logcumsumexpRK5arrayibb14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">logcumsumexp()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv49logsumexpRK5arrayb14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">logsumexp()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv49logsumexpRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">logsumexp()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv49logsumexpRK5arrayRKNSt6vectorIiEEb14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">logsumexp()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv49logsumexpRK5arrayib14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">logsumexp()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv43absRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">abs()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv48negativeRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">negative()</code></span></a>
- <a href="https://ml-explore.github.io/mlx/build/html/#_CPPv4miRK5array"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">operator-()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv44signRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">sign()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv411logical_notRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">logical_not()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv411logical_andRK5arrayRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">logical_and()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv4aaRK5arrayRK5array"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">operator&amp;&amp;()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv410logical_orRK5arrayRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">logical_or()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv4ooRK5arrayRK5array"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">operator||()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv410reciprocalRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">reciprocal()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv43addRK5arrayRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">add()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv4plRK5arrayRK5array"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">operator+()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv4I0Epl5array1TRK5array"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">operator+()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv4I0Epl5arrayRK5array1T"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">operator+()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv48subtractRK5arrayRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">subtract()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv4miRK5arrayRK5array"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">operator-()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv4I0Emi5array1TRK5array"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">operator-()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv4I0Emi5arrayRK5array1T"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">operator-()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv48multiplyRK5arrayRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">multiply()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv4mlRK5arrayRK5array"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">operator*()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv4I0Eml5array1TRK5array"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">operator*()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv4I0Eml5arrayRK5array1T"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">operator*()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv46divideRK5arrayRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">divide()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv4dvRK5arrayRK5array"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">operator/()</code></span></a>
- <a href="https://ml-explore.github.io/mlx/build/html/#_CPPv4dvdRK5array"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">operator/()</code></span></a>
- <a href="https://ml-explore.github.io/mlx/build/html/#_CPPv4dvRK5arrayd"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">operator/()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv46divmodRK5arrayRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">divmod()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv412floor_divideRK5arrayRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">floor_divide()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv49remainderRK5arrayRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">remainder()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv4rmRK5arrayRK5array"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">operator%()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv4I0Erm5array1TRK5array"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">operator%()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv4I0Erm5arrayRK5array1T"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">operator%()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv47maximumRK5arrayRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">maximum()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv47minimumRK5arrayRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">minimum()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv45floorRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">floor()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv44ceilRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">ceil()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv46squareRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">square()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv43expRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">exp()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv43sinRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">sin()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv43cosRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">cos()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv43tanRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">tan()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv46arcsinRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">arcsin()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv46arccosRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">arccos()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv46arctanRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">arctan()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv47arctan2RK5arrayRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">arctan2()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv44sinhRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">sinh()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv44coshRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">cosh()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv44tanhRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">tanh()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv47arcsinhRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">arcsinh()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv47arccoshRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">arccosh()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv47arctanhRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">arctanh()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv47degreesRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">degrees()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv47radiansRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">radians()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv43logRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">log()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv44log2RK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">log2()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv45log10RK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">log10()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv45log1pRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">log1p()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv49logaddexpRK5arrayRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">logaddexp()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv47sigmoidRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">sigmoid()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv43erfRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">erf()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv46erfinvRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">erfinv()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv45expm1RK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">expm1()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv413stop_gradientRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">stop_gradient()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv45roundRK5arrayi14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">round()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv45roundRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">round()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv46matmulRK5arrayRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">matmul()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv46gatherRK5arrayRKNSt6vectorI5arrayEERKNSt6vectorIiEERK5Shape14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">gather()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv46gatherRK5arrayRK5arrayiRK5Shape14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">gather()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv44kronRK5arrayRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">kron()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv44takeRK5arrayRK5arrayi14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">take()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv44takeRK5arrayii14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">take()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv44takeRK5arrayRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">take()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv44takeRK5arrayi14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">take()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv415take_along_axisRK5arrayRK5arrayi14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">take_along_axis()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv414put_along_axisRK5arrayRK5arrayRK5arrayi14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">put_along_axis()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv416scatter_add_axisRK5arrayRK5arrayRK5arrayi14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">scatter_add_axis()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv47scatterRK5arrayRKNSt6vectorI5arrayEERK5arrayRKNSt6vectorIiEE14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">scatter()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv47scatterRK5arrayRK5arrayRK5arrayi14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">scatter()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv411scatter_addRK5arrayRKNSt6vectorI5arrayEERK5arrayRKNSt6vectorIiEE14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">scatter_add()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv411scatter_addRK5arrayRK5arrayRK5arrayi14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">scatter_add()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv412scatter_prodRK5arrayRKNSt6vectorI5arrayEERK5arrayRKNSt6vectorIiEE14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">scatter_prod()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv412scatter_prodRK5arrayRK5arrayRK5arrayi14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">scatter_prod()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv411scatter_maxRK5arrayRKNSt6vectorI5arrayEERK5arrayRKNSt6vectorIiEE14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">scatter_max()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv411scatter_maxRK5arrayRK5arrayRK5arrayi14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">scatter_max()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv411scatter_minRK5arrayRKNSt6vectorI5arrayEERK5arrayRKNSt6vectorIiEE14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">scatter_min()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv411scatter_minRK5arrayRK5arrayRK5arrayi14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">scatter_min()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv44sqrtRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">sqrt()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv45rsqrtRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">rsqrt()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv47softmaxRK5arrayRKNSt6vectorIiEEb14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">softmax()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv47softmaxRK5arrayb14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">softmax()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv47softmaxRK5arrayib14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">softmax()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv45powerRK5arrayRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">power()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv46cumsumRK5arrayibb14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">cumsum()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv47cumprodRK5arrayibb14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">cumprod()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv46cummaxRK5arrayibb14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">cummax()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv46cumminRK5arrayibb14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">cummin()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv412conv_general5array5arrayNSt6vectorIiEENSt6vectorIiEENSt6vectorIiEENSt6vectorIiEENSt6vectorIiEEib14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">conv_general()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv412conv_generalRK5arrayRK5arrayNSt6vectorIiEENSt6vectorIiEENSt6vectorIiEENSt6vectorIiEEib14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">conv_general()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv46conv1dRK5arrayRK5arrayiiii14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">conv1d()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv46conv2dRK5arrayRK5arrayRKNSt4pairIiiEERKNSt4pairIiiEERKNSt4pairIiiEEi14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">conv2d()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv46conv3dRK5arrayRK5arrayRKNSt5tupleIiiiEERKNSt5tupleIiiiEERKNSt5tupleIiiiEEi14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">conv3d()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv416conv_transpose1dRK5arrayRK5arrayiiiii14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">conv_transpose1d()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv416conv_transpose2dRK5arrayRK5arrayRKNSt4pairIiiEERKNSt4pairIiiEERKNSt4pairIiiEERKNSt4pairIiiEEi14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">conv_transpose2d()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv416conv_transpose3dRK5arrayRK5arrayRKNSt5tupleIiiiEERKNSt5tupleIiiiEERKNSt5tupleIiiiEERKNSt5tupleIiiiEEi14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">conv_transpose3d()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv416quantized_matmul5array5array5array5arraybii14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">quantized_matmul()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv48quantizeRK5arrayii14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">quantize()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv410dequantizeRK5arrayRK5arrayRK5arrayii14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">dequantize()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv410gather_qmmRK5arrayRK5arrayRK5arrayRK5arrayNSt8optionalI5arrayEENSt8optionalI5arrayEEbiib14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">gather_qmm()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv49tensordotRK5arrayRK5arrayKi14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">tensordot()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv49tensordotRK5arrayRK5arrayRKNSt6vectorIiEERKNSt6vectorIiEE14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">tensordot()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv45outerRK5arrayRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">outer()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv45innerRK5arrayRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">inner()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv45addmm5array5array5arrayRKfRKf14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">addmm()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv415block_masked_mm5array5arrayiNSt8optionalI5arrayEENSt8optionalI5arrayEENSt8optionalI5arrayEE14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">block_masked_mm()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv49gather_mm5array5arrayNSt8optionalI5arrayEENSt8optionalI5arrayEEb14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">gather_mm()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv48diagonalRK5arrayiii14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">diagonal()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv44diagRK5arrayi14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">diag()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv45traceRK5arrayiii5Dtype14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">trace()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv45traceRK5arrayiii14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">trace()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv45traceRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">trace()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv47dependsRKNSt6vectorI5arrayEERKNSt6vectorI5arrayEE"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">depends()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv410atleast_1dRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">atleast_1d()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv410atleast_1dRKNSt6vectorI5arrayEE14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">atleast_1d()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv410atleast_2dRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">atleast_2d()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv410atleast_2dRKNSt6vectorI5arrayEE14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">atleast_2d()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv410atleast_3dRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">atleast_3d()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv410atleast_3dRKNSt6vectorI5arrayEE14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">atleast_3d()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv418number_of_elementsRK5arrayNSt6vectorIiEEb5Dtype14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">number_of_elements()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv49conjugateRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">conjugate()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv411bitwise_andRK5arrayRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">bitwise_and()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv4anRK5arrayRK5array"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">operator&amp;()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv410bitwise_orRK5arrayRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">bitwise_or()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv4orRK5arrayRK5array"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">operator|()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv411bitwise_xorRK5arrayRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">bitwise_xor()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv4eoRK5arrayRK5array"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">operator^()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv410left_shiftRK5arrayRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">left_shift()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv4lsRK5arrayRK5array"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">operator&lt;&lt;()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv411right_shiftRK5arrayRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">right_shift()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv4rsRK5arrayRK5array"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">operator&gt;&gt;()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv414bitwise_invertRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">bitwise_invert()</code></span></a>
- <a href="https://ml-explore.github.io/mlx/build/html/#_CPPv4coRK5array"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">operator~()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv44viewRK5arrayRK5Dtype14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">view()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv44rollRK5arrayi14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">roll()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv44rollRK5arrayRK5Shape14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">roll()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv44rollRK5arrayii14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">roll()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv44rollRK5arrayiRKNSt6vectorIiEE14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">roll()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv44rollRK5arrayRK5Shapei14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">roll()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv44rollRK5arrayRK5ShapeRKNSt6vectorIiEE14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">roll()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv44realRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">real()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv44imagRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">imag()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv410contiguousRK5arrayb14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">contiguous()</code></span></a>

</div>

</div>

</div>

<div id="searchbox">

</div>

<div id="operations" class="section">

<span id="cpp-ops"></span>

# Operations<a href="https://ml-explore.github.io/mlx/build/html/#operations"
class="headerlink" title="Link to this heading">#</a>

<span id="_CPPv36arangeddd5Dtype14StreamOrDevice"></span><span id="_CPPv26arangeddd5Dtype14StreamOrDevice"></span><span id="arange__double.double.double.Dtype.StreamOrDevice"></span><span id="group__ops_1ga7ca088b8090b9f84f2e08345cf3f835a" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">arange</span></span></span><span class="sig-paren">(</span><span class="kt"><span class="pre">double</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">start</span></span>, <span class="kt"><span class="pre">double</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">stop</span></span>, <span class="kt"><span class="pre">double</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">step</span></span>, <span class="n"><span class="pre">Dtype</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">dtype</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv46arangeddd5Dtype14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
A 1D array of numbers starting at <span class="pre">`start`</span>
(optional), stopping at stop, stepping by
<span class="pre">`step`</span> (optional).

<!-- -->

<span id="_CPPv36arangeddd14StreamOrDevice"></span><span id="_CPPv26arangeddd14StreamOrDevice"></span><span id="arange__double.double.double.StreamOrDevice"></span><span id="group__ops_1ga4c36b841dc5cba391dad029be5a0ad98" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">arange</span></span></span><span class="sig-paren">(</span><span class="kt"><span class="pre">double</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">start</span></span>, <span class="kt"><span class="pre">double</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">stop</span></span>, <span class="kt"><span class="pre">double</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">step</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv46arangeddd14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    

<!-- -->

<span id="_CPPv36arangedd5Dtype14StreamOrDevice"></span><span id="_CPPv26arangedd5Dtype14StreamOrDevice"></span><span id="arange__double.double.Dtype.StreamOrDevice"></span><span id="group__ops_1ga8d7cf9eb15e2daf1469058907e8abc85" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">arange</span></span></span><span class="sig-paren">(</span><span class="kt"><span class="pre">double</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">start</span></span>, <span class="kt"><span class="pre">double</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">stop</span></span>, <span class="n"><span class="pre">Dtype</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">dtype</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv46arangedd5Dtype14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    

<!-- -->

<span id="_CPPv36arangedd14StreamOrDevice"></span><span id="_CPPv26arangedd14StreamOrDevice"></span><span id="arange__double.double.StreamOrDevice"></span><span id="group__ops_1ga74566a14e69ba6a25f5a35e7ade5c282" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">arange</span></span></span><span class="sig-paren">(</span><span class="kt"><span class="pre">double</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">start</span></span>, <span class="kt"><span class="pre">double</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">stop</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv46arangedd14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    

<!-- -->

<span id="_CPPv36aranged5Dtype14StreamOrDevice"></span><span id="_CPPv26aranged5Dtype14StreamOrDevice"></span><span id="arange__double.Dtype.StreamOrDevice"></span><span id="group__ops_1ga345aa27af3dae3646b8b4b1068e89a3e" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">arange</span></span></span><span class="sig-paren">(</span><span class="kt"><span class="pre">double</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">stop</span></span>, <span class="n"><span class="pre">Dtype</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">dtype</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv46aranged5Dtype14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    

<!-- -->

<span id="_CPPv36aranged14StreamOrDevice"></span><span id="_CPPv26aranged14StreamOrDevice"></span><span id="arange__double.StreamOrDevice"></span><span id="group__ops_1gaae179075d0fe23f4bd53fdf8c41f4c70" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">arange</span></span></span><span class="sig-paren">(</span><span class="kt"><span class="pre">double</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">stop</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv46aranged14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    

<!-- -->

<span id="_CPPv36arangeiii14StreamOrDevice"></span><span id="_CPPv26arangeiii14StreamOrDevice"></span><span id="arange__i.i.i.StreamOrDevice"></span><span id="group__ops_1ga6b945f513077c2978afc1a952c884860" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">arange</span></span></span><span class="sig-paren">(</span><span class="kt"><span class="pre">int</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">start</span></span>, <span class="kt"><span class="pre">int</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">stop</span></span>, <span class="kt"><span class="pre">int</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">step</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv46arangeiii14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    

<!-- -->

<span id="_CPPv36arangeii14StreamOrDevice"></span><span id="_CPPv26arangeii14StreamOrDevice"></span><span id="arange__i.i.StreamOrDevice"></span><span id="group__ops_1ga1c39fcc6eaa1c1867735c7f849d708d6" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">arange</span></span></span><span class="sig-paren">(</span><span class="kt"><span class="pre">int</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">start</span></span>, <span class="kt"><span class="pre">int</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">stop</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv46arangeii14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    

<!-- -->

<span id="_CPPv36arangei14StreamOrDevice"></span><span id="_CPPv26arangei14StreamOrDevice"></span><span id="arange__i.StreamOrDevice"></span><span id="group__ops_1gafe6e4580452c873cac294f16129e633f" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">arange</span></span></span><span class="sig-paren">(</span><span class="kt"><span class="pre">int</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">stop</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv46arangei14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    

<!-- -->

<span id="_CPPv38linspaceddi5Dtype14StreamOrDevice"></span><span id="_CPPv28linspaceddi5Dtype14StreamOrDevice"></span><span id="linspace__double.double.i.Dtype.StreamOrDevice"></span><span id="group__ops_1ga968bcabed902311dcfbd903b0fb886ec" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">linspace</span></span></span><span class="sig-paren">(</span><span class="kt"><span class="pre">double</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">start</span></span>, <span class="kt"><span class="pre">double</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">stop</span></span>, <span class="kt"><span class="pre">int</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">num</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="m"><span class="pre">50</span></span>, <span class="n"><span class="pre">Dtype</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">dtype</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="n"><span class="pre">float32</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv48linspaceddi5Dtype14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
A 1D array of <span class="pre">`num`</span> evenly spaced numbers in
the range
<span class="pre">`[start,`</span>` `<span class="pre">`stop]`</span>

<!-- -->

<span id="_CPPv36astype5array5Dtype14StreamOrDevice"></span><span id="_CPPv26astype5array5Dtype14StreamOrDevice"></span><span id="astype__array.Dtype.StreamOrDevice"></span><span id="group__ops_1ga0e58c24fc5668e5a521e5b45e8370a62" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">astype</span></span></span><span class="sig-paren">(</span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">a</span></span>, <span class="n"><span class="pre">Dtype</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">dtype</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv46astype5array5Dtype14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Convert an array to the given data type.

<!-- -->

<span id="_CPPv310as_strided5array5Shape7Strides6size_t14StreamOrDevice"></span><span id="_CPPv210as_strided5array5Shape7Strides6size_t14StreamOrDevice"></span><span id="as_strided__array.Shape.Strides.s.StreamOrDevice"></span><span id="group__ops_1ga6085b03f2662ef2a61de523fd609f3bf" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">as_strided</span></span></span><span class="sig-paren">(</span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">a</span></span>, <span class="n"><span class="pre">Shape</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">shape</span></span>, <span class="n"><span class="pre">Strides</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">strides</span></span>, <span class="n"><span class="pre">size_t</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">offset</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv410as_strided5array5Shape7Strides6size_t14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Create a view of an array with the given shape and strides.

<!-- -->

<span id="_CPPv34copy5array14StreamOrDevice"></span><span id="_CPPv24copy5array14StreamOrDevice"></span><span id="copy__array.StreamOrDevice"></span><span id="group__ops_1gae306e93af12f774bd80bad6c231b09d6" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">copy</span></span></span><span class="sig-paren">(</span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">a</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv44copy5array14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Copy another array.

<!-- -->

<span id="_CPPv34full5Shape5array5Dtype14StreamOrDevice"></span><span id="_CPPv24full5Shape5array5Dtype14StreamOrDevice"></span><span id="full__Shape.array.Dtype.StreamOrDevice"></span><span id="group__ops_1ga1cf232308668fe3f4214c8b895ed4aee" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">full</span></span></span><span class="sig-paren">(</span><span class="n"><span class="pre">Shape</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">shape</span></span>, <span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">vals</span></span>, <span class="n"><span class="pre">Dtype</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">dtype</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv44full5Shape5array5Dtype14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Fill an array of the given shape with the given value(s).

<!-- -->

<span id="_CPPv34full5Shape5array14StreamOrDevice"></span><span id="_CPPv24full5Shape5array14StreamOrDevice"></span><span id="full__Shape.array.StreamOrDevice"></span><span id="group__ops_1ga59f6c844cbb173e108c3eeb11801f8c6" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">full</span></span></span><span class="sig-paren">(</span><span class="n"><span class="pre">Shape</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">shape</span></span>, <span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">vals</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv44full5Shape5array14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    

<!-- -->

<span id="_CPPv3I0E4full5Shape1T5Dtype14StreamOrDevice"></span><span id="_CPPv2I0E4full5Shape1T5Dtype14StreamOrDevice"></span><span class="k"><span class="pre">template</span></span><span class="p"><span class="pre">\<</span></span><span class="k"><span class="pre">typename</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">T</span></span></span><span class="p"><span class="pre">\></span></span>  
<span id="group__ops_1gaf073760b7b51fe35932da0d81c531a55" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">full</span></span></span><span class="sig-paren">(</span><span class="n"><span class="pre">Shape</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">shape</span></span>, <a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4I0E4full5array5Shape1T5Dtype14StreamOrDevice"
class="reference internal" title="full::T"><span class="n"><span
class="pre">T</span></span></a><span class="w"> </span><span class="n sig-param"><span class="pre">val</span></span>, <span class="n"><span class="pre">Dtype</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">dtype</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4I0E4full5array5Shape1T5Dtype14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    

<!-- -->

<span id="_CPPv3I0E4full5Shape1T14StreamOrDevice"></span><span id="_CPPv2I0E4full5Shape1T14StreamOrDevice"></span><span class="k"><span class="pre">template</span></span><span class="p"><span class="pre">\<</span></span><span class="k"><span class="pre">typename</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">T</span></span></span><span class="p"><span class="pre">\></span></span>  
<span id="group__ops_1gaf6f2cce92aff9b71756a3cc3c961fd5a" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">full</span></span></span><span class="sig-paren">(</span><span class="n"><span class="pre">Shape</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">shape</span></span>, <a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4I0E4full5array5Shape1T14StreamOrDevice"
class="reference internal" title="full::T"><span class="n"><span
class="pre">T</span></span></a><span class="w"> </span><span class="n sig-param"><span class="pre">val</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4I0E4full5array5Shape1T14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    

<!-- -->

<span id="_CPPv35zerosRK5Shape5Dtype14StreamOrDevice"></span><span id="_CPPv25zerosRK5Shape5Dtype14StreamOrDevice"></span><span id="zeros__ShapeCR.Dtype.StreamOrDevice"></span><span id="group__ops_1gae2cace3b388cec4e520659a91879e1c1" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">zeros</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">Shape</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">shape</span></span>, <span class="n"><span class="pre">Dtype</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">dtype</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv45zerosRK5Shape5Dtype14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Fill an array of the given shape with zeros.

<!-- -->

<span id="_CPPv35zerosRK5Shape14StreamOrDevice"></span><span id="_CPPv25zerosRK5Shape14StreamOrDevice"></span><span id="zeros__ShapeCR.StreamOrDevice"></span><span id="group__ops_1gac8aa722f5e798819b7091693173f1f36" class="target"></span><span class="k"><span class="pre">inline</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">zeros</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">Shape</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">shape</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv45zerosRK5Shape14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    

<!-- -->

<span id="_CPPv310zeros_likeRK5array14StreamOrDevice"></span><span id="_CPPv210zeros_likeRK5array14StreamOrDevice"></span><span id="zeros_like__arrayCR.StreamOrDevice"></span><span id="group__ops_1gafbb857094d784b38c78683a091ffdbde" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">zeros_like</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv410zeros_likeRK5array14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    

<!-- -->

<span id="_CPPv34onesRK5Shape5Dtype14StreamOrDevice"></span><span id="_CPPv24onesRK5Shape5Dtype14StreamOrDevice"></span><span id="ones__ShapeCR.Dtype.StreamOrDevice"></span><span id="group__ops_1gae0069146cf8c819b15ba29aa7231a3f0" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">ones</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">Shape</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">shape</span></span>, <span class="n"><span class="pre">Dtype</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">dtype</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv44onesRK5Shape5Dtype14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Fill an array of the given shape with ones.

<!-- -->

<span id="_CPPv34onesRK5Shape14StreamOrDevice"></span><span id="_CPPv24onesRK5Shape14StreamOrDevice"></span><span id="ones__ShapeCR.StreamOrDevice"></span><span id="group__ops_1gace4cf016562af58991f9f961170e156f" class="target"></span><span class="k"><span class="pre">inline</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">ones</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">Shape</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">shape</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv44onesRK5Shape14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    

<!-- -->

<span id="_CPPv39ones_likeRK5array14StreamOrDevice"></span><span id="_CPPv29ones_likeRK5array14StreamOrDevice"></span><span id="ones_like__arrayCR.StreamOrDevice"></span><span id="group__ops_1ga94f8d3b1906fee99da9cbe39f7be7d42" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">ones_like</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv49ones_likeRK5array14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    

<!-- -->

<span id="_CPPv33eyeiii5Dtype14StreamOrDevice"></span><span id="_CPPv23eyeiii5Dtype14StreamOrDevice"></span><span id="eye__i.i.i.Dtype.StreamOrDevice"></span><span id="group__ops_1ga45e9e68246b0d1cf03c3cc9c9e7e6ae3" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">eye</span></span></span><span class="sig-paren">(</span><span class="kt"><span class="pre">int</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">n</span></span>, <span class="kt"><span class="pre">int</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">m</span></span>, <span class="kt"><span class="pre">int</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">k</span></span>, <span class="n"><span class="pre">Dtype</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">dtype</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv43eyeiii5Dtype14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Fill an array of the given shape (n,m) with ones in the specified
diagonal k, and zeros everywhere else.

<!-- -->

<span id="_CPPv33eyei5Dtype14StreamOrDevice"></span><span id="_CPPv23eyei5Dtype14StreamOrDevice"></span><span id="eye__i.Dtype.StreamOrDevice"></span><span id="group__ops_1ga2c9011310a1fa7c82f942f54102c36dd" class="target"></span><span class="k"><span class="pre">inline</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">eye</span></span></span><span class="sig-paren">(</span><span class="kt"><span class="pre">int</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">n</span></span>, <span class="n"><span class="pre">Dtype</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">dtype</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv43eyei5Dtype14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    

<!-- -->

<span id="_CPPv33eyeii14StreamOrDevice"></span><span id="_CPPv23eyeii14StreamOrDevice"></span><span id="eye__i.i.StreamOrDevice"></span><span id="group__ops_1ga61657db78ef35d41112d362c869c25d2" class="target"></span><span class="k"><span class="pre">inline</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">eye</span></span></span><span class="sig-paren">(</span><span class="kt"><span class="pre">int</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">n</span></span>, <span class="kt"><span class="pre">int</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">m</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv43eyeii14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    

<!-- -->

<span id="_CPPv33eyeiii14StreamOrDevice"></span><span id="_CPPv23eyeiii14StreamOrDevice"></span><span id="eye__i.i.i.StreamOrDevice"></span><span id="group__ops_1ga908a15b42834be498a46856c99dfc779" class="target"></span><span class="k"><span class="pre">inline</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">eye</span></span></span><span class="sig-paren">(</span><span class="kt"><span class="pre">int</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">n</span></span>, <span class="kt"><span class="pre">int</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">m</span></span>, <span class="kt"><span class="pre">int</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">k</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv43eyeiii14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    

<!-- -->

<span id="_CPPv33eyei14StreamOrDevice"></span><span id="_CPPv23eyei14StreamOrDevice"></span><span id="eye__i.StreamOrDevice"></span><span id="group__ops_1gab777fcf6d4a89172c69ec3492548dc0f" class="target"></span><span class="k"><span class="pre">inline</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">eye</span></span></span><span class="sig-paren">(</span><span class="kt"><span class="pre">int</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">n</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv43eyei14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    

<!-- -->

<span id="_CPPv38identityi5Dtype14StreamOrDevice"></span><span id="_CPPv28identityi5Dtype14StreamOrDevice"></span><span id="identity__i.Dtype.StreamOrDevice"></span><span id="group__ops_1ga484eaa10d5e19a4ca46d3a9cd9fab600" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">identity</span></span></span><span class="sig-paren">(</span><span class="kt"><span class="pre">int</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">n</span></span>, <span class="n"><span class="pre">Dtype</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">dtype</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv48identityi5Dtype14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Create a square matrix of shape (n,n) of zeros, and ones in the major
diagonal.

<!-- -->

<span id="_CPPv38identityi14StreamOrDevice"></span><span id="_CPPv28identityi14StreamOrDevice"></span><span id="identity__i.StreamOrDevice"></span><span id="group__ops_1gad994d65ac6019c26b5ad6c41179d3424" class="target"></span><span class="k"><span class="pre">inline</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">identity</span></span></span><span class="sig-paren">(</span><span class="kt"><span class="pre">int</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">n</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv48identityi14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    

<!-- -->

<span id="_CPPv33triiii5Dtype14StreamOrDevice"></span><span id="_CPPv23triiii5Dtype14StreamOrDevice"></span><span id="tri__i.i.i.Dtype.StreamOrDevice"></span><span id="group__ops_1ga4f3389e5b89e70e862e7d2b40d6c7f78" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">tri</span></span></span><span class="sig-paren">(</span><span class="kt"><span class="pre">int</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">n</span></span>, <span class="kt"><span class="pre">int</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">m</span></span>, <span class="kt"><span class="pre">int</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">k</span></span>, <span class="n"><span class="pre">Dtype</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">type</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv43triiii5Dtype14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    

<!-- -->

<span id="_CPPv33trii5Dtype14StreamOrDevice"></span><span id="_CPPv23trii5Dtype14StreamOrDevice"></span><span id="tri__i.Dtype.StreamOrDevice"></span><span id="group__ops_1gac19a1bd6ed6d5c7bc9d258820189dbb5" class="target"></span><span class="k"><span class="pre">inline</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">tri</span></span></span><span class="sig-paren">(</span><span class="kt"><span class="pre">int</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">n</span></span>, <span class="n"><span class="pre">Dtype</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">type</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv43trii5Dtype14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    

<!-- -->

<span id="_CPPv34tril5arrayi14StreamOrDevice"></span><span id="_CPPv24tril5arrayi14StreamOrDevice"></span><span id="tril__array.i.StreamOrDevice"></span><span id="group__ops_1ga83e0bb45dc770cf014531d873b78c5a2" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">tril</span></span></span><span class="sig-paren">(</span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">x</span></span>, <span class="kt"><span class="pre">int</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">k</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="m"><span class="pre">0</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv44tril5arrayi14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    

<!-- -->

<span id="_CPPv34triu5arrayi14StreamOrDevice"></span><span id="_CPPv24triu5arrayi14StreamOrDevice"></span><span id="triu__array.i.StreamOrDevice"></span><span id="group__ops_1gaa9df5917876eeb0cb28b7fa81f880412" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">triu</span></span></span><span class="sig-paren">(</span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">x</span></span>, <span class="kt"><span class="pre">int</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">k</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="m"><span class="pre">0</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv44triu5arrayi14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    

<!-- -->

<span id="_CPPv37reshapeRK5array5Shape14StreamOrDevice"></span><span id="_CPPv27reshapeRK5array5Shape14StreamOrDevice"></span><span id="reshape__arrayCR.Shape.StreamOrDevice"></span><span id="group__ops_1ga084f03ce2b22258afb7c8b45e17af828" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">reshape</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="n"><span class="pre">Shape</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">shape</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv47reshapeRK5array5Shape14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Reshape an array to the given shape.

<!-- -->

<span id="_CPPv39unflattenRK5arrayi5Shape14StreamOrDevice"></span><span id="_CPPv29unflattenRK5arrayi5Shape14StreamOrDevice"></span><span id="unflatten__arrayCR.i.Shape.StreamOrDevice"></span><span id="group__ops_1ga666bcc2187a144247e8c0c224b016625" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">unflatten</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="kt"><span class="pre">int</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">axis</span></span>, <span class="n"><span class="pre">Shape</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">shape</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv49unflattenRK5arrayi5Shape14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Unflatten the axis to the given shape.

<!-- -->

<span id="_CPPv37flattenRK5arrayii14StreamOrDevice"></span><span id="_CPPv27flattenRK5arrayii14StreamOrDevice"></span><span id="flatten__arrayCR.i.i.StreamOrDevice"></span><span id="group__ops_1ga50aa98754b412bb57c083f6e3e95061f" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">flatten</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="kt"><span class="pre">int</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">start_axis</span></span>, <span class="kt"><span class="pre">int</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">end_axis</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="o"><span class="pre">-</span></span><span class="m"><span class="pre">1</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv47flattenRK5arrayii14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Flatten the dimensions in the range
<span class="pre">`[start_axis,`</span>` `<span class="pre">`end_axis]`</span>
.

<!-- -->

<span id="_CPPv37flattenRK5array14StreamOrDevice"></span><span id="_CPPv27flattenRK5array14StreamOrDevice"></span><span id="flatten__arrayCR.StreamOrDevice"></span><span id="group__ops_1gaa6adbc9c86f0ab27d8810a02e9e719fd" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">flatten</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv47flattenRK5array14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Flatten the array to 1D.

<!-- -->

<span id="_CPPv318hadamard_transformRK5arrayNSt8optionalIfEE14StreamOrDevice"></span><span id="_CPPv218hadamard_transformRK5arrayNSt8optionalIfEE14StreamOrDevice"></span><span id="hadamard_transform__arrayCR.std::optional:float:.StreamOrDevice"></span><span id="group__ops_1ga872d2c1806e67ce2596b24d056681074" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">hadamard_transform</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4StRK5arraybi14StreamOrDevice"
class="reference internal" title="std"><span class="n"><span
class="pre">std</span></span></a><span class="p"><span class="pre">::</span></span><span class="n"><span class="pre">optional</span></span><span class="p"><span class="pre">\<</span></span><span class="kt"><span class="pre">float</span></span><span class="p"><span class="pre">\></span></span><span class="w"> </span><span class="n sig-param"><span class="pre">scale</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4StRK5arraybi14StreamOrDevice"
class="reference internal" title="std"><span class="n"><span
class="pre">std</span></span></a><span class="p"><span class="pre">::</span></span><span class="n"><span class="pre">nullopt</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv418hadamard_transformRK5arrayNSt8optionalIfEE14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Multiply the array by the Hadamard matrix of corresponding size.

<!-- -->

<span id="_CPPv37squeezeRK5arrayRKNSt6vectorIiEE14StreamOrDevice"></span><span id="_CPPv27squeezeRK5arrayRKNSt6vectorIiEE14StreamOrDevice"></span><span id="squeeze__arrayCR.std::vector:i:CR.StreamOrDevice"></span><span id="group__ops_1ga710daa7ec721bd4d3f326082cb195576" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">squeeze</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4StRK5arraybi14StreamOrDevice"
class="reference internal" title="std"><span class="n"><span
class="pre">std</span></span></a><span class="p"><span class="pre">::</span></span><span class="n"><span class="pre">vector</span></span><span class="p"><span class="pre">\<</span></span><span class="kt"><span class="pre">int</span></span><span class="p"><span class="pre">\></span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">axes</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv47squeezeRK5arrayRKNSt6vectorIiEE14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Remove singleton dimensions at the given axes.

<!-- -->

<span id="_CPPv37squeezeRK5arrayi14StreamOrDevice"></span><span id="_CPPv27squeezeRK5arrayi14StreamOrDevice"></span><span id="squeeze__arrayCR.i.StreamOrDevice"></span><span id="group__ops_1ga700dd51b77379a3d2260a55783e8ebf3" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">squeeze</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="kt"><span class="pre">int</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">axis</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv47squeezeRK5arrayi14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Remove singleton dimensions at the given axis.

<!-- -->

<span id="_CPPv37squeezeRK5array14StreamOrDevice"></span><span id="_CPPv27squeezeRK5array14StreamOrDevice"></span><span id="squeeze__arrayCR.StreamOrDevice"></span><span id="group__ops_1ga58bad3c61fd85b95927a987ba1cf5dad" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">squeeze</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv47squeezeRK5array14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Remove all singleton dimensions.

<!-- -->

<span id="_CPPv311expand_dimsRK5arrayRKNSt6vectorIiEE14StreamOrDevice"></span><span id="_CPPv211expand_dimsRK5arrayRKNSt6vectorIiEE14StreamOrDevice"></span><span id="expand_dims__arrayCR.std::vector:i:CR.StreamOrDevice"></span><span id="group__ops_1ga717f11149a8c7b4cc3e33bbcc0a97133" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">expand_dims</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4StRK5arraybi14StreamOrDevice"
class="reference internal" title="std"><span class="n"><span
class="pre">std</span></span></a><span class="p"><span class="pre">::</span></span><span class="n"><span class="pre">vector</span></span><span class="p"><span class="pre">\<</span></span><span class="kt"><span class="pre">int</span></span><span class="p"><span class="pre">\></span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">axes</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv411expand_dimsRK5arrayRKNSt6vectorIiEE14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Add a singleton dimension at the given axes.

<!-- -->

<span id="_CPPv311expand_dimsRK5arrayi14StreamOrDevice"></span><span id="_CPPv211expand_dimsRK5arrayi14StreamOrDevice"></span><span id="expand_dims__arrayCR.i.StreamOrDevice"></span><span id="group__ops_1ga7a80adb4a5a36d18b5f234d4b034950a" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">expand_dims</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="kt"><span class="pre">int</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">axis</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv411expand_dimsRK5arrayi14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Add a singleton dimension at the given axis.

<!-- -->

<span id="_CPPv35sliceRK5array5Shape5Shape5Shape14StreamOrDevice"></span><span id="_CPPv25sliceRK5array5Shape5Shape5Shape14StreamOrDevice"></span><span id="slice__arrayCR.Shape.Shape.Shape.StreamOrDevice"></span><span id="group__ops_1ga29718cd5005dbcde0396b6fd65cc041d" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">slice</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="n"><span class="pre">Shape</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">start</span></span>, <span class="n"><span class="pre">Shape</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">stop</span></span>, <span class="n"><span class="pre">Shape</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">strides</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv45sliceRK5array5Shape5Shape5Shape14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Slice an array.

<!-- -->

<span id="_CPPv35sliceRK5arrayNSt16initializer_listIiEE5Shape5Shape14StreamOrDevice"></span><span id="_CPPv25sliceRK5arrayNSt16initializer_listIiEE5Shape5Shape14StreamOrDevice"></span><span id="slice__arrayCR.std::initializer_list:i:.Shape.Shape.StreamOrDevice"></span><span id="group__ops_1gaedcbdf4040f5e6a02a74b1ed1c6c2ebc" class="target"></span><span class="k"><span class="pre">inline</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">slice</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4StRK5arraybi14StreamOrDevice"
class="reference internal" title="std"><span class="n"><span
class="pre">std</span></span></a><span class="p"><span class="pre">::</span></span><span class="n"><span class="pre">initializer_list</span></span><span class="p"><span class="pre">\<</span></span><span class="kt"><span class="pre">int</span></span><span class="p"><span class="pre">\></span></span><span class="w"> </span><span class="n sig-param"><span class="pre">start</span></span>, <span class="n"><span class="pre">Shape</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">stop</span></span>, <span class="n"><span class="pre">Shape</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">strides</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv45sliceRK5arrayNSt16initializer_listIiEE5Shape5Shape14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    

<!-- -->

<span id="_CPPv35sliceRK5array5Shape5Shape14StreamOrDevice"></span><span id="_CPPv25sliceRK5array5Shape5Shape14StreamOrDevice"></span><span id="slice__arrayCR.Shape.Shape.StreamOrDevice"></span><span id="group__ops_1gaec56dcb94d5e7f7b885fb60b4bf4aa9d" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">slice</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="n"><span class="pre">Shape</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">start</span></span>, <span class="n"><span class="pre">Shape</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">stop</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv45sliceRK5array5Shape5Shape14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Slice an array with a stride of 1 in each dimension.

<!-- -->

<span id="_CPPv35sliceRK5arrayRK5arrayNSt6vectorIiEE5Shape14StreamOrDevice"></span><span id="_CPPv25sliceRK5arrayRK5arrayNSt6vectorIiEE5Shape14StreamOrDevice"></span><span id="slice__arrayCR.arrayCR.std::vector:i:.Shape.StreamOrDevice"></span><span id="group__ops_1ga797996e53ea34317a55dc2f314736b89" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">slice</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">start</span></span>, <a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4StRK5arraybi14StreamOrDevice"
class="reference internal" title="std"><span class="n"><span
class="pre">std</span></span></a><span class="p"><span class="pre">::</span></span><span class="n"><span class="pre">vector</span></span><span class="p"><span class="pre">\<</span></span><span class="kt"><span class="pre">int</span></span><span class="p"><span class="pre">\></span></span><span class="w"> </span><span class="n sig-param"><span class="pre">axes</span></span>, <span class="n"><span class="pre">Shape</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">slice_size</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv45sliceRK5arrayRK5arrayNSt6vectorIiEE5Shape14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Slice an array with dynamic starting indices.

<!-- -->

<span id="_CPPv312slice_updateRK5arrayRK5array5Shape5Shape5Shape14StreamOrDevice"></span><span id="_CPPv212slice_updateRK5arrayRK5array5Shape5Shape5Shape14StreamOrDevice"></span><span id="slice_update__arrayCR.arrayCR.Shape.Shape.Shape.StreamOrDevice"></span><span id="group__ops_1ga7b7e786985d27789aaed20bb2f9509be" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">slice_update</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">src</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">update</span></span>, <span class="n"><span class="pre">Shape</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">start</span></span>, <span class="n"><span class="pre">Shape</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">stop</span></span>, <span class="n"><span class="pre">Shape</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">strides</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv412slice_updateRK5arrayRK5array5Shape5Shape5Shape14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Update a slice from the source array.

<!-- -->

<span id="_CPPv312slice_updateRK5arrayRK5array5Shape5Shape14StreamOrDevice"></span><span id="_CPPv212slice_updateRK5arrayRK5array5Shape5Shape14StreamOrDevice"></span><span id="slice_update__arrayCR.arrayCR.Shape.Shape.StreamOrDevice"></span><span id="group__ops_1ga3b7c6136ae5a38dd6457b65c1833aa67" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">slice_update</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">src</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">update</span></span>, <span class="n"><span class="pre">Shape</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">start</span></span>, <span class="n"><span class="pre">Shape</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">stop</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv412slice_updateRK5arrayRK5array5Shape5Shape14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Update a slice from the source array with stride 1 in each dimension.

<!-- -->

<span id="_CPPv312slice_updateRK5arrayRK5arrayRK5arrayNSt6vectorIiEE14StreamOrDevice"></span><span id="_CPPv212slice_updateRK5arrayRK5arrayRK5arrayNSt6vectorIiEE14StreamOrDevice"></span><span id="slice_update__arrayCR.arrayCR.arrayCR.std::vector:i:.StreamOrDevice"></span><span id="group__ops_1gacd906ffb96149a4998c321cbf2231d7e" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">slice_update</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">src</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">update</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">start</span></span>, <a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4StRK5arraybi14StreamOrDevice"
class="reference internal" title="std"><span class="n"><span
class="pre">std</span></span></a><span class="p"><span class="pre">::</span></span><span class="n"><span class="pre">vector</span></span><span class="p"><span class="pre">\<</span></span><span class="kt"><span class="pre">int</span></span><span class="p"><span class="pre">\></span></span><span class="w"> </span><span class="n sig-param"><span class="pre">axes</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv412slice_updateRK5arrayRK5arrayRK5arrayNSt6vectorIiEE14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Update a slice from the source array with dynamic starting indices.

<!-- -->

<span id="_CPPv35splitRK5arrayii14StreamOrDevice"></span><span id="_CPPv25splitRK5arrayii14StreamOrDevice"></span><span id="split__arrayCR.i.i.StreamOrDevice"></span><span id="group__ops_1ga7534290bceab5fb3831a05d67bebce7d" class="target"></span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4StRK5arraybi14StreamOrDevice"
class="reference internal" title="std"><span class="n"><span
class="pre">std</span></span></a><span class="p"><span class="pre">::</span></span><span class="n"><span class="pre">vector</span></span><span class="p"><span class="pre">\<</span></span><span class="n"><span class="pre">array</span></span><span class="p"><span class="pre">\></span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">split</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="kt"><span class="pre">int</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">num_splits</span></span>, <span class="kt"><span class="pre">int</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">axis</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv45splitRK5arrayii14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Split an array into sub-arrays along a given axis.

<!-- -->

<span id="_CPPv35splitRK5arrayi14StreamOrDevice"></span><span id="_CPPv25splitRK5arrayi14StreamOrDevice"></span><span id="split__arrayCR.i.StreamOrDevice"></span><span id="group__ops_1ga56882d24e5fde59c266774624c892d41" class="target"></span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4StRK5arraybi14StreamOrDevice"
class="reference internal" title="std"><span class="n"><span
class="pre">std</span></span></a><span class="p"><span class="pre">::</span></span><span class="n"><span class="pre">vector</span></span><span class="p"><span class="pre">\<</span></span><span class="n"><span class="pre">array</span></span><span class="p"><span class="pre">\></span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">split</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="kt"><span class="pre">int</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">num_splits</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv45splitRK5arrayi14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    

<!-- -->

<span id="_CPPv35splitRK5arrayRK5Shapei14StreamOrDevice"></span><span id="_CPPv25splitRK5arrayRK5Shapei14StreamOrDevice"></span><span id="split__arrayCR.ShapeCR.i.StreamOrDevice"></span><span id="group__ops_1ga19005414e7d8020cd6e94e06bf399b09" class="target"></span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4StRK5arraybi14StreamOrDevice"
class="reference internal" title="std"><span class="n"><span
class="pre">std</span></span></a><span class="p"><span class="pre">::</span></span><span class="n"><span class="pre">vector</span></span><span class="p"><span class="pre">\<</span></span><span class="n"><span class="pre">array</span></span><span class="p"><span class="pre">\></span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">split</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">Shape</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">indices</span></span>, <span class="kt"><span class="pre">int</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">axis</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv45splitRK5arrayRK5Shapei14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    

<!-- -->

<span id="_CPPv35splitRK5arrayRK5Shape14StreamOrDevice"></span><span id="_CPPv25splitRK5arrayRK5Shape14StreamOrDevice"></span><span id="split__arrayCR.ShapeCR.StreamOrDevice"></span><span id="group__ops_1ga9ea089f42b9940510619052b7166d9ac" class="target"></span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4StRK5arraybi14StreamOrDevice"
class="reference internal" title="std"><span class="n"><span
class="pre">std</span></span></a><span class="p"><span class="pre">::</span></span><span class="n"><span class="pre">vector</span></span><span class="p"><span class="pre">\<</span></span><span class="n"><span class="pre">array</span></span><span class="p"><span class="pre">\></span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">split</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">Shape</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">indices</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv45splitRK5arrayRK5Shape14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    

<!-- -->

<span id="_CPPv38meshgridRKNSt6vectorI5arrayEEbRKNSt6stringE14StreamOrDevice"></span><span id="_CPPv28meshgridRKNSt6vectorI5arrayEEbRKNSt6stringE14StreamOrDevice"></span><span id="meshgrid__std::vector:array:CR.b.ssCR.StreamOrDevice"></span><span id="group__ops_1ga5ecddb74ba7861eb82eca8653501d5dc" class="target"></span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4StRK5arraybi14StreamOrDevice"
class="reference internal" title="std"><span class="n"><span
class="pre">std</span></span></a><span class="p"><span class="pre">::</span></span><span class="n"><span class="pre">vector</span></span><span class="p"><span class="pre">\<</span></span><span class="n"><span class="pre">array</span></span><span class="p"><span class="pre">\></span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">meshgrid</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4StRK5arraybi14StreamOrDevice"
class="reference internal" title="std"><span class="n"><span
class="pre">std</span></span></a><span class="p"><span class="pre">::</span></span><span class="n"><span class="pre">vector</span></span><span class="p"><span class="pre">\<</span></span><span class="n"><span class="pre">array</span></span><span class="p"><span class="pre">\></span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">arrays</span></span>, <span class="kt"><span class="pre">bool</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">sparse</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="k"><span class="pre">false</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4StRK5arraybi14StreamOrDevice"
class="reference internal" title="std"><span class="n"><span
class="pre">std</span></span></a><span class="p"><span class="pre">::</span></span><span class="n"><span class="pre">string</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">indexing</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="s"><span class="pre">"xy"</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv48meshgridRKNSt6vectorI5arrayEEbRKNSt6stringE14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
A vector of coordinate arrays from coordinate vectors.

<!-- -->

<span id="_CPPv34clipRK5arrayRKNSt8optionalI5arrayEERKNSt8optionalI5arrayEE14StreamOrDevice"></span><span id="_CPPv24clipRK5arrayRKNSt8optionalI5arrayEERKNSt8optionalI5arrayEE14StreamOrDevice"></span><span id="clip__arrayCR.std::optional:array:CR.std::optional:array:CR.StreamOrDevice"></span><span id="group__ops_1ga157cd7c23f9b306fee2e1eb2b9bf1dd8" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">clip</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4StRK5arraybi14StreamOrDevice"
class="reference internal" title="std"><span class="n"><span
class="pre">std</span></span></a><span class="p"><span class="pre">::</span></span><span class="n"><span class="pre">optional</span></span><span class="p"><span class="pre">\<</span></span><span class="n"><span class="pre">array</span></span><span class="p"><span class="pre">\></span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a_min</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4StRK5arraybi14StreamOrDevice"
class="reference internal" title="std"><span class="n"><span
class="pre">std</span></span></a><span class="p"><span class="pre">::</span></span><span class="n"><span class="pre">nullopt</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4StRK5arraybi14StreamOrDevice"
class="reference internal" title="std"><span class="n"><span
class="pre">std</span></span></a><span class="p"><span class="pre">::</span></span><span class="n"><span class="pre">optional</span></span><span class="p"><span class="pre">\<</span></span><span class="n"><span class="pre">array</span></span><span class="p"><span class="pre">\></span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a_max</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4StRK5arraybi14StreamOrDevice"
class="reference internal" title="std"><span class="n"><span
class="pre">std</span></span></a><span class="p"><span class="pre">::</span></span><span class="n"><span class="pre">nullopt</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv44clipRK5arrayRKNSt8optionalI5arrayEERKNSt8optionalI5arrayEE14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Clip (limit) the values in an array.

<!-- -->

<span id="_CPPv311concatenateNSt6vectorI5arrayEEi14StreamOrDevice"></span><span id="_CPPv211concatenateNSt6vectorI5arrayEEi14StreamOrDevice"></span><span id="concatenate__std::vector:array:.i.StreamOrDevice"></span><span id="group__ops_1ga52838af566948b1b96e7aa00832071b3" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">concatenate</span></span></span><span class="sig-paren">(</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4StRK5arraybi14StreamOrDevice"
class="reference internal" title="std"><span class="n"><span
class="pre">std</span></span></a><span class="p"><span class="pre">::</span></span><span class="n"><span class="pre">vector</span></span><span class="p"><span class="pre">\<</span></span><span class="n"><span class="pre">array</span></span><span class="p"><span class="pre">\></span></span><span class="w"> </span><span class="n sig-param"><span class="pre">arrays</span></span>, <span class="kt"><span class="pre">int</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">axis</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv411concatenateNSt6vectorI5arrayEEi14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Concatenate arrays along a given axis.

<!-- -->

<span id="_CPPv311concatenateNSt6vectorI5arrayEE14StreamOrDevice"></span><span id="_CPPv211concatenateNSt6vectorI5arrayEE14StreamOrDevice"></span><span id="concatenate__std::vector:array:.StreamOrDevice"></span><span id="group__ops_1ga666ac69778984fafdc2f51d296270468" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">concatenate</span></span></span><span class="sig-paren">(</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4StRK5arraybi14StreamOrDevice"
class="reference internal" title="std"><span class="n"><span
class="pre">std</span></span></a><span class="p"><span class="pre">::</span></span><span class="n"><span class="pre">vector</span></span><span class="p"><span class="pre">\<</span></span><span class="n"><span class="pre">array</span></span><span class="p"><span class="pre">\></span></span><span class="w"> </span><span class="n sig-param"><span class="pre">arrays</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv411concatenateNSt6vectorI5arrayEE14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    

<!-- -->

<span id="_CPPv35stackRKNSt6vectorI5arrayEEi14StreamOrDevice"></span><span id="_CPPv25stackRKNSt6vectorI5arrayEEi14StreamOrDevice"></span><span id="stack__std::vector:array:CR.i.StreamOrDevice"></span><span id="group__ops_1gaf8f2ec2b98a4b59eca73d7471df6e032" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">stack</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4StRK5arraybi14StreamOrDevice"
class="reference internal" title="std"><span class="n"><span
class="pre">std</span></span></a><span class="p"><span class="pre">::</span></span><span class="n"><span class="pre">vector</span></span><span class="p"><span class="pre">\<</span></span><span class="n"><span class="pre">array</span></span><span class="p"><span class="pre">\></span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">arrays</span></span>, <span class="kt"><span class="pre">int</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">axis</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv45stackRKNSt6vectorI5arrayEEi14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Stack arrays along a new axis.

<!-- -->

<span id="_CPPv35stackRKNSt6vectorI5arrayEE14StreamOrDevice"></span><span id="_CPPv25stackRKNSt6vectorI5arrayEE14StreamOrDevice"></span><span id="stack__std::vector:array:CR.StreamOrDevice"></span><span id="group__ops_1ga82216209dce901296fc737fe8efa5c94" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">stack</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4StRK5arraybi14StreamOrDevice"
class="reference internal" title="std"><span class="n"><span
class="pre">std</span></span></a><span class="p"><span class="pre">::</span></span><span class="n"><span class="pre">vector</span></span><span class="p"><span class="pre">\<</span></span><span class="n"><span class="pre">array</span></span><span class="p"><span class="pre">\></span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">arrays</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv45stackRKNSt6vectorI5arrayEE14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    

<!-- -->

<span id="_CPPv36repeatRK5arrayii14StreamOrDevice"></span><span id="_CPPv26repeatRK5arrayii14StreamOrDevice"></span><span id="repeat__arrayCR.i.i.StreamOrDevice"></span><span id="group__ops_1gab49e3a687e826554ed1574186e8ae974" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">repeat</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">arr</span></span>, <span class="kt"><span class="pre">int</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">repeats</span></span>, <span class="kt"><span class="pre">int</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">axis</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv46repeatRK5arrayii14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Repeat an array along an axis.

<!-- -->

<span id="_CPPv36repeatRK5arrayi14StreamOrDevice"></span><span id="_CPPv26repeatRK5arrayi14StreamOrDevice"></span><span id="repeat__arrayCR.i.StreamOrDevice"></span><span id="group__ops_1ga4f75f5d5db999f02f43ecbc6dccf3ba6" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">repeat</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">arr</span></span>, <span class="kt"><span class="pre">int</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">repeats</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv46repeatRK5arrayi14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    

<!-- -->

<span id="_CPPv34tileRK5arrayNSt6vectorIiEE14StreamOrDevice"></span><span id="_CPPv24tileRK5arrayNSt6vectorIiEE14StreamOrDevice"></span><span id="tile__arrayCR.std::vector:i:.StreamOrDevice"></span><span id="group__ops_1gab105a57b9a4d84496fe1e4d60e13d361" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">tile</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">arr</span></span>, <a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4StRK5arraybi14StreamOrDevice"
class="reference internal" title="std"><span class="n"><span
class="pre">std</span></span></a><span class="p"><span class="pre">::</span></span><span class="n"><span class="pre">vector</span></span><span class="p"><span class="pre">\<</span></span><span class="kt"><span class="pre">int</span></span><span class="p"><span class="pre">\></span></span><span class="w"> </span><span class="n sig-param"><span class="pre">reps</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv44tileRK5arrayNSt6vectorIiEE14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    

<!-- -->

<span id="_CPPv39transposeRK5arrayNSt6vectorIiEE14StreamOrDevice"></span><span id="_CPPv29transposeRK5arrayNSt6vectorIiEE14StreamOrDevice"></span><span id="transpose__arrayCR.std::vector:i:.StreamOrDevice"></span><span id="group__ops_1gac1869f3b7094869b44fe7ac4ce58638b" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">transpose</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4StRK5arraybi14StreamOrDevice"
class="reference internal" title="std"><span class="n"><span
class="pre">std</span></span></a><span class="p"><span class="pre">::</span></span><span class="n"><span class="pre">vector</span></span><span class="p"><span class="pre">\<</span></span><span class="kt"><span class="pre">int</span></span><span class="p"><span class="pre">\></span></span><span class="w"> </span><span class="n sig-param"><span class="pre">axes</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv49transposeRK5arrayNSt6vectorIiEE14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Permutes the dimensions according to the given axes.

<!-- -->

<span id="_CPPv39transposeRK5arrayNSt16initializer_listIiEE14StreamOrDevice"></span><span id="_CPPv29transposeRK5arrayNSt16initializer_listIiEE14StreamOrDevice"></span><span id="transpose__arrayCR.std::initializer_list:i:.StreamOrDevice"></span><span id="group__ops_1ga260ac332956f3a6bf1dfdb9095c84dc5" class="target"></span><span class="k"><span class="pre">inline</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">transpose</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4StRK5arraybi14StreamOrDevice"
class="reference internal" title="std"><span class="n"><span
class="pre">std</span></span></a><span class="p"><span class="pre">::</span></span><span class="n"><span class="pre">initializer_list</span></span><span class="p"><span class="pre">\<</span></span><span class="kt"><span class="pre">int</span></span><span class="p"><span class="pre">\></span></span><span class="w"> </span><span class="n sig-param"><span class="pre">axes</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv49transposeRK5arrayNSt16initializer_listIiEE14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    

<!-- -->

<span id="_CPPv38swapaxesRK5arrayii14StreamOrDevice"></span><span id="_CPPv28swapaxesRK5arrayii14StreamOrDevice"></span><span id="swapaxes__arrayCR.i.i.StreamOrDevice"></span><span id="group__ops_1gabc46eed81ab6c6247903e4ec0c4ec1fb" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">swapaxes</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="kt"><span class="pre">int</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">axis1</span></span>, <span class="kt"><span class="pre">int</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">axis2</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv48swapaxesRK5arrayii14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Swap two axes of an array.

<!-- -->

<span id="_CPPv38moveaxisRK5arrayii14StreamOrDevice"></span><span id="_CPPv28moveaxisRK5arrayii14StreamOrDevice"></span><span id="moveaxis__arrayCR.i.i.StreamOrDevice"></span><span id="group__ops_1ga24067d10a842db2c9d509ea48135a2c3" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">moveaxis</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="kt"><span class="pre">int</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">source</span></span>, <span class="kt"><span class="pre">int</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">destination</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv48moveaxisRK5arrayii14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Move an axis of an array.

<!-- -->

<span id="_CPPv33padRK5arrayRKNSt6vectorIiEERK5ShapeRK5ShapeRK5arrayRKNSt6stringE14StreamOrDevice"></span><span id="_CPPv23padRK5arrayRKNSt6vectorIiEERK5ShapeRK5ShapeRK5arrayRKNSt6stringE14StreamOrDevice"></span><span id="pad__arrayCR.std::vector:i:CR.ShapeCR.ShapeCR.arrayCR.ssCR.StreamOrDevice"></span><span id="group__ops_1gab95ebd20bd7c6d1c840007cc020cbc0c" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">pad</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4StRK5arraybi14StreamOrDevice"
class="reference internal" title="std"><span class="n"><span
class="pre">std</span></span></a><span class="p"><span class="pre">::</span></span><span class="n"><span class="pre">vector</span></span><span class="p"><span class="pre">\<</span></span><span class="kt"><span class="pre">int</span></span><span class="p"><span class="pre">\></span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">axes</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">Shape</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">low_pad_size</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">Shape</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">high_pad_size</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">pad_value</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="p"><span class="pre">(</span></span><span class="m"><span class="pre">0</span></span><span class="p"><span class="pre">)</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4StRK5arraybi14StreamOrDevice"
class="reference internal" title="std"><span class="n"><span
class="pre">std</span></span></a><span class="p"><span class="pre">::</span></span><span class="n"><span class="pre">string</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">mode</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="s"><span class="pre">"constant"</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv43padRK5arrayRKNSt6vectorIiEERK5ShapeRK5ShapeRK5arrayRKNSt6stringE14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Pad an array with a constant value.

<!-- -->

<span id="_CPPv33padRK5arrayRKNSt6vectorINSt4pairIiiEEEERK5arrayRKNSt6stringE14StreamOrDevice"></span><span id="_CPPv23padRK5arrayRKNSt6vectorINSt4pairIiiEEEERK5arrayRKNSt6stringE14StreamOrDevice"></span><span id="pad__arrayCR.std::vector:std::pair:i.i::CR.arrayCR.ssCR.StreamOrDevice"></span><span id="group__ops_1gad89f464e92c356faab1f1c2e763b1fb7" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">pad</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4StRK5arraybi14StreamOrDevice"
class="reference internal" title="std"><span class="n"><span
class="pre">std</span></span></a><span class="p"><span class="pre">::</span></span><span class="n"><span class="pre">vector</span></span><span class="p"><span class="pre">\<</span></span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4StRK5arraybi14StreamOrDevice"
class="reference internal" title="std"><span class="n"><span
class="pre">std</span></span></a><span class="p"><span class="pre">::</span></span><span class="n"><span class="pre">pair</span></span><span class="p"><span class="pre">\<</span></span><span class="kt"><span class="pre">int</span></span><span class="p"><span class="pre">,</span></span><span class="w"> </span><span class="kt"><span class="pre">int</span></span><span class="p"><span class="pre">\></span></span><span class="p"><span class="pre">\></span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">pad_width</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">pad_value</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="p"><span class="pre">(</span></span><span class="m"><span class="pre">0</span></span><span class="p"><span class="pre">)</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4StRK5arraybi14StreamOrDevice"
class="reference internal" title="std"><span class="n"><span
class="pre">std</span></span></a><span class="p"><span class="pre">::</span></span><span class="n"><span class="pre">string</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">mode</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="s"><span class="pre">"constant"</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv43padRK5arrayRKNSt6vectorINSt4pairIiiEEEERK5arrayRKNSt6stringE14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Pad an array with a constant value along all axes.

<!-- -->

<span id="_CPPv33padRK5arrayRKNSt4pairIiiEERK5arrayRKNSt6stringE14StreamOrDevice"></span><span id="_CPPv23padRK5arrayRKNSt4pairIiiEERK5arrayRKNSt6stringE14StreamOrDevice"></span><span id="pad__arrayCR.std::pair:i.i:CR.arrayCR.ssCR.StreamOrDevice"></span><span id="group__ops_1gab2d5a17d3d5225fed34905e786c31c5f" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">pad</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4StRK5arraybi14StreamOrDevice"
class="reference internal" title="std"><span class="n"><span
class="pre">std</span></span></a><span class="p"><span class="pre">::</span></span><span class="n"><span class="pre">pair</span></span><span class="p"><span class="pre">\<</span></span><span class="kt"><span class="pre">int</span></span><span class="p"><span class="pre">,</span></span><span class="w"> </span><span class="kt"><span class="pre">int</span></span><span class="p"><span class="pre">\></span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">pad_width</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">pad_value</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="p"><span class="pre">(</span></span><span class="m"><span class="pre">0</span></span><span class="p"><span class="pre">)</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4StRK5arraybi14StreamOrDevice"
class="reference internal" title="std"><span class="n"><span
class="pre">std</span></span></a><span class="p"><span class="pre">::</span></span><span class="n"><span class="pre">string</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">mode</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="s"><span class="pre">"constant"</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv43padRK5arrayRKNSt4pairIiiEERK5arrayRKNSt6stringE14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    

<!-- -->

<span id="_CPPv33padRK5arrayiRK5arrayRKNSt6stringE14StreamOrDevice"></span><span id="_CPPv23padRK5arrayiRK5arrayRKNSt6stringE14StreamOrDevice"></span><span id="pad__arrayCR.i.arrayCR.ssCR.StreamOrDevice"></span><span id="group__ops_1gaa73ac5674467b1d5c74de6fef7204c44" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">pad</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="kt"><span class="pre">int</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">pad_width</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">pad_value</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="p"><span class="pre">(</span></span><span class="m"><span class="pre">0</span></span><span class="p"><span class="pre">)</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4StRK5arraybi14StreamOrDevice"
class="reference internal" title="std"><span class="n"><span
class="pre">std</span></span></a><span class="p"><span class="pre">::</span></span><span class="n"><span class="pre">string</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">mode</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="s"><span class="pre">"constant"</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv43padRK5arrayiRK5arrayRKNSt6stringE14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    

<!-- -->

<span id="_CPPv39transposeRK5array14StreamOrDevice"></span><span id="_CPPv29transposeRK5array14StreamOrDevice"></span><span id="transpose__arrayCR.StreamOrDevice"></span><span id="group__ops_1ga68da0176fefbe0c0096783c6fd926c6a" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">transpose</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv49transposeRK5array14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Permutes the dimensions in reverse order.

<!-- -->

<span id="_CPPv312broadcast_toRK5arrayRK5Shape14StreamOrDevice"></span><span id="_CPPv212broadcast_toRK5arrayRK5Shape14StreamOrDevice"></span><span id="broadcast_to__arrayCR.ShapeCR.StreamOrDevice"></span><span id="group__ops_1ga2fd5891f11593b7f09550e884f969013" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">broadcast_to</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">Shape</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">shape</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv412broadcast_toRK5arrayRK5Shape14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Broadcast an array to a given shape.

<!-- -->

<span id="_CPPv316broadcast_arraysRKNSt6vectorI5arrayEE14StreamOrDevice"></span><span id="_CPPv216broadcast_arraysRKNSt6vectorI5arrayEE14StreamOrDevice"></span><span id="broadcast_arrays__std::vector:array:CR.StreamOrDevice"></span><span id="group__ops_1gab783890428b596f715dc7dd2057eae99" class="target"></span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4StRK5arraybi14StreamOrDevice"
class="reference internal" title="std"><span class="n"><span
class="pre">std</span></span></a><span class="p"><span class="pre">::</span></span><span class="n"><span class="pre">vector</span></span><span class="p"><span class="pre">\<</span></span><span class="n"><span class="pre">array</span></span><span class="p"><span class="pre">\></span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">broadcast_arrays</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4StRK5arraybi14StreamOrDevice"
class="reference internal" title="std"><span class="n"><span
class="pre">std</span></span></a><span class="p"><span class="pre">::</span></span><span class="n"><span class="pre">vector</span></span><span class="p"><span class="pre">\<</span></span><span class="n"><span class="pre">array</span></span><span class="p"><span class="pre">\></span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">inputs</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv416broadcast_arraysRKNSt6vectorI5arrayEE14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Broadcast a vector of arrays against one another.

<!-- -->

<span id="_CPPv35equalRK5arrayRK5array14StreamOrDevice"></span><span id="_CPPv25equalRK5arrayRK5array14StreamOrDevice"></span><span id="equal__arrayCR.arrayCR.StreamOrDevice"></span><span id="group__ops_1ga33638dc3a9972dd02be12d0eb85f9bde" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">equal</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">b</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv45equalRK5arrayRK5array14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Returns the bool array with (a == b) element-wise.

<!-- -->

<span id="_CPPv3eqRK5arrayRK5array"></span><span id="_CPPv2eqRK5arrayRK5array"></span><span id="eq-operator__arrayCR.arrayCR"></span><span id="group__ops_1gaa30cf69f3d22f65615f5e1696dd5703f" class="target"></span><span class="k"><span class="pre">inline</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="k"><span class="pre">operator</span></span><span class="o"><span class="pre">==</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">b</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4eqRK5arrayRK5array"
class="headerlink" title="Link to this definition">#</a>    

<!-- -->

<span id="_CPPv3I0Eeq1TRK5array"></span><span id="_CPPv2I0Eeq1TRK5array"></span><span class="k"><span class="pre">template</span></span><span class="p"><span class="pre">\<</span></span><span class="k"><span class="pre">typename</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">T</span></span></span><span class="p"><span class="pre">\></span></span>  
<span id="group__ops_1gaf115782d009ac2a547fcca395c9ec797" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="k"><span class="pre">operator</span></span><span class="o"><span class="pre">==</span></span></span><span class="sig-paren">(</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4I0Eeq5array1TRK5array"
class="reference internal" title="operator==::T"><span class="n"><span
class="pre">T</span></span></a><span class="w"> </span><span class="n sig-param"><span class="pre">a</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">b</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4I0Eeq5array1TRK5array"
class="headerlink" title="Link to this definition">#</a>    

<!-- -->

<span id="_CPPv3I0EeqRK5array1T"></span><span id="_CPPv2I0EeqRK5array1T"></span><span class="k"><span class="pre">template</span></span><span class="p"><span class="pre">\<</span></span><span class="k"><span class="pre">typename</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">T</span></span></span><span class="p"><span class="pre">\></span></span>  
<span id="group__ops_1ga3ad3ed7aece2650943a35082dbe3a0a5" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="k"><span class="pre">operator</span></span><span class="o"><span class="pre">==</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4I0Eeq5arrayRK5array1T"
class="reference internal" title="operator==::T"><span class="n"><span
class="pre">T</span></span></a><span class="w"> </span><span class="n sig-param"><span class="pre">b</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4I0Eeq5arrayRK5array1T"
class="headerlink" title="Link to this definition">#</a>    

<!-- -->

<span id="_CPPv39not_equalRK5arrayRK5array14StreamOrDevice"></span><span id="_CPPv29not_equalRK5arrayRK5array14StreamOrDevice"></span><span id="not_equal__arrayCR.arrayCR.StreamOrDevice"></span><span id="group__ops_1ga28f22c5d2c399eee53be7b3facc11103" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">not_equal</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">b</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv49not_equalRK5arrayRK5array14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Returns the bool array with (a != b) element-wise.

<!-- -->

<span id="_CPPv3neRK5arrayRK5array"></span><span id="_CPPv2neRK5arrayRK5array"></span><span id="neq-operator__arrayCR.arrayCR"></span><span id="group__ops_1ga0ac483d85f23252ca8757e9926d5a3c5" class="target"></span><span class="k"><span class="pre">inline</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="k"><span class="pre">operator</span></span><span class="o"><span class="pre">!=</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">b</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4neRK5arrayRK5array"
class="headerlink" title="Link to this definition">#</a>    

<!-- -->

<span id="_CPPv3I0Ene1TRK5array"></span><span id="_CPPv2I0Ene1TRK5array"></span><span class="k"><span class="pre">template</span></span><span class="p"><span class="pre">\<</span></span><span class="k"><span class="pre">typename</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">T</span></span></span><span class="p"><span class="pre">\></span></span>  
<span id="group__ops_1ga3fecba9f3cb9a19afd8ca492cf509ce0" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="k"><span class="pre">operator</span></span><span class="o"><span class="pre">!=</span></span></span><span class="sig-paren">(</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4I0Ene5array1TRK5array"
class="reference internal" title="operator!=::T"><span class="n"><span
class="pre">T</span></span></a><span class="w"> </span><span class="n sig-param"><span class="pre">a</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">b</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4I0Ene5array1TRK5array"
class="headerlink" title="Link to this definition">#</a>    

<!-- -->

<span id="_CPPv3I0EneRK5array1T"></span><span id="_CPPv2I0EneRK5array1T"></span><span class="k"><span class="pre">template</span></span><span class="p"><span class="pre">\<</span></span><span class="k"><span class="pre">typename</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">T</span></span></span><span class="p"><span class="pre">\></span></span>  
<span id="group__ops_1gaebbf1cfde388c7480159a03c92c9a385" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="k"><span class="pre">operator</span></span><span class="o"><span class="pre">!=</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4I0Ene5arrayRK5array1T"
class="reference internal" title="operator!=::T"><span class="n"><span
class="pre">T</span></span></a><span class="w"> </span><span class="n sig-param"><span class="pre">b</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4I0Ene5arrayRK5array1T"
class="headerlink" title="Link to this definition">#</a>    

<!-- -->

<span id="_CPPv37greaterRK5arrayRK5array14StreamOrDevice"></span><span id="_CPPv27greaterRK5arrayRK5array14StreamOrDevice"></span><span id="greater__arrayCR.arrayCR.StreamOrDevice"></span><span id="group__ops_1gaf4ec7bfc1ad13b891f1f3ef1772ef04d" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">greater</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">b</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv47greaterRK5arrayRK5array14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Returns bool array with (a \> b) element-wise.

<!-- -->

<span id="_CPPv3gtRK5arrayRK5array"></span><span id="_CPPv2gtRK5arrayRK5array"></span><span id="gt-operator__arrayCR.arrayCR"></span><span id="group__ops_1ga74fd2777adef10e6fe628a9cdadb01cb" class="target"></span><span class="k"><span class="pre">inline</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="k"><span class="pre">operator</span></span><span class="o"><span class="pre">\></span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">b</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4gtRK5arrayRK5array"
class="headerlink" title="Link to this definition">#</a>    

<!-- -->

<span id="_CPPv3I0Egt1TRK5array"></span><span id="_CPPv2I0Egt1TRK5array"></span><span class="k"><span class="pre">template</span></span><span class="p"><span class="pre">\<</span></span><span class="k"><span class="pre">typename</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">T</span></span></span><span class="p"><span class="pre">\></span></span>  
<span id="group__ops_1ga32e106e794e2c32e4e7decee2df2477f" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="k"><span class="pre">operator</span></span><span class="o"><span class="pre">\></span></span></span><span class="sig-paren">(</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4I0Egt5array1TRK5array"
class="reference internal" title="operator&gt;::T"><span class="n"><span
class="pre">T</span></span></a><span class="w"> </span><span class="n sig-param"><span class="pre">a</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">b</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4I0Egt5array1TRK5array"
class="headerlink" title="Link to this definition">#</a>    

<!-- -->

<span id="_CPPv3I0EgtRK5array1T"></span><span id="_CPPv2I0EgtRK5array1T"></span><span class="k"><span class="pre">template</span></span><span class="p"><span class="pre">\<</span></span><span class="k"><span class="pre">typename</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">T</span></span></span><span class="p"><span class="pre">\></span></span>  
<span id="group__ops_1ga96552b90e89923c5d2064cc427775ec5" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="k"><span class="pre">operator</span></span><span class="o"><span class="pre">\></span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4I0Egt5arrayRK5array1T"
class="reference internal" title="operator&gt;::T"><span class="n"><span
class="pre">T</span></span></a><span class="w"> </span><span class="n sig-param"><span class="pre">b</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4I0Egt5arrayRK5array1T"
class="headerlink" title="Link to this definition">#</a>    

<!-- -->

<span id="_CPPv313greater_equalRK5arrayRK5array14StreamOrDevice"></span><span id="_CPPv213greater_equalRK5arrayRK5array14StreamOrDevice"></span><span id="greater_equal__arrayCR.arrayCR.StreamOrDevice"></span><span id="group__ops_1ga7153071bcfff6faad21332163fb9a430" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">greater_equal</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">b</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv413greater_equalRK5arrayRK5array14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Returns bool array with (a \>= b) element-wise.

<!-- -->

<span id="_CPPv3geRK5arrayRK5array"></span><span id="_CPPv2geRK5arrayRK5array"></span><span id="gte-operator__arrayCR.arrayCR"></span><span id="group__ops_1ga3a41895f25ed083a36994d95fa102546" class="target"></span><span class="k"><span class="pre">inline</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="k"><span class="pre">operator</span></span><span class="o"><span class="pre">\>=</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">b</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4geRK5arrayRK5array"
class="headerlink" title="Link to this definition">#</a>    

<!-- -->

<span id="_CPPv3I0Ege1TRK5array"></span><span id="_CPPv2I0Ege1TRK5array"></span><span class="k"><span class="pre">template</span></span><span class="p"><span class="pre">\<</span></span><span class="k"><span class="pre">typename</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">T</span></span></span><span class="p"><span class="pre">\></span></span>  
<span id="group__ops_1gaf509f2cb3b18963232f20d6c3bd229b2" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="k"><span class="pre">operator</span></span><span class="o"><span class="pre">\>=</span></span></span><span class="sig-paren">(</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4I0Ege5array1TRK5array"
class="reference internal" title="operator&gt;=::T"><span
class="n"><span class="pre">T</span></span></a><span class="w"> </span><span class="n sig-param"><span class="pre">a</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">b</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4I0Ege5array1TRK5array"
class="headerlink" title="Link to this definition">#</a>    

<!-- -->

<span id="_CPPv3I0EgeRK5array1T"></span><span id="_CPPv2I0EgeRK5array1T"></span><span class="k"><span class="pre">template</span></span><span class="p"><span class="pre">\<</span></span><span class="k"><span class="pre">typename</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">T</span></span></span><span class="p"><span class="pre">\></span></span>  
<span id="group__ops_1gafa0eb25d5978674bfc9e59d4145ec590" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="k"><span class="pre">operator</span></span><span class="o"><span class="pre">\>=</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4I0Ege5arrayRK5array1T"
class="reference internal" title="operator&gt;=::T"><span
class="n"><span class="pre">T</span></span></a><span class="w"> </span><span class="n sig-param"><span class="pre">b</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4I0Ege5arrayRK5array1T"
class="headerlink" title="Link to this definition">#</a>    

<!-- -->

<span id="_CPPv34lessRK5arrayRK5array14StreamOrDevice"></span><span id="_CPPv24lessRK5arrayRK5array14StreamOrDevice"></span><span id="less__arrayCR.arrayCR.StreamOrDevice"></span><span id="group__ops_1ga9142b8d717699a8abfa2a7398891ff8a" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">less</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">b</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv44lessRK5arrayRK5array14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Returns bool array with (a \< b) element-wise.

<!-- -->

<span id="_CPPv3ltRK5arrayRK5array"></span><span id="_CPPv2ltRK5arrayRK5array"></span><span id="lt-operator__arrayCR.arrayCR"></span><span id="group__ops_1gaee41e2b8f61d563200ff03575ac1d6c3" class="target"></span><span class="k"><span class="pre">inline</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="k"><span class="pre">operator</span></span><span class="o"><span class="pre">\<</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">b</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4ltRK5arrayRK5array"
class="headerlink" title="Link to this definition">#</a>    

<!-- -->

<span id="_CPPv3I0Elt1TRK5array"></span><span id="_CPPv2I0Elt1TRK5array"></span><span class="k"><span class="pre">template</span></span><span class="p"><span class="pre">\<</span></span><span class="k"><span class="pre">typename</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">T</span></span></span><span class="p"><span class="pre">\></span></span>  
<span id="group__ops_1ga1ef8ea11cf15ce628c54201fa42748ef" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="k"><span class="pre">operator</span></span><span class="o"><span class="pre">\<</span></span></span><span class="sig-paren">(</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4I0Elt5array1TRK5array"
class="reference internal" title="operator&lt;::T"><span class="n"><span
class="pre">T</span></span></a><span class="w"> </span><span class="n sig-param"><span class="pre">a</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">b</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4I0Elt5array1TRK5array"
class="headerlink" title="Link to this definition">#</a>    

<!-- -->

<span id="_CPPv3I0EltRK5array1T"></span><span id="_CPPv2I0EltRK5array1T"></span><span class="k"><span class="pre">template</span></span><span class="p"><span class="pre">\<</span></span><span class="k"><span class="pre">typename</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">T</span></span></span><span class="p"><span class="pre">\></span></span>  
<span id="group__ops_1ga95e72226dc7a79c40b3d16f990922050" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="k"><span class="pre">operator</span></span><span class="o"><span class="pre">\<</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4I0Elt5arrayRK5array1T"
class="reference internal" title="operator&lt;::T"><span class="n"><span
class="pre">T</span></span></a><span class="w"> </span><span class="n sig-param"><span class="pre">b</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4I0Elt5arrayRK5array1T"
class="headerlink" title="Link to this definition">#</a>    

<!-- -->

<span id="_CPPv310less_equalRK5arrayRK5array14StreamOrDevice"></span><span id="_CPPv210less_equalRK5arrayRK5array14StreamOrDevice"></span><span id="less_equal__arrayCR.arrayCR.StreamOrDevice"></span><span id="group__ops_1ga0d49e0c7011d0573c369c13c8f045a09" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">less_equal</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">b</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv410less_equalRK5arrayRK5array14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Returns bool array with (a \<= b) element-wise.

<!-- -->

<span id="_CPPv3leRK5arrayRK5array"></span><span id="_CPPv2leRK5arrayRK5array"></span><span id="lte-operator__arrayCR.arrayCR"></span><span id="group__ops_1ga4c8b8a1632944acaae50f0de6c23ece6" class="target"></span><span class="k"><span class="pre">inline</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="k"><span class="pre">operator</span></span><span class="o"><span class="pre">\<=</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">b</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4leRK5arrayRK5array"
class="headerlink" title="Link to this definition">#</a>    

<!-- -->

<span id="_CPPv3I0Ele1TRK5array"></span><span id="_CPPv2I0Ele1TRK5array"></span><span class="k"><span class="pre">template</span></span><span class="p"><span class="pre">\<</span></span><span class="k"><span class="pre">typename</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">T</span></span></span><span class="p"><span class="pre">\></span></span>  
<span id="group__ops_1ga150a9be467c9f91482a6d6fc13504bc4" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="k"><span class="pre">operator</span></span><span class="o"><span class="pre">\<=</span></span></span><span class="sig-paren">(</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4I0Ele5array1TRK5array"
class="reference internal" title="operator&lt;=::T"><span
class="n"><span class="pre">T</span></span></a><span class="w"> </span><span class="n sig-param"><span class="pre">a</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">b</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4I0Ele5array1TRK5array"
class="headerlink" title="Link to this definition">#</a>    

<!-- -->

<span id="_CPPv3I0EleRK5array1T"></span><span id="_CPPv2I0EleRK5array1T"></span><span class="k"><span class="pre">template</span></span><span class="p"><span class="pre">\<</span></span><span class="k"><span class="pre">typename</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">T</span></span></span><span class="p"><span class="pre">\></span></span>  
<span id="group__ops_1ga624eeccef0cc4b130e1325abfea057cb" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="k"><span class="pre">operator</span></span><span class="o"><span class="pre">\<=</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4I0Ele5arrayRK5array1T"
class="reference internal" title="operator&lt;=::T"><span
class="n"><span class="pre">T</span></span></a><span class="w"> </span><span class="n sig-param"><span class="pre">b</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4I0Ele5arrayRK5array1T"
class="headerlink" title="Link to this definition">#</a>    

<!-- -->

<span id="_CPPv311array_equalRK5arrayRK5arrayb14StreamOrDevice"></span><span id="_CPPv211array_equalRK5arrayRK5arrayb14StreamOrDevice"></span><span id="array_equal__arrayCR.arrayCR.b.StreamOrDevice"></span><span id="group__ops_1ga8f3059336ee0c87207b1f8c6ab312645" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">array_equal</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">b</span></span>, <span class="kt"><span class="pre">bool</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">equal_nan</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv411array_equalRK5arrayRK5arrayb14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
True if two arrays have the same shape and elements.

<!-- -->

<span id="_CPPv311array_equalRK5arrayRK5array14StreamOrDevice"></span><span id="_CPPv211array_equalRK5arrayRK5array14StreamOrDevice"></span><span id="array_equal__arrayCR.arrayCR.StreamOrDevice"></span><span id="group__ops_1gaf79cf0271ca0105d7b14295a90d0ed14" class="target"></span><span class="k"><span class="pre">inline</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">array_equal</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">b</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv411array_equalRK5arrayRK5array14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    

<!-- -->

<span id="_CPPv35isnanRK5array14StreamOrDevice"></span><span id="_CPPv25isnanRK5array14StreamOrDevice"></span><span id="isnan__arrayCR.StreamOrDevice"></span><span id="group__ops_1ga175592792471b0ffb45196dca4711ba6" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">isnan</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv45isnanRK5array14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    

<!-- -->

<span id="_CPPv35isinfRK5array14StreamOrDevice"></span><span id="_CPPv25isinfRK5array14StreamOrDevice"></span><span id="isinf__arrayCR.StreamOrDevice"></span><span id="group__ops_1ga8fc238d5e5d1153e69da8b36015d9844" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">isinf</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv45isinfRK5array14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    

<!-- -->

<span id="_CPPv38isfiniteRK5array14StreamOrDevice"></span><span id="_CPPv28isfiniteRK5array14StreamOrDevice"></span><span id="isfinite__arrayCR.StreamOrDevice"></span><span id="group__ops_1ga725ff0789f934b1fdd54ee29e47022ff" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">isfinite</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv48isfiniteRK5array14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    

<!-- -->

<span id="_CPPv38isposinfRK5array14StreamOrDevice"></span><span id="_CPPv28isposinfRK5array14StreamOrDevice"></span><span id="isposinf__arrayCR.StreamOrDevice"></span><span id="group__ops_1gad80f7c4a58c12b6cb30a8b9a73008993" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">isposinf</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv48isposinfRK5array14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    

<!-- -->

<span id="_CPPv38isneginfRK5array14StreamOrDevice"></span><span id="_CPPv28isneginfRK5array14StreamOrDevice"></span><span id="isneginf__arrayCR.StreamOrDevice"></span><span id="group__ops_1ga1940523da381ed7be50656a3bc465ff3" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">isneginf</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv48isneginfRK5array14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    

<!-- -->

<span id="_CPPv35whereRK5arrayRK5arrayRK5array14StreamOrDevice"></span><span id="_CPPv25whereRK5arrayRK5arrayRK5array14StreamOrDevice"></span><span id="where__arrayCR.arrayCR.arrayCR.StreamOrDevice"></span><span id="group__ops_1ga8a2056f8c9bb30914c40bcf509386491" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">where</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">condition</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">x</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">y</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv45whereRK5arrayRK5arrayRK5array14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Select from x or y depending on condition.

<!-- -->

<span id="_CPPv310nan_to_numRK5arrayfKNSt8optionalIfEEKNSt8optionalIfEE14StreamOrDevice"></span><span id="_CPPv210nan_to_numRK5arrayfKNSt8optionalIfEEKNSt8optionalIfEE14StreamOrDevice"></span><span id="nan_to_num__arrayCR.float.std::optional:float:C.std::optional:float:C.StreamOrDevice"></span><span id="group__ops_1gab1467c6a9e675152e768afd6dcfb61de" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">nan_to_num</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="kt"><span class="pre">float</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">nan</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="m"><span class="pre">0.0f</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4StRK5arraybi14StreamOrDevice"
class="reference internal" title="std"><span class="n"><span
class="pre">std</span></span></a><span class="p"><span class="pre">::</span></span><span class="n"><span class="pre">optional</span></span><span class="p"><span class="pre">\<</span></span><span class="kt"><span class="pre">float</span></span><span class="p"><span class="pre">\></span></span><span class="w"> </span><span class="n sig-param"><span class="pre">posinf</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4StRK5arraybi14StreamOrDevice"
class="reference internal" title="std"><span class="n"><span
class="pre">std</span></span></a><span class="p"><span class="pre">::</span></span><span class="n"><span class="pre">nullopt</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4StRK5arraybi14StreamOrDevice"
class="reference internal" title="std"><span class="n"><span
class="pre">std</span></span></a><span class="p"><span class="pre">::</span></span><span class="n"><span class="pre">optional</span></span><span class="p"><span class="pre">\<</span></span><span class="kt"><span class="pre">float</span></span><span class="p"><span class="pre">\></span></span><span class="w"> </span><span class="n sig-param"><span class="pre">neginf</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4StRK5arraybi14StreamOrDevice"
class="reference internal" title="std"><span class="n"><span
class="pre">std</span></span></a><span class="p"><span class="pre">::</span></span><span class="n"><span class="pre">nullopt</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv410nan_to_numRK5arrayfKNSt8optionalIfEEKNSt8optionalIfEE14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Replace NaN and infinities with finite numbers.

<!-- -->

<span id="_CPPv33allRK5arrayb14StreamOrDevice"></span><span id="_CPPv23allRK5arrayb14StreamOrDevice"></span><span id="all__arrayCR.b.StreamOrDevice"></span><span id="group__ops_1ga3b1b90ef1275ca17655b6d7f25d3ee68" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">all</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="kt"><span class="pre">bool</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">keepdims</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv43allRK5arrayb14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
True if all elements in the array are true (or non-zero).

<!-- -->

<span id="_CPPv33allRK5array14StreamOrDevice"></span><span id="_CPPv23allRK5array14StreamOrDevice"></span><span id="all__arrayCR.StreamOrDevice"></span><span id="group__ops_1ga3689e12e8f42dadb4cbe2b07dc4099f4" class="target"></span><span class="k"><span class="pre">inline</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">all</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv43allRK5array14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    

<!-- -->

<span id="_CPPv38allcloseRK5arrayRK5arrayddb14StreamOrDevice"></span><span id="_CPPv28allcloseRK5arrayRK5arrayddb14StreamOrDevice"></span><span id="allclose__arrayCR.arrayCR.double.double.b.StreamOrDevice"></span><span id="group__ops_1gaf0cd4257de7542daf9faf5e605e31020" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">allclose</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">b</span></span>, <span class="kt"><span class="pre">double</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">rtol</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="m"><span class="pre">1e-5</span></span>, <span class="kt"><span class="pre">double</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">atol</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="m"><span class="pre">1e-8</span></span>, <span class="kt"><span class="pre">bool</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">equal_nan</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="k"><span class="pre">false</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv48allcloseRK5arrayRK5arrayddb14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
True if the two arrays are equal within the specified tolerance.

<!-- -->

<span id="_CPPv37iscloseRK5arrayRK5arrayddb14StreamOrDevice"></span><span id="_CPPv27iscloseRK5arrayRK5arrayddb14StreamOrDevice"></span><span id="isclose__arrayCR.arrayCR.double.double.b.StreamOrDevice"></span><span id="group__ops_1ga51eac95c04400921c54716de14b52491" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">isclose</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">b</span></span>, <span class="kt"><span class="pre">double</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">rtol</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="m"><span class="pre">1e-5</span></span>, <span class="kt"><span class="pre">double</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">atol</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="m"><span class="pre">1e-8</span></span>, <span class="kt"><span class="pre">bool</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">equal_nan</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="k"><span class="pre">false</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv47iscloseRK5arrayRK5arrayddb14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Returns a boolean array where two arrays are element-wise equal within
the specified tolerance.

<!-- -->

<span id="_CPPv33allRK5arrayRKNSt6vectorIiEEb14StreamOrDevice"></span><span id="_CPPv23allRK5arrayRKNSt6vectorIiEEb14StreamOrDevice"></span><span id="all__arrayCR.std::vector:i:CR.b.StreamOrDevice"></span><span id="group__ops_1gac0919c6ba53aea35a7683dea7e9a9a59" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">all</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4StRK5arraybi14StreamOrDevice"
class="reference internal" title="std"><span class="n"><span
class="pre">std</span></span></a><span class="p"><span class="pre">::</span></span><span class="n"><span class="pre">vector</span></span><span class="p"><span class="pre">\<</span></span><span class="kt"><span class="pre">int</span></span><span class="p"><span class="pre">\></span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">axes</span></span>, <span class="kt"><span class="pre">bool</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">keepdims</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="k"><span class="pre">false</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv43allRK5arrayRKNSt6vectorIiEEb14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Reduces the input along the given axes.

An output value is true if all the corresponding inputs are true.

<!-- -->

<span id="_CPPv33allRK5arrayib14StreamOrDevice"></span><span id="_CPPv23allRK5arrayib14StreamOrDevice"></span><span id="all__arrayCR.i.b.StreamOrDevice"></span><span id="group__ops_1gae2d5fcc5b62d673cca76c08b7b4afbbc" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">all</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="kt"><span class="pre">int</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">axis</span></span>, <span class="kt"><span class="pre">bool</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">keepdims</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="k"><span class="pre">false</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv43allRK5arrayib14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Reduces the input along the given axis.

An output value is true if all the corresponding inputs are true.

<!-- -->

<span id="_CPPv33anyRK5arrayb14StreamOrDevice"></span><span id="_CPPv23anyRK5arrayb14StreamOrDevice"></span><span id="any__arrayCR.b.StreamOrDevice"></span><span id="group__ops_1ga8598dd718fb05cb28535e250372d4e6f" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">any</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="kt"><span class="pre">bool</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">keepdims</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv43anyRK5arrayb14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
True if any elements in the array are true (or non-zero).

<!-- -->

<span id="_CPPv33anyRK5array14StreamOrDevice"></span><span id="_CPPv23anyRK5array14StreamOrDevice"></span><span id="any__arrayCR.StreamOrDevice"></span><span id="group__ops_1gad37df97f253a963bece124198dbaf9ba" class="target"></span><span class="k"><span class="pre">inline</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">any</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv43anyRK5array14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    

<!-- -->

<span id="_CPPv33anyRK5arrayRKNSt6vectorIiEEb14StreamOrDevice"></span><span id="_CPPv23anyRK5arrayRKNSt6vectorIiEEb14StreamOrDevice"></span><span id="any__arrayCR.std::vector:i:CR.b.StreamOrDevice"></span><span id="group__ops_1gaf240618fc8b06debf5f56e97e84f18ef" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">any</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4StRK5arraybi14StreamOrDevice"
class="reference internal" title="std"><span class="n"><span
class="pre">std</span></span></a><span class="p"><span class="pre">::</span></span><span class="n"><span class="pre">vector</span></span><span class="p"><span class="pre">\<</span></span><span class="kt"><span class="pre">int</span></span><span class="p"><span class="pre">\></span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">axes</span></span>, <span class="kt"><span class="pre">bool</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">keepdims</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="k"><span class="pre">false</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv43anyRK5arrayRKNSt6vectorIiEEb14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Reduces the input along the given axes.

An output value is true if any of the corresponding inputs are true.

<!-- -->

<span id="_CPPv33anyRK5arrayib14StreamOrDevice"></span><span id="_CPPv23anyRK5arrayib14StreamOrDevice"></span><span id="any__arrayCR.i.b.StreamOrDevice"></span><span id="group__ops_1gab1d56277d468a55227f4dad6bc2fc1ce" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">any</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="kt"><span class="pre">int</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">axis</span></span>, <span class="kt"><span class="pre">bool</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">keepdims</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="k"><span class="pre">false</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv43anyRK5arrayib14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Reduces the input along the given axis.

An output value is true if any of the corresponding inputs are true.

<!-- -->

<span id="_CPPv33sumRK5arrayb14StreamOrDevice"></span><span id="_CPPv23sumRK5arrayb14StreamOrDevice"></span><span id="sum__arrayCR.b.StreamOrDevice"></span><span id="group__ops_1gade905ee92eb6ab7edfc312aeddfbaeb6" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">sum</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="kt"><span class="pre">bool</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">keepdims</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv43sumRK5arrayb14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Sums the elements of an array.

<!-- -->

<span id="_CPPv33sumRK5array14StreamOrDevice"></span><span id="_CPPv23sumRK5array14StreamOrDevice"></span><span id="sum__arrayCR.StreamOrDevice"></span><span id="group__ops_1ga3627754d7868487bdab1bd83f05d9c81" class="target"></span><span class="k"><span class="pre">inline</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">sum</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv43sumRK5array14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    

<!-- -->

<span id="_CPPv33sumRK5arrayRKNSt6vectorIiEEb14StreamOrDevice"></span><span id="_CPPv23sumRK5arrayRKNSt6vectorIiEEb14StreamOrDevice"></span><span id="sum__arrayCR.std::vector:i:CR.b.StreamOrDevice"></span><span id="group__ops_1gaccd0a6be2c5b5128fdc2d87b5c8e67f4" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">sum</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4StRK5arraybi14StreamOrDevice"
class="reference internal" title="std"><span class="n"><span
class="pre">std</span></span></a><span class="p"><span class="pre">::</span></span><span class="n"><span class="pre">vector</span></span><span class="p"><span class="pre">\<</span></span><span class="kt"><span class="pre">int</span></span><span class="p"><span class="pre">\></span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">axes</span></span>, <span class="kt"><span class="pre">bool</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">keepdims</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="k"><span class="pre">false</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv43sumRK5arrayRKNSt6vectorIiEEb14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Sums the elements of an array along the given axes.

<!-- -->

<span id="_CPPv33sumRK5arrayib14StreamOrDevice"></span><span id="_CPPv23sumRK5arrayib14StreamOrDevice"></span><span id="sum__arrayCR.i.b.StreamOrDevice"></span><span id="group__ops_1gafcd39b0bf39a56c26a967981c7ab8a8d" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">sum</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="kt"><span class="pre">int</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">axis</span></span>, <span class="kt"><span class="pre">bool</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">keepdims</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="k"><span class="pre">false</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv43sumRK5arrayib14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Sums the elements of an array along the given axis.

<!-- -->

<span id="_CPPv34meanRK5arrayb14StreamOrDevice"></span><span id="_CPPv24meanRK5arrayb14StreamOrDevice"></span><span id="mean__arrayCR.b.StreamOrDevice"></span><span id="group__ops_1gade46e768fd46b8b640eb16f26abeecef" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">mean</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="kt"><span class="pre">bool</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">keepdims</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv44meanRK5arrayb14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Computes the mean of the elements of an array.

<!-- -->

<span id="_CPPv34meanRK5array14StreamOrDevice"></span><span id="_CPPv24meanRK5array14StreamOrDevice"></span><span id="mean__arrayCR.StreamOrDevice"></span><span id="group__ops_1ga52b59fdd8e8430538e564f5bbcfa31e6" class="target"></span><span class="k"><span class="pre">inline</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">mean</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv44meanRK5array14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    

<!-- -->

<span id="_CPPv34meanRK5arrayRKNSt6vectorIiEEb14StreamOrDevice"></span><span id="_CPPv24meanRK5arrayRKNSt6vectorIiEEb14StreamOrDevice"></span><span id="mean__arrayCR.std::vector:i:CR.b.StreamOrDevice"></span><span id="group__ops_1ga066161f3d3e395a1d76c638cb680d444" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">mean</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4StRK5arraybi14StreamOrDevice"
class="reference internal" title="std"><span class="n"><span
class="pre">std</span></span></a><span class="p"><span class="pre">::</span></span><span class="n"><span class="pre">vector</span></span><span class="p"><span class="pre">\<</span></span><span class="kt"><span class="pre">int</span></span><span class="p"><span class="pre">\></span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">axes</span></span>, <span class="kt"><span class="pre">bool</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">keepdims</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="k"><span class="pre">false</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv44meanRK5arrayRKNSt6vectorIiEEb14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Computes the mean of the elements of an array along the given axes.

<!-- -->

<span id="_CPPv34meanRK5arrayib14StreamOrDevice"></span><span id="_CPPv24meanRK5arrayib14StreamOrDevice"></span><span id="mean__arrayCR.i.b.StreamOrDevice"></span><span id="group__ops_1ga45fba73eab0e3b6e128ed3ce2f43a5da" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">mean</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="kt"><span class="pre">int</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">axis</span></span>, <span class="kt"><span class="pre">bool</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">keepdims</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="k"><span class="pre">false</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv44meanRK5arrayib14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Computes the mean of the elements of an array along the given axis.

<!-- -->

<span id="_CPPv33varRK5arraybi14StreamOrDevice"></span><span id="_CPPv23varRK5arraybi14StreamOrDevice"></span><span id="var__arrayCR.b.i.StreamOrDevice"></span><span id="group__ops_1ga7e133df686439588a8cd1fb10ce0c6e9" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">var</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="kt"><span class="pre">bool</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">keepdims</span></span>, <span class="kt"><span class="pre">int</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">ddof</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="m"><span class="pre">0</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv43varRK5arraybi14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Computes the variance of the elements of an array.

<!-- -->

<span id="_CPPv33varRK5array14StreamOrDevice"></span><span id="_CPPv23varRK5array14StreamOrDevice"></span><span id="var__arrayCR.StreamOrDevice"></span><span id="group__ops_1ga7d7b38d118fa2613214078ef0f7d5a42" class="target"></span><span class="k"><span class="pre">inline</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">var</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv43varRK5array14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    

<!-- -->

<span id="_CPPv33varRK5arrayRKNSt6vectorIiEEbi14StreamOrDevice"></span><span id="_CPPv23varRK5arrayRKNSt6vectorIiEEbi14StreamOrDevice"></span><span id="var__arrayCR.std::vector:i:CR.b.i.StreamOrDevice"></span><span id="group__ops_1ga78ddeb966cbe7a5b0aa17e1de43025f2" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">var</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4StRK5arraybi14StreamOrDevice"
class="reference internal" title="std"><span class="n"><span
class="pre">std</span></span></a><span class="p"><span class="pre">::</span></span><span class="n"><span class="pre">vector</span></span><span class="p"><span class="pre">\<</span></span><span class="kt"><span class="pre">int</span></span><span class="p"><span class="pre">\></span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">axes</span></span>, <span class="kt"><span class="pre">bool</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">keepdims</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="k"><span class="pre">false</span></span>, <span class="kt"><span class="pre">int</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">ddof</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="m"><span class="pre">0</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv43varRK5arrayRKNSt6vectorIiEEbi14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Computes the variance of the elements of an array along the given axes.

<!-- -->

<span id="_CPPv33varRK5arrayibi14StreamOrDevice"></span><span id="_CPPv23varRK5arrayibi14StreamOrDevice"></span><span id="var__arrayCR.i.b.i.StreamOrDevice"></span><span id="group__ops_1ga4fbf3e3f98f2e4956faf87af320aa9d0" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">var</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="kt"><span class="pre">int</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">axis</span></span>, <span class="kt"><span class="pre">bool</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">keepdims</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="k"><span class="pre">false</span></span>, <span class="kt"><span class="pre">int</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">ddof</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="m"><span class="pre">0</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv43varRK5arrayibi14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Computes the variance of the elements of an array along the given axis.

<!-- -->

<span id="_CPPv3StRK5arraybi14StreamOrDevice"></span><span id="_CPPv2StRK5arraybi14StreamOrDevice"></span><span id="std__arrayCR.b.i.StreamOrDevice"></span><span id="group__ops_1ga2a466024f8061febc0a64be557644cb0" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">std</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="kt"><span class="pre">bool</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">keepdims</span></span>, <span class="kt"><span class="pre">int</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">ddof</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="m"><span class="pre">0</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4StRK5arraybi14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Computes the standard deviation of the elements of an array.

<!-- -->

<span id="_CPPv3StRK5array14StreamOrDevice"></span><span id="_CPPv2StRK5array14StreamOrDevice"></span><span id="std__arrayCR.StreamOrDevice"></span><span id="group__ops_1gafdcb04d77c64405a3990078a77dd984c" class="target"></span><span class="k"><span class="pre">inline</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">std</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4StRK5array14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    

<!-- -->

<span id="_CPPv3StRK5arrayRKNSt6vectorIiEEbi14StreamOrDevice"></span><span id="_CPPv2StRK5arrayRKNSt6vectorIiEEbi14StreamOrDevice"></span><span id="std__arrayCR.std::vector:i:CR.b.i.StreamOrDevice"></span><span id="group__ops_1ga7f649970bf38b987b6ef847054f3c2f8" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">std</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4StRK5arrayRKNSt6vectorIiEEbi14StreamOrDevice"
class="reference internal" title="std"><span class="n"><span
class="pre">std</span></span></a><span class="p"><span class="pre">::</span></span><span class="n"><span class="pre">vector</span></span><span class="p"><span class="pre">\<</span></span><span class="kt"><span class="pre">int</span></span><span class="p"><span class="pre">\></span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">axes</span></span>, <span class="kt"><span class="pre">bool</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">keepdims</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="k"><span class="pre">false</span></span>, <span class="kt"><span class="pre">int</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">ddof</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="m"><span class="pre">0</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4StRK5arrayRKNSt6vectorIiEEbi14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Computes the standard deviatoin of the elements of an array along the
given axes.

<!-- -->

<span id="_CPPv3StRK5arrayibi14StreamOrDevice"></span><span id="_CPPv2StRK5arrayibi14StreamOrDevice"></span><span id="std__arrayCR.i.b.i.StreamOrDevice"></span><span id="group__ops_1ga62721a206df671ef5797449eea97af9f" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">std</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="kt"><span class="pre">int</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">axis</span></span>, <span class="kt"><span class="pre">bool</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">keepdims</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="k"><span class="pre">false</span></span>, <span class="kt"><span class="pre">int</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">ddof</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="m"><span class="pre">0</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4StRK5arrayibi14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Computes the standard deviation of the elements of an array along the
given axis.

<!-- -->

<span id="_CPPv34prodRK5arrayb14StreamOrDevice"></span><span id="_CPPv24prodRK5arrayb14StreamOrDevice"></span><span id="prod__arrayCR.b.StreamOrDevice"></span><span id="group__ops_1ga4a09b7241d564d92548bc2773eb1d544" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">prod</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="kt"><span class="pre">bool</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">keepdims</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv44prodRK5arrayb14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
The product of all elements of the array.

<!-- -->

<span id="_CPPv34prodRK5array14StreamOrDevice"></span><span id="_CPPv24prodRK5array14StreamOrDevice"></span><span id="prod__arrayCR.StreamOrDevice"></span><span id="group__ops_1ga61832191f3c42ea549cf04953edc3602" class="target"></span><span class="k"><span class="pre">inline</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">prod</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv44prodRK5array14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    

<!-- -->

<span id="_CPPv34prodRK5arrayRKNSt6vectorIiEEb14StreamOrDevice"></span><span id="_CPPv24prodRK5arrayRKNSt6vectorIiEEb14StreamOrDevice"></span><span id="prod__arrayCR.std::vector:i:CR.b.StreamOrDevice"></span><span id="group__ops_1ga2b3935108f641e20a70dbf63f540d970" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">prod</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4StRK5arraybi14StreamOrDevice"
class="reference internal" title="std"><span class="n"><span
class="pre">std</span></span></a><span class="p"><span class="pre">::</span></span><span class="n"><span class="pre">vector</span></span><span class="p"><span class="pre">\<</span></span><span class="kt"><span class="pre">int</span></span><span class="p"><span class="pre">\></span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">axes</span></span>, <span class="kt"><span class="pre">bool</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">keepdims</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="k"><span class="pre">false</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv44prodRK5arrayRKNSt6vectorIiEEb14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
The product of the elements of an array along the given axes.

<!-- -->

<span id="_CPPv34prodRK5arrayib14StreamOrDevice"></span><span id="_CPPv24prodRK5arrayib14StreamOrDevice"></span><span id="prod__arrayCR.i.b.StreamOrDevice"></span><span id="group__ops_1ga8a10a10b81c69996d0aca8ba401f8ff0" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">prod</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="kt"><span class="pre">int</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">axis</span></span>, <span class="kt"><span class="pre">bool</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">keepdims</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="k"><span class="pre">false</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv44prodRK5arrayib14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
The product of the elements of an array along the given axis.

<!-- -->

<span id="_CPPv33maxRK5arrayb14StreamOrDevice"></span><span id="_CPPv23maxRK5arrayb14StreamOrDevice"></span><span id="max__arrayCR.b.StreamOrDevice"></span><span id="group__ops_1ga7fed87d96cc7741d8267f4eac83f5fe7" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">max</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="kt"><span class="pre">bool</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">keepdims</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv43maxRK5arrayb14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
The maximum of all elements of the array.

<!-- -->

<span id="_CPPv33maxRK5array14StreamOrDevice"></span><span id="_CPPv23maxRK5array14StreamOrDevice"></span><span id="max__arrayCR.StreamOrDevice"></span><span id="group__ops_1ga25be91d70a5f40341db0615a0b8bfedc" class="target"></span><span class="k"><span class="pre">inline</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">max</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv43maxRK5array14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    

<!-- -->

<span id="_CPPv33maxRK5arrayRKNSt6vectorIiEEb14StreamOrDevice"></span><span id="_CPPv23maxRK5arrayRKNSt6vectorIiEEb14StreamOrDevice"></span><span id="max__arrayCR.std::vector:i:CR.b.StreamOrDevice"></span><span id="group__ops_1ga1ca7b6b91fe2459a7d83897bf013827f" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">max</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4StRK5arraybi14StreamOrDevice"
class="reference internal" title="std"><span class="n"><span
class="pre">std</span></span></a><span class="p"><span class="pre">::</span></span><span class="n"><span class="pre">vector</span></span><span class="p"><span class="pre">\<</span></span><span class="kt"><span class="pre">int</span></span><span class="p"><span class="pre">\></span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">axes</span></span>, <span class="kt"><span class="pre">bool</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">keepdims</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="k"><span class="pre">false</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv43maxRK5arrayRKNSt6vectorIiEEb14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
The maximum of the elements of an array along the given axes.

<!-- -->

<span id="_CPPv33maxRK5arrayib14StreamOrDevice"></span><span id="_CPPv23maxRK5arrayib14StreamOrDevice"></span><span id="max__arrayCR.i.b.StreamOrDevice"></span><span id="group__ops_1ga7b638050e03a93f2896c981bc2850a47" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">max</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="kt"><span class="pre">int</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">axis</span></span>, <span class="kt"><span class="pre">bool</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">keepdims</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="k"><span class="pre">false</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv43maxRK5arrayib14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
The maximum of the elements of an array along the given axis.

<!-- -->

<span id="_CPPv33minRK5arrayb14StreamOrDevice"></span><span id="_CPPv23minRK5arrayb14StreamOrDevice"></span><span id="min__arrayCR.b.StreamOrDevice"></span><span id="group__ops_1gab27599802617a4c8f9964ab5f4ffee12" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">min</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="kt"><span class="pre">bool</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">keepdims</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv43minRK5arrayb14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
The minimum of all elements of the array.

<!-- -->

<span id="_CPPv33minRK5array14StreamOrDevice"></span><span id="_CPPv23minRK5array14StreamOrDevice"></span><span id="min__arrayCR.StreamOrDevice"></span><span id="group__ops_1ga0140b91e9cdfc3fef0da8e332f65a9e8" class="target"></span><span class="k"><span class="pre">inline</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">min</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv43minRK5array14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    

<!-- -->

<span id="_CPPv33minRK5arrayRKNSt6vectorIiEEb14StreamOrDevice"></span><span id="_CPPv23minRK5arrayRKNSt6vectorIiEEb14StreamOrDevice"></span><span id="min__arrayCR.std::vector:i:CR.b.StreamOrDevice"></span><span id="group__ops_1ga6efb83cd46436678c8f8c4af15cc00f5" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">min</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4StRK5arraybi14StreamOrDevice"
class="reference internal" title="std"><span class="n"><span
class="pre">std</span></span></a><span class="p"><span class="pre">::</span></span><span class="n"><span class="pre">vector</span></span><span class="p"><span class="pre">\<</span></span><span class="kt"><span class="pre">int</span></span><span class="p"><span class="pre">\></span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">axes</span></span>, <span class="kt"><span class="pre">bool</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">keepdims</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="k"><span class="pre">false</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv43minRK5arrayRKNSt6vectorIiEEb14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
The minimum of the elements of an array along the given axes.

<!-- -->

<span id="_CPPv33minRK5arrayib14StreamOrDevice"></span><span id="_CPPv23minRK5arrayib14StreamOrDevice"></span><span id="min__arrayCR.i.b.StreamOrDevice"></span><span id="group__ops_1ga36fa315eef677f4143868f552cd26d03" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">min</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="kt"><span class="pre">int</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">axis</span></span>, <span class="kt"><span class="pre">bool</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">keepdims</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="k"><span class="pre">false</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv43minRK5arrayib14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
The minimum of the elements of an array along the given axis.

<!-- -->

<span id="_CPPv36argminRK5arrayb14StreamOrDevice"></span><span id="_CPPv26argminRK5arrayb14StreamOrDevice"></span><span id="argmin__arrayCR.b.StreamOrDevice"></span><span id="group__ops_1ga7c3bd5ef430a71dfd298e626741e3c71" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">argmin</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="kt"><span class="pre">bool</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">keepdims</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv46argminRK5arrayb14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Returns the index of the minimum value in the array.

<!-- -->

<span id="_CPPv36argminRK5array14StreamOrDevice"></span><span id="_CPPv26argminRK5array14StreamOrDevice"></span><span id="argmin__arrayCR.StreamOrDevice"></span><span id="group__ops_1ga6bc577c5ab10cd9c848ba81321595070" class="target"></span><span class="k"><span class="pre">inline</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">argmin</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv46argminRK5array14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    

<!-- -->

<span id="_CPPv36argminRK5arrayib14StreamOrDevice"></span><span id="_CPPv26argminRK5arrayib14StreamOrDevice"></span><span id="argmin__arrayCR.i.b.StreamOrDevice"></span><span id="group__ops_1gaf66dc3c77b88e4009e0678eda41eca81" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">argmin</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="kt"><span class="pre">int</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">axis</span></span>, <span class="kt"><span class="pre">bool</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">keepdims</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="k"><span class="pre">false</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv46argminRK5arrayib14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Returns the indices of the minimum values along a given axis.

<!-- -->

<span id="_CPPv36argmaxRK5arrayb14StreamOrDevice"></span><span id="_CPPv26argmaxRK5arrayb14StreamOrDevice"></span><span id="argmax__arrayCR.b.StreamOrDevice"></span><span id="group__ops_1gae60b0b5339b9c50b9970260faf613e83" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">argmax</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="kt"><span class="pre">bool</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">keepdims</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv46argmaxRK5arrayb14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Returns the index of the maximum value in the array.

<!-- -->

<span id="_CPPv36argmaxRK5array14StreamOrDevice"></span><span id="_CPPv26argmaxRK5array14StreamOrDevice"></span><span id="argmax__arrayCR.StreamOrDevice"></span><span id="group__ops_1gae6f6c5a840320b336fdc9687e0ed56c8" class="target"></span><span class="k"><span class="pre">inline</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">argmax</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv46argmaxRK5array14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    

<!-- -->

<span id="_CPPv36argmaxRK5arrayib14StreamOrDevice"></span><span id="_CPPv26argmaxRK5arrayib14StreamOrDevice"></span><span id="argmax__arrayCR.i.b.StreamOrDevice"></span><span id="group__ops_1ga2efa67466510fc26ab9ea8dff30f2ba5" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">argmax</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="kt"><span class="pre">int</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">axis</span></span>, <span class="kt"><span class="pre">bool</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">keepdims</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="k"><span class="pre">false</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv46argmaxRK5arrayib14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Returns the indices of the maximum values along a given axis.

<!-- -->

<span id="_CPPv34sortRK5array14StreamOrDevice"></span><span id="_CPPv24sortRK5array14StreamOrDevice"></span><span id="sort__arrayCR.StreamOrDevice"></span><span id="group__ops_1ga7fb616054665b3c2d61fa234f501f079" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">sort</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv44sortRK5array14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Returns a sorted copy of the flattened array.

<!-- -->

<span id="_CPPv34sortRK5arrayi14StreamOrDevice"></span><span id="_CPPv24sortRK5arrayi14StreamOrDevice"></span><span id="sort__arrayCR.i.StreamOrDevice"></span><span id="group__ops_1gaae1bc47aa737f705d0e5884270063fea" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">sort</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="kt"><span class="pre">int</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">axis</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv44sortRK5arrayi14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Returns a sorted copy of the array along a given axis.

<!-- -->

<span id="_CPPv37argsortRK5array14StreamOrDevice"></span><span id="_CPPv27argsortRK5array14StreamOrDevice"></span><span id="argsort__arrayCR.StreamOrDevice"></span><span id="group__ops_1ga8df3b2703bf671457422894dd870cdc5" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">argsort</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv47argsortRK5array14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Returns indices that sort the flattened array.

<!-- -->

<span id="_CPPv37argsortRK5arrayi14StreamOrDevice"></span><span id="_CPPv27argsortRK5arrayi14StreamOrDevice"></span><span id="argsort__arrayCR.i.StreamOrDevice"></span><span id="group__ops_1ga7878e0daa5a75f44e57b5fe948fa3ef6" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">argsort</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="kt"><span class="pre">int</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">axis</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv47argsortRK5arrayi14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Returns indices that sort the array along a given axis.

<!-- -->

<span id="_CPPv39partitionRK5arrayi14StreamOrDevice"></span><span id="_CPPv29partitionRK5arrayi14StreamOrDevice"></span><span id="partition__arrayCR.i.StreamOrDevice"></span><span id="group__ops_1gac1b30830a972fb9a2601379ad2b32405" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">partition</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="kt"><span class="pre">int</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">kth</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv49partitionRK5arrayi14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Returns a partitioned copy of the flattened array such that the smaller
kth elements are first.

<!-- -->

<span id="_CPPv39partitionRK5arrayii14StreamOrDevice"></span><span id="_CPPv29partitionRK5arrayii14StreamOrDevice"></span><span id="partition__arrayCR.i.i.StreamOrDevice"></span><span id="group__ops_1ga4fbea3a5f66cf81e3c119d1661119321" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">partition</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="kt"><span class="pre">int</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">kth</span></span>, <span class="kt"><span class="pre">int</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">axis</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv49partitionRK5arrayii14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Returns a partitioned copy of the array along a given axis such that the
smaller kth elements are first.

<!-- -->

<span id="_CPPv312argpartitionRK5arrayi14StreamOrDevice"></span><span id="_CPPv212argpartitionRK5arrayi14StreamOrDevice"></span><span id="argpartition__arrayCR.i.StreamOrDevice"></span><span id="group__ops_1gaf301c49c10fa9b95a9e8dc52ead1a8dd" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">argpartition</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="kt"><span class="pre">int</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">kth</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv412argpartitionRK5arrayi14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Returns indices that partition the flattened array such that the smaller
kth elements are first.

<!-- -->

<span id="_CPPv312argpartitionRK5arrayii14StreamOrDevice"></span><span id="_CPPv212argpartitionRK5arrayii14StreamOrDevice"></span><span id="argpartition__arrayCR.i.i.StreamOrDevice"></span><span id="group__ops_1ga7b15c654c7463def57857a0e239989a3" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">argpartition</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="kt"><span class="pre">int</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">kth</span></span>, <span class="kt"><span class="pre">int</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">axis</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv412argpartitionRK5arrayii14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Returns indices that partition the array along a given axis such that
the smaller kth elements are first.

<!-- -->

<span id="_CPPv34topkRK5arrayi14StreamOrDevice"></span><span id="_CPPv24topkRK5arrayi14StreamOrDevice"></span><span id="topk__arrayCR.i.StreamOrDevice"></span><span id="group__ops_1ga5487dd887c43e5341f3e68ffe47f0f5a" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">topk</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="kt"><span class="pre">int</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">k</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv44topkRK5arrayi14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Returns topk elements of the flattened array.

<!-- -->

<span id="_CPPv34topkRK5arrayii14StreamOrDevice"></span><span id="_CPPv24topkRK5arrayii14StreamOrDevice"></span><span id="topk__arrayCR.i.i.StreamOrDevice"></span><span id="group__ops_1ga35b8436c79ff953f6c809598b646f498" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">topk</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="kt"><span class="pre">int</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">k</span></span>, <span class="kt"><span class="pre">int</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">axis</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv44topkRK5arrayii14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Returns topk elements of the array along a given axis.

<!-- -->

<span id="_CPPv312logcumsumexpRK5arrayibb14StreamOrDevice"></span><span id="_CPPv212logcumsumexpRK5arrayibb14StreamOrDevice"></span><span id="logcumsumexp__arrayCR.i.b.b.StreamOrDevice"></span><span id="group__ops_1gaeb541012d52d7082365048f9e094cae4" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">logcumsumexp</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="kt"><span class="pre">int</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">axis</span></span>, <span class="kt"><span class="pre">bool</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">reverse</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="k"><span class="pre">false</span></span>, <span class="kt"><span class="pre">bool</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">inclusive</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="k"><span class="pre">true</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv412logcumsumexpRK5arrayibb14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Cumulative logsumexp of an array.

<!-- -->

<span id="_CPPv39logsumexpRK5arrayb14StreamOrDevice"></span><span id="_CPPv29logsumexpRK5arrayb14StreamOrDevice"></span><span id="logsumexp__arrayCR.b.StreamOrDevice"></span><span id="group__ops_1gacff4eb57c085d571e722083680267ac5" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">logsumexp</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="kt"><span class="pre">bool</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">keepdims</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv49logsumexpRK5arrayb14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
The logsumexp of all elements of the array.

<!-- -->

<span id="_CPPv39logsumexpRK5array14StreamOrDevice"></span><span id="_CPPv29logsumexpRK5array14StreamOrDevice"></span><span id="logsumexp__arrayCR.StreamOrDevice"></span><span id="group__ops_1ga59be50b4e92f1dc20b53460cefa3910d" class="target"></span><span class="k"><span class="pre">inline</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">logsumexp</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv49logsumexpRK5array14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    

<!-- -->

<span id="_CPPv39logsumexpRK5arrayRKNSt6vectorIiEEb14StreamOrDevice"></span><span id="_CPPv29logsumexpRK5arrayRKNSt6vectorIiEEb14StreamOrDevice"></span><span id="logsumexp__arrayCR.std::vector:i:CR.b.StreamOrDevice"></span><span id="group__ops_1gae3969c7bd24c4f3ab97831df28239689" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">logsumexp</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4StRK5arraybi14StreamOrDevice"
class="reference internal" title="std"><span class="n"><span
class="pre">std</span></span></a><span class="p"><span class="pre">::</span></span><span class="n"><span class="pre">vector</span></span><span class="p"><span class="pre">\<</span></span><span class="kt"><span class="pre">int</span></span><span class="p"><span class="pre">\></span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">axes</span></span>, <span class="kt"><span class="pre">bool</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">keepdims</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="k"><span class="pre">false</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv49logsumexpRK5arrayRKNSt6vectorIiEEb14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
The logsumexp of the elements of an array along the given axes.

<!-- -->

<span id="_CPPv39logsumexpRK5arrayib14StreamOrDevice"></span><span id="_CPPv29logsumexpRK5arrayib14StreamOrDevice"></span><span id="logsumexp__arrayCR.i.b.StreamOrDevice"></span><span id="group__ops_1gafef5cb2159c16a60a95470cc823bdd44" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">logsumexp</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="kt"><span class="pre">int</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">axis</span></span>, <span class="kt"><span class="pre">bool</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">keepdims</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="k"><span class="pre">false</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv49logsumexpRK5arrayib14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
The logsumexp of the elements of an array along the given axis.

<!-- -->

<span id="_CPPv33absRK5array14StreamOrDevice"></span><span id="_CPPv23absRK5array14StreamOrDevice"></span><span id="abs__arrayCR.StreamOrDevice"></span><span id="group__ops_1ga5528e80f5e8bad71e106a0cf9edd8920" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">abs</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv43absRK5array14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Absolute value of elements in an array.

<!-- -->

<span id="_CPPv38negativeRK5array14StreamOrDevice"></span><span id="_CPPv28negativeRK5array14StreamOrDevice"></span><span id="negative__arrayCR.StreamOrDevice"></span><span id="group__ops_1ga95d9a9425533b5ed1707eb00184dffc6" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">negative</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv48negativeRK5array14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Negate an array.

<!-- -->

<span id="_CPPv3miRK5array"></span><span id="_CPPv2miRK5array"></span><span id="sub-operator__arrayCR"></span><span id="group__ops_1gade2eea48989f4caaf36e89f7bd2a8816" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="k"><span class="pre">operator</span></span><span class="o"><span class="pre">-</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span><span class="sig-paren">)</span><a href="https://ml-explore.github.io/mlx/build/html/#_CPPv4miRK5array"
class="headerlink" title="Link to this definition">#</a>    

<!-- -->

<span id="_CPPv34signRK5array14StreamOrDevice"></span><span id="_CPPv24signRK5array14StreamOrDevice"></span><span id="sign__arrayCR.StreamOrDevice"></span><span id="group__ops_1ga20f1a1a8c0cd6206485f9363f3915faa" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">sign</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv44signRK5array14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
The sign of the elements in an array.

<!-- -->

<span id="_CPPv311logical_notRK5array14StreamOrDevice"></span><span id="_CPPv211logical_notRK5array14StreamOrDevice"></span><span id="logical_not__arrayCR.StreamOrDevice"></span><span id="group__ops_1gabca78d34ce93f0de2814e62225bb2a53" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">logical_not</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv411logical_notRK5array14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Logical not of an array.

<!-- -->

<span id="_CPPv311logical_andRK5arrayRK5array14StreamOrDevice"></span><span id="_CPPv211logical_andRK5arrayRK5array14StreamOrDevice"></span><span id="logical_and__arrayCR.arrayCR.StreamOrDevice"></span><span id="group__ops_1ga768977cda8d68cf23f464a6af9907876" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">logical_and</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">b</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv411logical_andRK5arrayRK5array14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Logical and of two arrays.

<!-- -->

<span id="_CPPv3aaRK5arrayRK5array"></span><span id="_CPPv2aaRK5arrayRK5array"></span><span id="sand-operator__arrayCR.arrayCR"></span><span id="group__ops_1gaee1d774bb0843601d7a0a4257d616ae3" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="k"><span class="pre">operator</span></span><span class="o"><span class="pre">&&</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">b</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4aaRK5arrayRK5array"
class="headerlink" title="Link to this definition">#</a>    

<!-- -->

<span id="_CPPv310logical_orRK5arrayRK5array14StreamOrDevice"></span><span id="_CPPv210logical_orRK5arrayRK5array14StreamOrDevice"></span><span id="logical_or__arrayCR.arrayCR.StreamOrDevice"></span><span id="group__ops_1ga23768728e4dd070c917fbb0ed0d0c2ec" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">logical_or</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">b</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv410logical_orRK5arrayRK5array14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Logical or of two arrays.

<!-- -->

<span id="_CPPv3ooRK5arrayRK5array"></span><span id="_CPPv2ooRK5arrayRK5array"></span><span id="sor-operator__arrayCR.arrayCR"></span><span id="group__ops_1ga27af56a98270d4d76d139f0f9171b83a" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="k"><span class="pre">operator</span></span><span class="o"><span class="pre">\|\|</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">b</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4ooRK5arrayRK5array"
class="headerlink" title="Link to this definition">#</a>    

<!-- -->

<span id="_CPPv310reciprocalRK5array14StreamOrDevice"></span><span id="_CPPv210reciprocalRK5array14StreamOrDevice"></span><span id="reciprocal__arrayCR.StreamOrDevice"></span><span id="group__ops_1ga4d29556bb93e2f66916116cf1f062b36" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">reciprocal</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv410reciprocalRK5array14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
The reciprocal (1/x) of the elements in an array.

<!-- -->

<span id="_CPPv33addRK5arrayRK5array14StreamOrDevice"></span><span id="_CPPv23addRK5arrayRK5array14StreamOrDevice"></span><span id="add__arrayCR.arrayCR.StreamOrDevice"></span><span id="group__ops_1ga2d32d67cfd76785a72c43d89b94dc7d7" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">add</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">b</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv43addRK5arrayRK5array14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Add two arrays.

<!-- -->

<span id="_CPPv3plRK5arrayRK5array"></span><span id="_CPPv2plRK5arrayRK5array"></span><span id="add-operator__arrayCR.arrayCR"></span><span id="group__ops_1ga26e5a043eaaaf066d1400adac9c11d0c" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="k"><span class="pre">operator</span></span><span class="o"><span class="pre">+</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">b</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4plRK5arrayRK5array"
class="headerlink" title="Link to this definition">#</a>    

<!-- -->

<span id="_CPPv3I0Epl1TRK5array"></span><span id="_CPPv2I0Epl1TRK5array"></span><span class="k"><span class="pre">template</span></span><span class="p"><span class="pre">\<</span></span><span class="k"><span class="pre">typename</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">T</span></span></span><span class="p"><span class="pre">\></span></span>  
<span id="group__ops_1ga7d0ec8d01e7cefa6a6b25f11876761b5" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="k"><span class="pre">operator</span></span><span class="o"><span class="pre">+</span></span></span><span class="sig-paren">(</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4I0Epl5array1TRK5array"
class="reference internal" title="operator+::T"><span class="n"><span
class="pre">T</span></span></a><span class="w"> </span><span class="n sig-param"><span class="pre">a</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">b</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4I0Epl5array1TRK5array"
class="headerlink" title="Link to this definition">#</a>    

<!-- -->

<span id="_CPPv3I0EplRK5array1T"></span><span id="_CPPv2I0EplRK5array1T"></span><span class="k"><span class="pre">template</span></span><span class="p"><span class="pre">\<</span></span><span class="k"><span class="pre">typename</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">T</span></span></span><span class="p"><span class="pre">\></span></span>  
<span id="group__ops_1ga7cc080a4f9d4a667f2099aa0dbfefadd" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="k"><span class="pre">operator</span></span><span class="o"><span class="pre">+</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4I0Epl5arrayRK5array1T"
class="reference internal" title="operator+::T"><span class="n"><span
class="pre">T</span></span></a><span class="w"> </span><span class="n sig-param"><span class="pre">b</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4I0Epl5arrayRK5array1T"
class="headerlink" title="Link to this definition">#</a>    

<!-- -->

<span id="_CPPv38subtractRK5arrayRK5array14StreamOrDevice"></span><span id="_CPPv28subtractRK5arrayRK5array14StreamOrDevice"></span><span id="subtract__arrayCR.arrayCR.StreamOrDevice"></span><span id="group__ops_1ga196c240d3d0fcbb4713802c485e15133" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">subtract</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">b</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv48subtractRK5arrayRK5array14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Subtract two arrays.

<!-- -->

<span id="_CPPv3miRK5arrayRK5array"></span><span id="_CPPv2miRK5arrayRK5array"></span><span id="sub-operator__arrayCR.arrayCR"></span><span id="group__ops_1ga0c7f3cb36d4ca516c7a33142f88b9181" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="k"><span class="pre">operator</span></span><span class="o"><span class="pre">-</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">b</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4miRK5arrayRK5array"
class="headerlink" title="Link to this definition">#</a>    

<!-- -->

<span id="_CPPv3I0Emi1TRK5array"></span><span id="_CPPv2I0Emi1TRK5array"></span><span class="k"><span class="pre">template</span></span><span class="p"><span class="pre">\<</span></span><span class="k"><span class="pre">typename</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">T</span></span></span><span class="p"><span class="pre">\></span></span>  
<span id="group__ops_1gae68d3d0691ba951501218e98439f3465" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="k"><span class="pre">operator</span></span><span class="o"><span class="pre">-</span></span></span><span class="sig-paren">(</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4I0Emi5array1TRK5array"
class="reference internal" title="operator-::T"><span class="n"><span
class="pre">T</span></span></a><span class="w"> </span><span class="n sig-param"><span class="pre">a</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">b</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4I0Emi5array1TRK5array"
class="headerlink" title="Link to this definition">#</a>    

<!-- -->

<span id="_CPPv3I0EmiRK5array1T"></span><span id="_CPPv2I0EmiRK5array1T"></span><span class="k"><span class="pre">template</span></span><span class="p"><span class="pre">\<</span></span><span class="k"><span class="pre">typename</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">T</span></span></span><span class="p"><span class="pre">\></span></span>  
<span id="group__ops_1gaf5e5d882c51ad0a0ea315c274d5439b2" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="k"><span class="pre">operator</span></span><span class="o"><span class="pre">-</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4I0Emi5arrayRK5array1T"
class="reference internal" title="operator-::T"><span class="n"><span
class="pre">T</span></span></a><span class="w"> </span><span class="n sig-param"><span class="pre">b</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4I0Emi5arrayRK5array1T"
class="headerlink" title="Link to this definition">#</a>    

<!-- -->

<span id="_CPPv38multiplyRK5arrayRK5array14StreamOrDevice"></span><span id="_CPPv28multiplyRK5arrayRK5array14StreamOrDevice"></span><span id="multiply__arrayCR.arrayCR.StreamOrDevice"></span><span id="group__ops_1gaf57392e641640b5d06e4c99518391c38" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">multiply</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">b</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv48multiplyRK5arrayRK5array14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Multiply two arrays.

<!-- -->

<span id="_CPPv3mlRK5arrayRK5array"></span><span id="_CPPv2mlRK5arrayRK5array"></span><span id="mul-operator__arrayCR.arrayCR"></span><span id="group__ops_1ga26c33f5cdb6fc10d272acd6e208034e0" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="k"><span class="pre">operator</span></span><span class="o"><span class="pre">\*</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">b</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4mlRK5arrayRK5array"
class="headerlink" title="Link to this definition">#</a>    

<!-- -->

<span id="_CPPv3I0Eml1TRK5array"></span><span id="_CPPv2I0Eml1TRK5array"></span><span class="k"><span class="pre">template</span></span><span class="p"><span class="pre">\<</span></span><span class="k"><span class="pre">typename</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">T</span></span></span><span class="p"><span class="pre">\></span></span>  
<span id="group__ops_1gac22a67f7de797b1ae59029843cbdcab6" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="k"><span class="pre">operator</span></span><span class="o"><span class="pre">\*</span></span></span><span class="sig-paren">(</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4I0Eml5array1TRK5array"
class="reference internal" title="operator*::T"><span class="n"><span
class="pre">T</span></span></a><span class="w"> </span><span class="n sig-param"><span class="pre">a</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">b</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4I0Eml5array1TRK5array"
class="headerlink" title="Link to this definition">#</a>    

<!-- -->

<span id="_CPPv3I0EmlRK5array1T"></span><span id="_CPPv2I0EmlRK5array1T"></span><span class="k"><span class="pre">template</span></span><span class="p"><span class="pre">\<</span></span><span class="k"><span class="pre">typename</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">T</span></span></span><span class="p"><span class="pre">\></span></span>  
<span id="group__ops_1ga6f2369ed5fae8ff9b1528670a004dde2" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="k"><span class="pre">operator</span></span><span class="o"><span class="pre">\*</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4I0Eml5arrayRK5array1T"
class="reference internal" title="operator*::T"><span class="n"><span
class="pre">T</span></span></a><span class="w"> </span><span class="n sig-param"><span class="pre">b</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4I0Eml5arrayRK5array1T"
class="headerlink" title="Link to this definition">#</a>    

<!-- -->

<span id="_CPPv36divideRK5arrayRK5array14StreamOrDevice"></span><span id="_CPPv26divideRK5arrayRK5array14StreamOrDevice"></span><span id="divide__arrayCR.arrayCR.StreamOrDevice"></span><span id="group__ops_1ga77472dd06cfa7a30a42e4fd927bd859f" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">divide</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">b</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv46divideRK5arrayRK5array14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Divide two arrays.

<!-- -->

<span id="_CPPv3dvRK5arrayRK5array"></span><span id="_CPPv2dvRK5arrayRK5array"></span><span id="div-operator__arrayCR.arrayCR"></span><span id="group__ops_1gaeedf77f722b394429f1a7f6c367883bf" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="k"><span class="pre">operator</span></span><span class="o"><span class="pre">/</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">b</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4dvRK5arrayRK5array"
class="headerlink" title="Link to this definition">#</a>    

<!-- -->

<span id="_CPPv3dvdRK5array"></span><span id="_CPPv2dvdRK5array"></span><span id="div-operator__double.arrayCR"></span><span id="group__ops_1ga7366ec7f453be2a4dc449f0faa1bf554" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="k"><span class="pre">operator</span></span><span class="o"><span class="pre">/</span></span></span><span class="sig-paren">(</span><span class="kt"><span class="pre">double</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">a</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">b</span></span><span class="sig-paren">)</span><a href="https://ml-explore.github.io/mlx/build/html/#_CPPv4dvdRK5array"
class="headerlink" title="Link to this definition">#</a>    

<!-- -->

<span id="_CPPv3dvRK5arrayd"></span><span id="_CPPv2dvRK5arrayd"></span><span id="div-operator__arrayCR.double"></span><span id="group__ops_1gadfb324ae9b4feb2c7ea0ac6ade639f38" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="k"><span class="pre">operator</span></span><span class="o"><span class="pre">/</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="kt"><span class="pre">double</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">b</span></span><span class="sig-paren">)</span><a href="https://ml-explore.github.io/mlx/build/html/#_CPPv4dvRK5arrayd"
class="headerlink" title="Link to this definition">#</a>    

<!-- -->

<span id="_CPPv36divmodRK5arrayRK5array14StreamOrDevice"></span><span id="_CPPv26divmodRK5arrayRK5array14StreamOrDevice"></span><span id="divmod__arrayCR.arrayCR.StreamOrDevice"></span><span id="group__ops_1gaa30ebc0a8376dbc3f7e46a47052b5894" class="target"></span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4StRK5arraybi14StreamOrDevice"
class="reference internal" title="std"><span class="n"><span
class="pre">std</span></span></a><span class="p"><span class="pre">::</span></span><span class="n"><span class="pre">vector</span></span><span class="p"><span class="pre">\<</span></span><span class="n"><span class="pre">array</span></span><span class="p"><span class="pre">\></span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">divmod</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">b</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv46divmodRK5arrayRK5array14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Compute the element-wise quotient and remainder.

<!-- -->

<span id="_CPPv312floor_divideRK5arrayRK5array14StreamOrDevice"></span><span id="_CPPv212floor_divideRK5arrayRK5array14StreamOrDevice"></span><span id="floor_divide__arrayCR.arrayCR.StreamOrDevice"></span><span id="group__ops_1ga05b4c6054d028107869511f927da01cd" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">floor_divide</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">b</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv412floor_divideRK5arrayRK5array14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Compute integer division.

Equivalent to doing floor(a / x).

<!-- -->

<span id="_CPPv39remainderRK5arrayRK5array14StreamOrDevice"></span><span id="_CPPv29remainderRK5arrayRK5array14StreamOrDevice"></span><span id="remainder__arrayCR.arrayCR.StreamOrDevice"></span><span id="group__ops_1ga99f5c904f724156a814d7817188351d2" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">remainder</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">b</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv49remainderRK5arrayRK5array14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Compute the element-wise remainder of division.

<!-- -->

<span id="_CPPv3rmRK5arrayRK5array"></span><span id="_CPPv2rmRK5arrayRK5array"></span><span id="mod-operator__arrayCR.arrayCR"></span><span id="group__ops_1gab3bfbf82b1e4de7b00bbcf1a2255fbde" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="k"><span class="pre">operator</span></span><span class="o"><span class="pre">%</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">b</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4rmRK5arrayRK5array"
class="headerlink" title="Link to this definition">#</a>    

<!-- -->

<span id="_CPPv3I0Erm1TRK5array"></span><span id="_CPPv2I0Erm1TRK5array"></span><span class="k"><span class="pre">template</span></span><span class="p"><span class="pre">\<</span></span><span class="k"><span class="pre">typename</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">T</span></span></span><span class="p"><span class="pre">\></span></span>  
<span id="group__ops_1ga50817666f0b82afcbf4a123486af9908" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="k"><span class="pre">operator</span></span><span class="o"><span class="pre">%</span></span></span><span class="sig-paren">(</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4I0Erm5array1TRK5array"
class="reference internal" title="operator%::T"><span class="n"><span
class="pre">T</span></span></a><span class="w"> </span><span class="n sig-param"><span class="pre">a</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">b</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4I0Erm5array1TRK5array"
class="headerlink" title="Link to this definition">#</a>    

<!-- -->

<span id="_CPPv3I0ErmRK5array1T"></span><span id="_CPPv2I0ErmRK5array1T"></span><span class="k"><span class="pre">template</span></span><span class="p"><span class="pre">\<</span></span><span class="k"><span class="pre">typename</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">T</span></span></span><span class="p"><span class="pre">\></span></span>  
<span id="group__ops_1ga46c01daa07433542a477d216e13a8480" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="k"><span class="pre">operator</span></span><span class="o"><span class="pre">%</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4I0Erm5arrayRK5array1T"
class="reference internal" title="operator%::T"><span class="n"><span
class="pre">T</span></span></a><span class="w"> </span><span class="n sig-param"><span class="pre">b</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4I0Erm5arrayRK5array1T"
class="headerlink" title="Link to this definition">#</a>    

<!-- -->

<span id="_CPPv37maximumRK5arrayRK5array14StreamOrDevice"></span><span id="_CPPv27maximumRK5arrayRK5array14StreamOrDevice"></span><span id="maximum__arrayCR.arrayCR.StreamOrDevice"></span><span id="group__ops_1ga7ade2ea305e2e4219c3609443fb5db8d" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">maximum</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">b</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv47maximumRK5arrayRK5array14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Element-wise maximum between two arrays.

<!-- -->

<span id="_CPPv37minimumRK5arrayRK5array14StreamOrDevice"></span><span id="_CPPv27minimumRK5arrayRK5array14StreamOrDevice"></span><span id="minimum__arrayCR.arrayCR.StreamOrDevice"></span><span id="group__ops_1ga49ba00c090f81f331c91b0c97040bce0" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">minimum</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">b</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv47minimumRK5arrayRK5array14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Element-wise minimum between two arrays.

<!-- -->

<span id="_CPPv35floorRK5array14StreamOrDevice"></span><span id="_CPPv25floorRK5array14StreamOrDevice"></span><span id="floor__arrayCR.StreamOrDevice"></span><span id="group__ops_1ga8d656904aa2690b60955ae745aecfc30" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">floor</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv45floorRK5array14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Floor the element of an array.

<!-- -->

<span id="_CPPv34ceilRK5array14StreamOrDevice"></span><span id="_CPPv24ceilRK5array14StreamOrDevice"></span><span id="ceil__arrayCR.StreamOrDevice"></span><span id="group__ops_1ga1404ecceff83fd9b9139b7520f55e096" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">ceil</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv44ceilRK5array14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Ceil the element of an array.

<!-- -->

<span id="_CPPv36squareRK5array14StreamOrDevice"></span><span id="_CPPv26squareRK5array14StreamOrDevice"></span><span id="square__arrayCR.StreamOrDevice"></span><span id="group__ops_1ga1234e4c39cfa79f19d4bdb5b8ea4d45e" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">square</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv46squareRK5array14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Square the elements of an array.

<!-- -->

<span id="_CPPv33expRK5array14StreamOrDevice"></span><span id="_CPPv23expRK5array14StreamOrDevice"></span><span id="exp__arrayCR.StreamOrDevice"></span><span id="group__ops_1ga8a3b04e23e347d99ecf411fd6f4e5125" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">exp</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv43expRK5array14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Exponential of the elements of an array.

<!-- -->

<span id="_CPPv33sinRK5array14StreamOrDevice"></span><span id="_CPPv23sinRK5array14StreamOrDevice"></span><span id="sin__arrayCR.StreamOrDevice"></span><span id="group__ops_1gaebf0a73ad3732fba39df37826c235692" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">sin</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv43sinRK5array14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Sine of the elements of an array.

<!-- -->

<span id="_CPPv33cosRK5array14StreamOrDevice"></span><span id="_CPPv23cosRK5array14StreamOrDevice"></span><span id="cos__arrayCR.StreamOrDevice"></span><span id="group__ops_1ga39dfdf72b556012aa35ff27a94116e74" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">cos</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv43cosRK5array14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Cosine of the elements of an array.

<!-- -->

<span id="_CPPv33tanRK5array14StreamOrDevice"></span><span id="_CPPv23tanRK5array14StreamOrDevice"></span><span id="tan__arrayCR.StreamOrDevice"></span><span id="group__ops_1ga3f10e89a4bcb1a8fa44fb33b8d1176a5" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">tan</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv43tanRK5array14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Tangent of the elements of an array.

<!-- -->

<span id="_CPPv36arcsinRK5array14StreamOrDevice"></span><span id="_CPPv26arcsinRK5array14StreamOrDevice"></span><span id="arcsin__arrayCR.StreamOrDevice"></span><span id="group__ops_1ga8770e8c8f23f13343911f4c9d6e1c619" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">arcsin</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv46arcsinRK5array14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Arc Sine of the elements of an array.

<!-- -->

<span id="_CPPv36arccosRK5array14StreamOrDevice"></span><span id="_CPPv26arccosRK5array14StreamOrDevice"></span><span id="arccos__arrayCR.StreamOrDevice"></span><span id="group__ops_1ga08bec7cb10c84466487b507fc5bf9776" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">arccos</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv46arccosRK5array14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Arc Cosine of the elements of an array.

<!-- -->

<span id="_CPPv36arctanRK5array14StreamOrDevice"></span><span id="_CPPv26arctanRK5array14StreamOrDevice"></span><span id="arctan__arrayCR.StreamOrDevice"></span><span id="group__ops_1gaa041f3f070e68f4946db07516b7d092e" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">arctan</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv46arctanRK5array14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Arc Tangent of the elements of an array.

<!-- -->

<span id="_CPPv37arctan2RK5arrayRK5array14StreamOrDevice"></span><span id="_CPPv27arctan2RK5arrayRK5array14StreamOrDevice"></span><span id="arctan2__arrayCR.arrayCR.StreamOrDevice"></span><span id="group__ops_1ga6caba9c92b5989123501f909cc7da354" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">arctan2</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">b</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv47arctan2RK5arrayRK5array14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Inverse tangent of the ratio of two arrays.

<!-- -->

<span id="_CPPv34sinhRK5array14StreamOrDevice"></span><span id="_CPPv24sinhRK5array14StreamOrDevice"></span><span id="sinh__arrayCR.StreamOrDevice"></span><span id="group__ops_1gaf532375c6563dbd6e329bdedf0224dd7" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">sinh</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv44sinhRK5array14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Hyperbolic Sine of the elements of an array.

<!-- -->

<span id="_CPPv34coshRK5array14StreamOrDevice"></span><span id="_CPPv24coshRK5array14StreamOrDevice"></span><span id="cosh__arrayCR.StreamOrDevice"></span><span id="group__ops_1ga2181b71cda88007a3092be4795ff0715" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">cosh</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv44coshRK5array14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Hyperbolic Cosine of the elements of an array.

<!-- -->

<span id="_CPPv34tanhRK5array14StreamOrDevice"></span><span id="_CPPv24tanhRK5array14StreamOrDevice"></span><span id="tanh__arrayCR.StreamOrDevice"></span><span id="group__ops_1ga5efb19aa0dfa42d8a3d5e1dfd569cd6d" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">tanh</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv44tanhRK5array14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Hyperbolic Tangent of the elements of an array.

<!-- -->

<span id="_CPPv37arcsinhRK5array14StreamOrDevice"></span><span id="_CPPv27arcsinhRK5array14StreamOrDevice"></span><span id="arcsinh__arrayCR.StreamOrDevice"></span><span id="group__ops_1gac62e2cedc49ef2c90dd8584000317450" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">arcsinh</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv47arcsinhRK5array14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Inverse Hyperbolic Sine of the elements of an array.

<!-- -->

<span id="_CPPv37arccoshRK5array14StreamOrDevice"></span><span id="_CPPv27arccoshRK5array14StreamOrDevice"></span><span id="arccosh__arrayCR.StreamOrDevice"></span><span id="group__ops_1gaafafcfcebdf7248679c8543d0c0497e5" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">arccosh</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv47arccoshRK5array14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Inverse Hyperbolic Cosine of the elements of an array.

<!-- -->

<span id="_CPPv37arctanhRK5array14StreamOrDevice"></span><span id="_CPPv27arctanhRK5array14StreamOrDevice"></span><span id="arctanh__arrayCR.StreamOrDevice"></span><span id="group__ops_1gab46a35925a04c5a9d2ec7898ee55358e" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">arctanh</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv47arctanhRK5array14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Inverse Hyperbolic Tangent of the elements of an array.

<!-- -->

<span id="_CPPv37degreesRK5array14StreamOrDevice"></span><span id="_CPPv27degreesRK5array14StreamOrDevice"></span><span id="degrees__arrayCR.StreamOrDevice"></span><span id="group__ops_1ga3a70569b50e1083c5ded199d73fb960c" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">degrees</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv47degreesRK5array14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Convert the elements of an array from Radians to Degrees.

<!-- -->

<span id="_CPPv37radiansRK5array14StreamOrDevice"></span><span id="_CPPv27radiansRK5array14StreamOrDevice"></span><span id="radians__arrayCR.StreamOrDevice"></span><span id="group__ops_1ga07aa8059adba5b9a8818027b8aafd31e" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">radians</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv47radiansRK5array14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Convert the elements of an array from Degrees to Radians.

<!-- -->

<span id="_CPPv33logRK5array14StreamOrDevice"></span><span id="_CPPv23logRK5array14StreamOrDevice"></span><span id="log__arrayCR.StreamOrDevice"></span><span id="group__ops_1ga6fb22d4926133573e430fcc92f4eef31" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">log</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv43logRK5array14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Natural logarithm of the elements of an array.

<!-- -->

<span id="_CPPv34log2RK5array14StreamOrDevice"></span><span id="_CPPv24log2RK5array14StreamOrDevice"></span><span id="log2__arrayCR.StreamOrDevice"></span><span id="group__ops_1ga144228d7222d15af3a135b8b0f3fa21b" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">log2</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv44log2RK5array14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Log base 2 of the elements of an array.

<!-- -->

<span id="_CPPv35log10RK5array14StreamOrDevice"></span><span id="_CPPv25log10RK5array14StreamOrDevice"></span><span id="log10__arrayCR.StreamOrDevice"></span><span id="group__ops_1ga1fdcc7fc8819caf2e6f1c327ed4e9b9e" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">log10</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv45log10RK5array14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Log base 10 of the elements of an array.

<!-- -->

<span id="_CPPv35log1pRK5array14StreamOrDevice"></span><span id="_CPPv25log1pRK5array14StreamOrDevice"></span><span id="log1p__arrayCR.StreamOrDevice"></span><span id="group__ops_1ga20a1f4270c35b0fa544f5105a87a1604" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">log1p</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv45log1pRK5array14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Natural logarithm of one plus elements in the array:
<span class="pre">`log(1`</span>` `<span class="pre">`+`</span>` `<span class="pre">`a)`</span>.

<!-- -->

<span id="_CPPv39logaddexpRK5arrayRK5array14StreamOrDevice"></span><span id="_CPPv29logaddexpRK5arrayRK5array14StreamOrDevice"></span><span id="logaddexp__arrayCR.arrayCR.StreamOrDevice"></span><span id="group__ops_1gaf985df6609c6bd75a14a844655d89eaa" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">logaddexp</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">b</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv49logaddexpRK5arrayRK5array14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Log-add-exp of one elements in the array:
<span class="pre">`log(exp(a)`</span>` `<span class="pre">`+`</span>` `<span class="pre">`exp(b))`</span>.

<!-- -->

<span id="_CPPv37sigmoidRK5array14StreamOrDevice"></span><span id="_CPPv27sigmoidRK5array14StreamOrDevice"></span><span id="sigmoid__arrayCR.StreamOrDevice"></span><span id="group__ops_1ga708abf8f79609cd6831db7c38cafac0e" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">sigmoid</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv47sigmoidRK5array14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Element-wise logistic sigmoid of the array:
<span class="pre">`1`</span>` `<span class="pre">`/`</span>` `<span class="pre">`(1`</span>` `<span class="pre">`+`</span>` `<span class="pre">`exp(-x)`</span>.

<!-- -->

<span id="_CPPv33erfRK5array14StreamOrDevice"></span><span id="_CPPv23erfRK5array14StreamOrDevice"></span><span id="erf__arrayCR.StreamOrDevice"></span><span id="group__ops_1ga292a335240fd5d6d625fb7a340ff5eb0" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">erf</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv43erfRK5array14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Computes the error function of the elements of an array.

<!-- -->

<span id="_CPPv36erfinvRK5array14StreamOrDevice"></span><span id="_CPPv26erfinvRK5array14StreamOrDevice"></span><span id="erfinv__arrayCR.StreamOrDevice"></span><span id="group__ops_1ga76fb9062c64264e34d2e07013390557c" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">erfinv</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv46erfinvRK5array14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Computes the inverse error function of the elements of an array.

<!-- -->

<span id="_CPPv35expm1RK5array14StreamOrDevice"></span><span id="_CPPv25expm1RK5array14StreamOrDevice"></span><span id="expm1__arrayCR.StreamOrDevice"></span><span id="group__ops_1ga54ca54f06bfb2be15b163a5209e2a0f0" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">expm1</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv45expm1RK5array14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Computes the expm1 function of the elements of an array.

<!-- -->

<span id="_CPPv313stop_gradientRK5array14StreamOrDevice"></span><span id="_CPPv213stop_gradientRK5array14StreamOrDevice"></span><span id="stop_gradient__arrayCR.StreamOrDevice"></span><span id="group__ops_1ga36bc28f1deb2fe668ca9ae1e447b6b1f" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">stop_gradient</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv413stop_gradientRK5array14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Stop the flow of gradients.

<!-- -->

<span id="_CPPv35roundRK5arrayi14StreamOrDevice"></span><span id="_CPPv25roundRK5arrayi14StreamOrDevice"></span><span id="round__arrayCR.i.StreamOrDevice"></span><span id="group__ops_1ga2d74d43f007a069384e89d8416525331" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">round</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="kt"><span class="pre">int</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">decimals</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv45roundRK5arrayi14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Round a floating point number.

<!-- -->

<span id="_CPPv35roundRK5array14StreamOrDevice"></span><span id="_CPPv25roundRK5array14StreamOrDevice"></span><span id="round__arrayCR.StreamOrDevice"></span><span id="group__ops_1gaf18fb7e98bf8cf3b7fbc5e64c988a95b" class="target"></span><span class="k"><span class="pre">inline</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">round</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv45roundRK5array14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    

<!-- -->

<span id="_CPPv36matmulRK5arrayRK5array14StreamOrDevice"></span><span id="_CPPv26matmulRK5arrayRK5array14StreamOrDevice"></span><span id="matmul__arrayCR.arrayCR.StreamOrDevice"></span><span id="group__ops_1ga753d59f5a9f5f2362865ee83b4dced2a" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">matmul</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">b</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv46matmulRK5arrayRK5array14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Matrix-matrix multiplication.

<!-- -->

<span id="_CPPv36gatherRK5arrayRKNSt6vectorI5arrayEERKNSt6vectorIiEERK5Shape14StreamOrDevice"></span><span id="_CPPv26gatherRK5arrayRKNSt6vectorI5arrayEERKNSt6vectorIiEERK5Shape14StreamOrDevice"></span><span id="gather__arrayCR.std::vector:array:CR.std::vector:i:CR.ShapeCR.StreamOrDevice"></span><span id="group__ops_1ga8fcc3ad0677c834c36b72d5b2ebba6d0" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">gather</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4StRK5arraybi14StreamOrDevice"
class="reference internal" title="std"><span class="n"><span
class="pre">std</span></span></a><span class="p"><span class="pre">::</span></span><span class="n"><span class="pre">vector</span></span><span class="p"><span class="pre">\<</span></span><span class="n"><span class="pre">array</span></span><span class="p"><span class="pre">\></span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">indices</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4StRK5arraybi14StreamOrDevice"
class="reference internal" title="std"><span class="n"><span
class="pre">std</span></span></a><span class="p"><span class="pre">::</span></span><span class="n"><span class="pre">vector</span></span><span class="p"><span class="pre">\<</span></span><span class="kt"><span class="pre">int</span></span><span class="p"><span class="pre">\></span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">axes</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">Shape</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">slice_sizes</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv46gatherRK5arrayRKNSt6vectorI5arrayEERKNSt6vectorIiEERK5Shape14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Gather array entries given indices and slices.

<!-- -->

<span id="_CPPv36gatherRK5arrayRK5arrayiRK5Shape14StreamOrDevice"></span><span id="_CPPv26gatherRK5arrayRK5arrayiRK5Shape14StreamOrDevice"></span><span id="gather__arrayCR.arrayCR.i.ShapeCR.StreamOrDevice"></span><span id="group__ops_1gafe2bd174c9953ed7f12664f7abaca0e6" class="target"></span><span class="k"><span class="pre">inline</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">gather</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">indices</span></span>, <span class="kt"><span class="pre">int</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">axis</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">Shape</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">slice_sizes</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv46gatherRK5arrayRK5arrayiRK5Shape14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    

<!-- -->

<span id="_CPPv34kronRK5arrayRK5array14StreamOrDevice"></span><span id="_CPPv24kronRK5arrayRK5array14StreamOrDevice"></span><span id="kron__arrayCR.arrayCR.StreamOrDevice"></span><span id="group__ops_1ga6df16248cb68bc73644cdb1698967c19" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">kron</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">b</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv44kronRK5arrayRK5array14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Compute the Kronecker product of two arrays.

<!-- -->

<span id="_CPPv34takeRK5arrayRK5arrayi14StreamOrDevice"></span><span id="_CPPv24takeRK5arrayRK5arrayi14StreamOrDevice"></span><span id="take__arrayCR.arrayCR.i.StreamOrDevice"></span><span id="group__ops_1gac2fc270882fcfa81eb8bd068cc0d86d7" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">take</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">indices</span></span>, <span class="kt"><span class="pre">int</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">axis</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv44takeRK5arrayRK5arrayi14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Take array slices at the given indices of the specified axis.

<!-- -->

<span id="_CPPv34takeRK5arrayii14StreamOrDevice"></span><span id="_CPPv24takeRK5arrayii14StreamOrDevice"></span><span id="take__arrayCR.i.i.StreamOrDevice"></span><span id="group__ops_1ga731af77b7de547a73336f91c829c40b4" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">take</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="kt"><span class="pre">int</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">index</span></span>, <span class="kt"><span class="pre">int</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">axis</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv44takeRK5arrayii14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    

<!-- -->

<span id="_CPPv34takeRK5arrayRK5array14StreamOrDevice"></span><span id="_CPPv24takeRK5arrayRK5array14StreamOrDevice"></span><span id="take__arrayCR.arrayCR.StreamOrDevice"></span><span id="group__ops_1ga45d0f423a5e030440ef753f36c5aabf1" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">take</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">indices</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv44takeRK5arrayRK5array14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Take array entries at the given indices treating the array as flattened.

<!-- -->

<span id="_CPPv34takeRK5arrayi14StreamOrDevice"></span><span id="_CPPv24takeRK5arrayi14StreamOrDevice"></span><span id="take__arrayCR.i.StreamOrDevice"></span><span id="group__ops_1ga7b3c9b4c8ee02dc23cfd7dcd855dbf20" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">take</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="kt"><span class="pre">int</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">index</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv44takeRK5arrayi14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    

<!-- -->

<span id="_CPPv315take_along_axisRK5arrayRK5arrayi14StreamOrDevice"></span><span id="_CPPv215take_along_axisRK5arrayRK5arrayi14StreamOrDevice"></span><span id="take_along_axis__arrayCR.arrayCR.i.StreamOrDevice"></span><span id="group__ops_1gae0a81d4f983e296a87302e36d65bfc76" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">take_along_axis</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">indices</span></span>, <span class="kt"><span class="pre">int</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">axis</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv415take_along_axisRK5arrayRK5arrayi14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Take array entries given indices along the axis.

<!-- -->

<span id="_CPPv314put_along_axisRK5arrayRK5arrayRK5arrayi14StreamOrDevice"></span><span id="_CPPv214put_along_axisRK5arrayRK5arrayRK5arrayi14StreamOrDevice"></span><span id="put_along_axis__arrayCR.arrayCR.arrayCR.i.StreamOrDevice"></span><span id="group__ops_1ga8e0caebf43cd65bd40e4ce97922cd06b" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">put_along_axis</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">indices</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">values</span></span>, <span class="kt"><span class="pre">int</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">axis</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv414put_along_axisRK5arrayRK5arrayRK5arrayi14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Put the values into the array at the given indices along the axis.

<!-- -->

<span id="_CPPv316scatter_add_axisRK5arrayRK5arrayRK5arrayi14StreamOrDevice"></span><span id="_CPPv216scatter_add_axisRK5arrayRK5arrayRK5arrayi14StreamOrDevice"></span><span id="scatter_add_axis__arrayCR.arrayCR.arrayCR.i.StreamOrDevice"></span><span id="group__ops_1gab3fd98c0d06b84b836f93bddbd7a2a0d" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">scatter_add_axis</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">indices</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">values</span></span>, <span class="kt"><span class="pre">int</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">axis</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv416scatter_add_axisRK5arrayRK5arrayRK5arrayi14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Add the values into the array at the given indices along the axis.

<!-- -->

<span id="_CPPv37scatterRK5arrayRKNSt6vectorI5arrayEERK5arrayRKNSt6vectorIiEE14StreamOrDevice"></span><span id="_CPPv27scatterRK5arrayRKNSt6vectorI5arrayEERK5arrayRKNSt6vectorIiEE14StreamOrDevice"></span><span id="scatter__arrayCR.std::vector:array:CR.arrayCR.std::vector:i:CR.StreamOrDevice"></span><span id="group__ops_1gad438be8f90bae9d37c6853b8f4225d61" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">scatter</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4StRK5arraybi14StreamOrDevice"
class="reference internal" title="std"><span class="n"><span
class="pre">std</span></span></a><span class="p"><span class="pre">::</span></span><span class="n"><span class="pre">vector</span></span><span class="p"><span class="pre">\<</span></span><span class="n"><span class="pre">array</span></span><span class="p"><span class="pre">\></span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">indices</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">updates</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4StRK5arraybi14StreamOrDevice"
class="reference internal" title="std"><span class="n"><span
class="pre">std</span></span></a><span class="p"><span class="pre">::</span></span><span class="n"><span class="pre">vector</span></span><span class="p"><span class="pre">\<</span></span><span class="kt"><span class="pre">int</span></span><span class="p"><span class="pre">\></span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">axes</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv47scatterRK5arrayRKNSt6vectorI5arrayEERK5arrayRKNSt6vectorIiEE14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Scatter updates to the given indices.

The parameters <span class="pre">`indices`</span> and
<span class="pre">`axes`</span> determine the locations of
<span class="pre">`a`</span> that are updated with the values in
<span class="pre">`updates`</span>. Assuming 1-d
<span class="pre">`indices`</span> for simplicity,
<span class="pre">`indices[i]`</span> are the indices on axis
<span class="pre">`axes[i]`</span> to which the values in
<span class="pre">`updates`</span> will be applied. Note each array in
<span class="pre">`indices`</span> is assigned to a corresponding axis
and hence
<span class="pre">`indices.size()`</span>` `<span class="pre">`==`</span>` `<span class="pre">`axes.size()`</span>.
If an index/axis pair is not provided then indices along that axis are
assumed to be zero.

Note the rank of <span class="pre">`updates`</span> must be equal to the
sum of the rank of the broadcasted <span class="pre">`indices`</span>
and the rank of <span class="pre">`a`</span>. In other words, assuming
the arrays in <span class="pre">`indices`</span> have the same shape,
<span class="pre">`updates.ndim()`</span>` `<span class="pre">`==`</span>` `<span class="pre">`indices[0].ndim()`</span>` `<span class="pre">`+`</span>` `<span class="pre">`a.ndim()`</span>.
The leading dimensions of <span class="pre">`updates`</span> correspond
to the indices, and the remaining <span class="pre">`a.ndim()`</span>
dimensions are the values that will be applied to the given location in
<span class="pre">`a`</span>.

For example:

<div class="highlight-python notranslate">

<div class="highlight">

    auto in = zeros({4, 4}, float32);
    auto indices = array({2});
    auto updates = reshape(arange(1, 3, float32), {1, 1, 2});
    std::vector<int> axes{0};

    auto out = scatter(in, {indices}, updates, axes);

</div>

</div>

will produce:

<div class="highlight-python notranslate">

<div class="highlight">

    array([[0, 0, 0, 0],
           [0, 0, 0, 0],
           [1, 2, 0, 0],
           [0, 0, 0, 0]], dtype=float32)

</div>

</div>

This scatters the two-element row vector
<span class="pre">`[1,`</span>` `<span class="pre">`2]`</span> starting
at the <span class="pre">`(2,`</span>` `<span class="pre">`0)`</span>
position of <span class="pre">`a`</span>.

Adding another element to <span class="pre">`indices`</span> will
scatter into another location of <span class="pre">`a`</span>. We also
have to add an another update for the new index:

<div class="highlight-python notranslate">

<div class="highlight">

    auto in = zeros({4, 4}, float32);
    auto indices = array({2, 0});
    auto updates = reshape(arange(1, 5, float32), {2, 1, 2});
    std::vector<int> axes{0};

    auto out = scatter(in, {indices}, updates, axes):

</div>

</div>

will produce:

<div class="highlight-python notranslate">

<div class="highlight">

    array([[3, 4, 0, 0],
           [0, 0, 0, 0],
           [1, 2, 0, 0],
           [0, 0, 0, 0]], dtype=float32)

</div>

</div>

To control the scatter location on an additional axis, add another index
array to <span class="pre">`indices`</span> and another axis to
<span class="pre">`axes`</span>:

<div class="highlight-python notranslate">

<div class="highlight">

    auto in = zeros({4, 4}, float32);
    auto indices = std::vector{array({2, 0}), array({1, 2})};
    auto updates = reshape(arange(1, 5, float32), {2, 1, 2});
    std::vector<int> axes{0, 1};

    auto out = scatter(in, indices, updates, axes);

</div>

</div>

will produce:

<div class="highlight-python notranslate">

<div class="highlight">

    array([[0, 0, 3, 4],
          [0, 0, 0, 0],
          [0, 1, 2, 0],
          [0, 0, 0, 0]], dtype=float32)

</div>

</div>

Items in indices are broadcasted together. This means:

<div class="highlight-python notranslate">

<div class="highlight">

    auto indices = std::vector{array({2, 0}), array({1})};

</div>

</div>

is equivalent to:

<div class="highlight-python notranslate">

<div class="highlight">

    auto indices = std::vector{array({2, 0}), array({1, 1})};

</div>

</div>

Note, <span class="pre">`scatter`</span> does not perform bounds
checking on the indices and updates. Out-of-bounds accesses on
<span class="pre">`a`</span> are undefined and typically result in
unintended or invalid memory writes.

<!-- -->

<span id="_CPPv37scatterRK5arrayRK5arrayRK5arrayi14StreamOrDevice"></span><span id="_CPPv27scatterRK5arrayRK5arrayRK5arrayi14StreamOrDevice"></span><span id="scatter__arrayCR.arrayCR.arrayCR.i.StreamOrDevice"></span><span id="group__ops_1gac2c2b379a3ce959dbe1c4a68f112edfe" class="target"></span><span class="k"><span class="pre">inline</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">scatter</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">indices</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">updates</span></span>, <span class="kt"><span class="pre">int</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">axis</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv47scatterRK5arrayRK5arrayRK5arrayi14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    

<!-- -->

<span id="_CPPv311scatter_addRK5arrayRKNSt6vectorI5arrayEERK5arrayRKNSt6vectorIiEE14StreamOrDevice"></span><span id="_CPPv211scatter_addRK5arrayRKNSt6vectorI5arrayEERK5arrayRKNSt6vectorIiEE14StreamOrDevice"></span><span id="scatter_add__arrayCR.std::vector:array:CR.arrayCR.std::vector:i:CR.StreamOrDevice"></span><span id="group__ops_1gacd14c2b5cfebf343fc2d672722f8d174" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">scatter_add</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4StRK5arraybi14StreamOrDevice"
class="reference internal" title="std"><span class="n"><span
class="pre">std</span></span></a><span class="p"><span class="pre">::</span></span><span class="n"><span class="pre">vector</span></span><span class="p"><span class="pre">\<</span></span><span class="n"><span class="pre">array</span></span><span class="p"><span class="pre">\></span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">indices</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">updates</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4StRK5arraybi14StreamOrDevice"
class="reference internal" title="std"><span class="n"><span
class="pre">std</span></span></a><span class="p"><span class="pre">::</span></span><span class="n"><span class="pre">vector</span></span><span class="p"><span class="pre">\<</span></span><span class="kt"><span class="pre">int</span></span><span class="p"><span class="pre">\></span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">axes</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv411scatter_addRK5arrayRKNSt6vectorI5arrayEERK5arrayRKNSt6vectorIiEE14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Scatter and add updates to given indices.

<!-- -->

<span id="_CPPv311scatter_addRK5arrayRK5arrayRK5arrayi14StreamOrDevice"></span><span id="_CPPv211scatter_addRK5arrayRK5arrayRK5arrayi14StreamOrDevice"></span><span id="scatter_add__arrayCR.arrayCR.arrayCR.i.StreamOrDevice"></span><span id="group__ops_1gac13318518e5703f1273c5366eb523a5a" class="target"></span><span class="k"><span class="pre">inline</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">scatter_add</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">indices</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">updates</span></span>, <span class="kt"><span class="pre">int</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">axis</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv411scatter_addRK5arrayRK5arrayRK5arrayi14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    

<!-- -->

<span id="_CPPv312scatter_prodRK5arrayRKNSt6vectorI5arrayEERK5arrayRKNSt6vectorIiEE14StreamOrDevice"></span><span id="_CPPv212scatter_prodRK5arrayRKNSt6vectorI5arrayEERK5arrayRKNSt6vectorIiEE14StreamOrDevice"></span><span id="scatter_prod__arrayCR.std::vector:array:CR.arrayCR.std::vector:i:CR.StreamOrDevice"></span><span id="group__ops_1ga3708b5bcb61e2c63d213c4ce6ad0ffc0" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">scatter_prod</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4StRK5arraybi14StreamOrDevice"
class="reference internal" title="std"><span class="n"><span
class="pre">std</span></span></a><span class="p"><span class="pre">::</span></span><span class="n"><span class="pre">vector</span></span><span class="p"><span class="pre">\<</span></span><span class="n"><span class="pre">array</span></span><span class="p"><span class="pre">\></span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">indices</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">updates</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4StRK5arraybi14StreamOrDevice"
class="reference internal" title="std"><span class="n"><span
class="pre">std</span></span></a><span class="p"><span class="pre">::</span></span><span class="n"><span class="pre">vector</span></span><span class="p"><span class="pre">\<</span></span><span class="kt"><span class="pre">int</span></span><span class="p"><span class="pre">\></span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">axes</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv412scatter_prodRK5arrayRKNSt6vectorI5arrayEERK5arrayRKNSt6vectorIiEE14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Scatter and prod updates to given indices.

<!-- -->

<span id="_CPPv312scatter_prodRK5arrayRK5arrayRK5arrayi14StreamOrDevice"></span><span id="_CPPv212scatter_prodRK5arrayRK5arrayRK5arrayi14StreamOrDevice"></span><span id="scatter_prod__arrayCR.arrayCR.arrayCR.i.StreamOrDevice"></span><span id="group__ops_1gaf83c53c453faa9083ba27e4b97539339" class="target"></span><span class="k"><span class="pre">inline</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">scatter_prod</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">indices</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">updates</span></span>, <span class="kt"><span class="pre">int</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">axis</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv412scatter_prodRK5arrayRK5arrayRK5arrayi14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    

<!-- -->

<span id="_CPPv311scatter_maxRK5arrayRKNSt6vectorI5arrayEERK5arrayRKNSt6vectorIiEE14StreamOrDevice"></span><span id="_CPPv211scatter_maxRK5arrayRKNSt6vectorI5arrayEERK5arrayRKNSt6vectorIiEE14StreamOrDevice"></span><span id="scatter_max__arrayCR.std::vector:array:CR.arrayCR.std::vector:i:CR.StreamOrDevice"></span><span id="group__ops_1ga05881a4157cd113c9392d168a79e6673" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">scatter_max</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4StRK5arraybi14StreamOrDevice"
class="reference internal" title="std"><span class="n"><span
class="pre">std</span></span></a><span class="p"><span class="pre">::</span></span><span class="n"><span class="pre">vector</span></span><span class="p"><span class="pre">\<</span></span><span class="n"><span class="pre">array</span></span><span class="p"><span class="pre">\></span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">indices</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">updates</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4StRK5arraybi14StreamOrDevice"
class="reference internal" title="std"><span class="n"><span
class="pre">std</span></span></a><span class="p"><span class="pre">::</span></span><span class="n"><span class="pre">vector</span></span><span class="p"><span class="pre">\<</span></span><span class="kt"><span class="pre">int</span></span><span class="p"><span class="pre">\></span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">axes</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv411scatter_maxRK5arrayRKNSt6vectorI5arrayEERK5arrayRKNSt6vectorIiEE14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Scatter and max updates to given linear indices.

<!-- -->

<span id="_CPPv311scatter_maxRK5arrayRK5arrayRK5arrayi14StreamOrDevice"></span><span id="_CPPv211scatter_maxRK5arrayRK5arrayRK5arrayi14StreamOrDevice"></span><span id="scatter_max__arrayCR.arrayCR.arrayCR.i.StreamOrDevice"></span><span id="group__ops_1ga9adda5f9202bb3486e4d9e1114e3a56f" class="target"></span><span class="k"><span class="pre">inline</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">scatter_max</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">indices</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">updates</span></span>, <span class="kt"><span class="pre">int</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">axis</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv411scatter_maxRK5arrayRK5arrayRK5arrayi14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    

<!-- -->

<span id="_CPPv311scatter_minRK5arrayRKNSt6vectorI5arrayEERK5arrayRKNSt6vectorIiEE14StreamOrDevice"></span><span id="_CPPv211scatter_minRK5arrayRKNSt6vectorI5arrayEERK5arrayRKNSt6vectorIiEE14StreamOrDevice"></span><span id="scatter_min__arrayCR.std::vector:array:CR.arrayCR.std::vector:i:CR.StreamOrDevice"></span><span id="group__ops_1ga0ca16b7579dfc899f3f7fd40245ba7c5" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">scatter_min</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4StRK5arraybi14StreamOrDevice"
class="reference internal" title="std"><span class="n"><span
class="pre">std</span></span></a><span class="p"><span class="pre">::</span></span><span class="n"><span class="pre">vector</span></span><span class="p"><span class="pre">\<</span></span><span class="n"><span class="pre">array</span></span><span class="p"><span class="pre">\></span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">indices</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">updates</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4StRK5arraybi14StreamOrDevice"
class="reference internal" title="std"><span class="n"><span
class="pre">std</span></span></a><span class="p"><span class="pre">::</span></span><span class="n"><span class="pre">vector</span></span><span class="p"><span class="pre">\<</span></span><span class="kt"><span class="pre">int</span></span><span class="p"><span class="pre">\></span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">axes</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv411scatter_minRK5arrayRKNSt6vectorI5arrayEERK5arrayRKNSt6vectorIiEE14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Scatter and min updates to given linear indices.

<!-- -->

<span id="_CPPv311scatter_minRK5arrayRK5arrayRK5arrayi14StreamOrDevice"></span><span id="_CPPv211scatter_minRK5arrayRK5arrayRK5arrayi14StreamOrDevice"></span><span id="scatter_min__arrayCR.arrayCR.arrayCR.i.StreamOrDevice"></span><span id="group__ops_1ga51fa762a997c243ca7a19e1ed3e83199" class="target"></span><span class="k"><span class="pre">inline</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">scatter_min</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">indices</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">updates</span></span>, <span class="kt"><span class="pre">int</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">axis</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv411scatter_minRK5arrayRK5arrayRK5arrayi14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    

<!-- -->

<span id="_CPPv34sqrtRK5array14StreamOrDevice"></span><span id="_CPPv24sqrtRK5array14StreamOrDevice"></span><span id="sqrt__arrayCR.StreamOrDevice"></span><span id="group__ops_1ga297f853b3d90ec8ae81263977ba2ddb1" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">sqrt</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv44sqrtRK5array14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Square root the elements of an array.

<!-- -->

<span id="_CPPv35rsqrtRK5array14StreamOrDevice"></span><span id="_CPPv25rsqrtRK5array14StreamOrDevice"></span><span id="rsqrt__arrayCR.StreamOrDevice"></span><span id="group__ops_1ga102f23aa0b0c3d3296a321c694617aa1" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">rsqrt</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv45rsqrtRK5array14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Square root and reciprocal the elements of an array.

<!-- -->

<span id="_CPPv37softmaxRK5arrayRKNSt6vectorIiEEb14StreamOrDevice"></span><span id="_CPPv27softmaxRK5arrayRKNSt6vectorIiEEb14StreamOrDevice"></span><span id="softmax__arrayCR.std::vector:i:CR.b.StreamOrDevice"></span><span id="group__ops_1ga7e9bb08b43c8fd0444b7d3c9e09dc1c6" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">softmax</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4StRK5arraybi14StreamOrDevice"
class="reference internal" title="std"><span class="n"><span
class="pre">std</span></span></a><span class="p"><span class="pre">::</span></span><span class="n"><span class="pre">vector</span></span><span class="p"><span class="pre">\<</span></span><span class="kt"><span class="pre">int</span></span><span class="p"><span class="pre">\></span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">axes</span></span>, <span class="kt"><span class="pre">bool</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">precise</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="k"><span class="pre">false</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv47softmaxRK5arrayRKNSt6vectorIiEEb14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Softmax of an array.

<!-- -->

<span id="_CPPv37softmaxRK5arrayb14StreamOrDevice"></span><span id="_CPPv27softmaxRK5arrayb14StreamOrDevice"></span><span id="softmax__arrayCR.b.StreamOrDevice"></span><span id="group__ops_1ga1ae3614d07d873892a530d14c3857d0b" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">softmax</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="kt"><span class="pre">bool</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">precise</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="k"><span class="pre">false</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv47softmaxRK5arrayb14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Softmax of an array.

<!-- -->

<span id="_CPPv37softmaxRK5arrayib14StreamOrDevice"></span><span id="_CPPv27softmaxRK5arrayib14StreamOrDevice"></span><span id="softmax__arrayCR.i.b.StreamOrDevice"></span><span id="group__ops_1ga06f570d73716a24303e6de3aaba4457b" class="target"></span><span class="k"><span class="pre">inline</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">softmax</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="kt"><span class="pre">int</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">axis</span></span>, <span class="kt"><span class="pre">bool</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">precise</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="k"><span class="pre">false</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv47softmaxRK5arrayib14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Softmax of an array.

<!-- -->

<span id="_CPPv35powerRK5arrayRK5array14StreamOrDevice"></span><span id="_CPPv25powerRK5arrayRK5array14StreamOrDevice"></span><span id="power__arrayCR.arrayCR.StreamOrDevice"></span><span id="group__ops_1ga7972058715c26559dff9c9ae2a3ef76d" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">power</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">b</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv45powerRK5arrayRK5array14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Raise elements of a to the power of b element-wise.

<!-- -->

<span id="_CPPv36cumsumRK5arrayibb14StreamOrDevice"></span><span id="_CPPv26cumsumRK5arrayibb14StreamOrDevice"></span><span id="cumsum__arrayCR.i.b.b.StreamOrDevice"></span><span id="group__ops_1gaddc825a5c173e195ab0fda83ad630420" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">cumsum</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="kt"><span class="pre">int</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">axis</span></span>, <span class="kt"><span class="pre">bool</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">reverse</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="k"><span class="pre">false</span></span>, <span class="kt"><span class="pre">bool</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">inclusive</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="k"><span class="pre">true</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv46cumsumRK5arrayibb14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Cumulative sum of an array.

<!-- -->

<span id="_CPPv37cumprodRK5arrayibb14StreamOrDevice"></span><span id="_CPPv27cumprodRK5arrayibb14StreamOrDevice"></span><span id="cumprod__arrayCR.i.b.b.StreamOrDevice"></span><span id="group__ops_1ga0d71dfbc14ef3ed564b0c5ee26af680f" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">cumprod</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="kt"><span class="pre">int</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">axis</span></span>, <span class="kt"><span class="pre">bool</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">reverse</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="k"><span class="pre">false</span></span>, <span class="kt"><span class="pre">bool</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">inclusive</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="k"><span class="pre">true</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv47cumprodRK5arrayibb14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Cumulative product of an array.

<!-- -->

<span id="_CPPv36cummaxRK5arrayibb14StreamOrDevice"></span><span id="_CPPv26cummaxRK5arrayibb14StreamOrDevice"></span><span id="cummax__arrayCR.i.b.b.StreamOrDevice"></span><span id="group__ops_1gaee37cac8476e8f8d666bcded5bc59143" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">cummax</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="kt"><span class="pre">int</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">axis</span></span>, <span class="kt"><span class="pre">bool</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">reverse</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="k"><span class="pre">false</span></span>, <span class="kt"><span class="pre">bool</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">inclusive</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="k"><span class="pre">true</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv46cummaxRK5arrayibb14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Cumulative max of an array.

<!-- -->

<span id="_CPPv36cumminRK5arrayibb14StreamOrDevice"></span><span id="_CPPv26cumminRK5arrayibb14StreamOrDevice"></span><span id="cummin__arrayCR.i.b.b.StreamOrDevice"></span><span id="group__ops_1ga19c1bf6929fe8d66b9cd408946aea6a8" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">cummin</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="kt"><span class="pre">int</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">axis</span></span>, <span class="kt"><span class="pre">bool</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">reverse</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="k"><span class="pre">false</span></span>, <span class="kt"><span class="pre">bool</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">inclusive</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="k"><span class="pre">true</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv46cumminRK5arrayibb14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Cumulative min of an array.

<!-- -->

<span id="_CPPv312conv_general5array5arrayNSt6vectorIiEENSt6vectorIiEENSt6vectorIiEENSt6vectorIiEENSt6vectorIiEEib14StreamOrDevice"></span><span id="_CPPv212conv_general5array5arrayNSt6vectorIiEENSt6vectorIiEENSt6vectorIiEENSt6vectorIiEENSt6vectorIiEEib14StreamOrDevice"></span><span id="conv_general__array.array.std::vector:i:.std::vector:i:.std::vector:i:.std::vector:i:.std::vector:i:.i.b.StreamOrDevice"></span><span id="group__ops_1ga2236e5dfc7e52e28abf6c21675d0a51e" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">conv_general</span></span></span><span class="sig-paren">(</span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">input</span></span>, <span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">weight</span></span>, <a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4StRK5arraybi14StreamOrDevice"
class="reference internal" title="std"><span class="n"><span
class="pre">std</span></span></a><span class="p"><span class="pre">::</span></span><span class="n"><span class="pre">vector</span></span><span class="p"><span class="pre">\<</span></span><span class="kt"><span class="pre">int</span></span><span class="p"><span class="pre">\></span></span><span class="w"> </span><span class="n sig-param"><span class="pre">stride</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span>, <a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4StRK5arraybi14StreamOrDevice"
class="reference internal" title="std"><span class="n"><span
class="pre">std</span></span></a><span class="p"><span class="pre">::</span></span><span class="n"><span class="pre">vector</span></span><span class="p"><span class="pre">\<</span></span><span class="kt"><span class="pre">int</span></span><span class="p"><span class="pre">\></span></span><span class="w"> </span><span class="n sig-param"><span class="pre">padding_lo</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span>, <a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4StRK5arraybi14StreamOrDevice"
class="reference internal" title="std"><span class="n"><span
class="pre">std</span></span></a><span class="p"><span class="pre">::</span></span><span class="n"><span class="pre">vector</span></span><span class="p"><span class="pre">\<</span></span><span class="kt"><span class="pre">int</span></span><span class="p"><span class="pre">\></span></span><span class="w"> </span><span class="n sig-param"><span class="pre">padding_hi</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span>, <a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4StRK5arraybi14StreamOrDevice"
class="reference internal" title="std"><span class="n"><span
class="pre">std</span></span></a><span class="p"><span class="pre">::</span></span><span class="n"><span class="pre">vector</span></span><span class="p"><span class="pre">\<</span></span><span class="kt"><span class="pre">int</span></span><span class="p"><span class="pre">\></span></span><span class="w"> </span><span class="n sig-param"><span class="pre">kernel_dilation</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span>, <a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4StRK5arraybi14StreamOrDevice"
class="reference internal" title="std"><span class="n"><span
class="pre">std</span></span></a><span class="p"><span class="pre">::</span></span><span class="n"><span class="pre">vector</span></span><span class="p"><span class="pre">\<</span></span><span class="kt"><span class="pre">int</span></span><span class="p"><span class="pre">\></span></span><span class="w"> </span><span class="n sig-param"><span class="pre">input_dilation</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span>, <span class="kt"><span class="pre">int</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">groups</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="m"><span class="pre">1</span></span>, <span class="kt"><span class="pre">bool</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">flip</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="k"><span class="pre">false</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv412conv_general5array5arrayNSt6vectorIiEENSt6vectorIiEENSt6vectorIiEENSt6vectorIiEENSt6vectorIiEEib14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
General convolution with a filter.

<!-- -->

<span id="_CPPv312conv_generalRK5arrayRK5arrayNSt6vectorIiEENSt6vectorIiEENSt6vectorIiEENSt6vectorIiEEib14StreamOrDevice"></span><span id="_CPPv212conv_generalRK5arrayRK5arrayNSt6vectorIiEENSt6vectorIiEENSt6vectorIiEENSt6vectorIiEEib14StreamOrDevice"></span><span id="conv_general__arrayCR.arrayCR.std::vector:i:.std::vector:i:.std::vector:i:.std::vector:i:.i.b.StreamOrDevice"></span><span id="group__ops_1gab59f89942cd1efaadffe9e8762e3c99d" class="target"></span><span class="k"><span class="pre">inline</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">conv_general</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">input</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">weight</span></span>, <a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4StRK5arraybi14StreamOrDevice"
class="reference internal" title="std"><span class="n"><span
class="pre">std</span></span></a><span class="p"><span class="pre">::</span></span><span class="n"><span class="pre">vector</span></span><span class="p"><span class="pre">\<</span></span><span class="kt"><span class="pre">int</span></span><span class="p"><span class="pre">\></span></span><span class="w"> </span><span class="n sig-param"><span class="pre">stride</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span>, <a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4StRK5arraybi14StreamOrDevice"
class="reference internal" title="std"><span class="n"><span
class="pre">std</span></span></a><span class="p"><span class="pre">::</span></span><span class="n"><span class="pre">vector</span></span><span class="p"><span class="pre">\<</span></span><span class="kt"><span class="pre">int</span></span><span class="p"><span class="pre">\></span></span><span class="w"> </span><span class="n sig-param"><span class="pre">padding</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span>, <a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4StRK5arraybi14StreamOrDevice"
class="reference internal" title="std"><span class="n"><span
class="pre">std</span></span></a><span class="p"><span class="pre">::</span></span><span class="n"><span class="pre">vector</span></span><span class="p"><span class="pre">\<</span></span><span class="kt"><span class="pre">int</span></span><span class="p"><span class="pre">\></span></span><span class="w"> </span><span class="n sig-param"><span class="pre">kernel_dilation</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span>, <a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4StRK5arraybi14StreamOrDevice"
class="reference internal" title="std"><span class="n"><span
class="pre">std</span></span></a><span class="p"><span class="pre">::</span></span><span class="n"><span class="pre">vector</span></span><span class="p"><span class="pre">\<</span></span><span class="kt"><span class="pre">int</span></span><span class="p"><span class="pre">\></span></span><span class="w"> </span><span class="n sig-param"><span class="pre">input_dilation</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span>, <span class="kt"><span class="pre">int</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">groups</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="m"><span class="pre">1</span></span>, <span class="kt"><span class="pre">bool</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">flip</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="k"><span class="pre">false</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv412conv_generalRK5arrayRK5arrayNSt6vectorIiEENSt6vectorIiEENSt6vectorIiEENSt6vectorIiEEib14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
General convolution with a filter.

<!-- -->

<span id="_CPPv36conv1dRK5arrayRK5arrayiiii14StreamOrDevice"></span><span id="_CPPv26conv1dRK5arrayRK5arrayiiii14StreamOrDevice"></span><span id="conv1d__arrayCR.arrayCR.i.i.i.i.StreamOrDevice"></span><span id="group__ops_1ga30d47e08093c03a3676f235f9f559411" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">conv1d</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">input</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">weight</span></span>, <span class="kt"><span class="pre">int</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">stride</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="m"><span class="pre">1</span></span>, <span class="kt"><span class="pre">int</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">padding</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="m"><span class="pre">0</span></span>, <span class="kt"><span class="pre">int</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">dilation</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="m"><span class="pre">1</span></span>, <span class="kt"><span class="pre">int</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">groups</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="m"><span class="pre">1</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv46conv1dRK5arrayRK5arrayiiii14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
1D convolution with a filter

<!-- -->

<span id="_CPPv36conv2dRK5arrayRK5arrayRKNSt4pairIiiEERKNSt4pairIiiEERKNSt4pairIiiEEi14StreamOrDevice"></span><span id="_CPPv26conv2dRK5arrayRK5arrayRKNSt4pairIiiEERKNSt4pairIiiEERKNSt4pairIiiEEi14StreamOrDevice"></span><span id="conv2d__arrayCR.arrayCR.std::pair:i.i:CR.std::pair:i.i:CR.std::pair:i.i:CR.i.StreamOrDevice"></span><span id="group__ops_1ga73b02833229678786e7f302d458d5a83" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">conv2d</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">input</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">weight</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4StRK5arraybi14StreamOrDevice"
class="reference internal" title="std"><span class="n"><span
class="pre">std</span></span></a><span class="p"><span class="pre">::</span></span><span class="n"><span class="pre">pair</span></span><span class="p"><span class="pre">\<</span></span><span class="kt"><span class="pre">int</span></span><span class="p"><span class="pre">,</span></span><span class="w"> </span><span class="kt"><span class="pre">int</span></span><span class="p"><span class="pre">\></span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">stride</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="m"><span class="pre">1</span></span><span class="p"><span class="pre">,</span></span><span class="w"> </span><span class="m"><span class="pre">1</span></span><span class="p"><span class="pre">}</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4StRK5arraybi14StreamOrDevice"
class="reference internal" title="std"><span class="n"><span
class="pre">std</span></span></a><span class="p"><span class="pre">::</span></span><span class="n"><span class="pre">pair</span></span><span class="p"><span class="pre">\<</span></span><span class="kt"><span class="pre">int</span></span><span class="p"><span class="pre">,</span></span><span class="w"> </span><span class="kt"><span class="pre">int</span></span><span class="p"><span class="pre">\></span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">padding</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="m"><span class="pre">0</span></span><span class="p"><span class="pre">,</span></span><span class="w"> </span><span class="m"><span class="pre">0</span></span><span class="p"><span class="pre">}</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4StRK5arraybi14StreamOrDevice"
class="reference internal" title="std"><span class="n"><span
class="pre">std</span></span></a><span class="p"><span class="pre">::</span></span><span class="n"><span class="pre">pair</span></span><span class="p"><span class="pre">\<</span></span><span class="kt"><span class="pre">int</span></span><span class="p"><span class="pre">,</span></span><span class="w"> </span><span class="kt"><span class="pre">int</span></span><span class="p"><span class="pre">\></span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">dilation</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="m"><span class="pre">1</span></span><span class="p"><span class="pre">,</span></span><span class="w"> </span><span class="m"><span class="pre">1</span></span><span class="p"><span class="pre">}</span></span>, <span class="kt"><span class="pre">int</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">groups</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="m"><span class="pre">1</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv46conv2dRK5arrayRK5arrayRKNSt4pairIiiEERKNSt4pairIiiEERKNSt4pairIiiEEi14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
2D convolution with a filter

<!-- -->

<span id="_CPPv36conv3dRK5arrayRK5arrayRKNSt5tupleIiiiEERKNSt5tupleIiiiEERKNSt5tupleIiiiEEi14StreamOrDevice"></span><span id="_CPPv26conv3dRK5arrayRK5arrayRKNSt5tupleIiiiEERKNSt5tupleIiiiEERKNSt5tupleIiiiEEi14StreamOrDevice"></span><span id="conv3d__arrayCR.arrayCR.std::tuple:i.i.i:CR.std::tuple:i.i.i:CR.std::tuple:i.i.i:CR.i.StreamOrDevice"></span><span id="group__ops_1ga6e9907d2f14dc4803e4306b3dbc4b3ca" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">conv3d</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">input</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">weight</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4StRK5arraybi14StreamOrDevice"
class="reference internal" title="std"><span class="n"><span
class="pre">std</span></span></a><span class="p"><span class="pre">::</span></span><span class="n"><span class="pre">tuple</span></span><span class="p"><span class="pre">\<</span></span><span class="kt"><span class="pre">int</span></span><span class="p"><span class="pre">,</span></span><span class="w"> </span><span class="kt"><span class="pre">int</span></span><span class="p"><span class="pre">,</span></span><span class="w"> </span><span class="kt"><span class="pre">int</span></span><span class="p"><span class="pre">\></span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">stride</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="m"><span class="pre">1</span></span><span class="p"><span class="pre">,</span></span><span class="w"> </span><span class="m"><span class="pre">1</span></span><span class="p"><span class="pre">,</span></span><span class="w"> </span><span class="m"><span class="pre">1</span></span><span class="p"><span class="pre">}</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4StRK5arraybi14StreamOrDevice"
class="reference internal" title="std"><span class="n"><span
class="pre">std</span></span></a><span class="p"><span class="pre">::</span></span><span class="n"><span class="pre">tuple</span></span><span class="p"><span class="pre">\<</span></span><span class="kt"><span class="pre">int</span></span><span class="p"><span class="pre">,</span></span><span class="w"> </span><span class="kt"><span class="pre">int</span></span><span class="p"><span class="pre">,</span></span><span class="w"> </span><span class="kt"><span class="pre">int</span></span><span class="p"><span class="pre">\></span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">padding</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="m"><span class="pre">0</span></span><span class="p"><span class="pre">,</span></span><span class="w"> </span><span class="m"><span class="pre">0</span></span><span class="p"><span class="pre">,</span></span><span class="w"> </span><span class="m"><span class="pre">0</span></span><span class="p"><span class="pre">}</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4StRK5arraybi14StreamOrDevice"
class="reference internal" title="std"><span class="n"><span
class="pre">std</span></span></a><span class="p"><span class="pre">::</span></span><span class="n"><span class="pre">tuple</span></span><span class="p"><span class="pre">\<</span></span><span class="kt"><span class="pre">int</span></span><span class="p"><span class="pre">,</span></span><span class="w"> </span><span class="kt"><span class="pre">int</span></span><span class="p"><span class="pre">,</span></span><span class="w"> </span><span class="kt"><span class="pre">int</span></span><span class="p"><span class="pre">\></span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">dilation</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="m"><span class="pre">1</span></span><span class="p"><span class="pre">,</span></span><span class="w"> </span><span class="m"><span class="pre">1</span></span><span class="p"><span class="pre">,</span></span><span class="w"> </span><span class="m"><span class="pre">1</span></span><span class="p"><span class="pre">}</span></span>, <span class="kt"><span class="pre">int</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">groups</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="m"><span class="pre">1</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv46conv3dRK5arrayRK5arrayRKNSt5tupleIiiiEERKNSt5tupleIiiiEERKNSt5tupleIiiiEEi14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
3D convolution with a filter

<!-- -->

<span id="_CPPv316conv_transpose1dRK5arrayRK5arrayiiiii14StreamOrDevice"></span><span id="_CPPv216conv_transpose1dRK5arrayRK5arrayiiiii14StreamOrDevice"></span><span id="conv_transpose1d__arrayCR.arrayCR.i.i.i.i.i.StreamOrDevice"></span><span id="group__ops_1ga6e02c7cd0a5844a260b3c6e7d45e0811" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">conv_transpose1d</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">input</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">weight</span></span>, <span class="kt"><span class="pre">int</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">stride</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="m"><span class="pre">1</span></span>, <span class="kt"><span class="pre">int</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">padding</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="m"><span class="pre">0</span></span>, <span class="kt"><span class="pre">int</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">dilation</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="m"><span class="pre">1</span></span>, <span class="kt"><span class="pre">int</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">output_padding</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="m"><span class="pre">0</span></span>, <span class="kt"><span class="pre">int</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">groups</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="m"><span class="pre">1</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv416conv_transpose1dRK5arrayRK5arrayiiiii14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
1D transposed convolution with a filter

<!-- -->

<span id="_CPPv316conv_transpose2dRK5arrayRK5arrayRKNSt4pairIiiEERKNSt4pairIiiEERKNSt4pairIiiEERKNSt4pairIiiEEi14StreamOrDevice"></span><span id="_CPPv216conv_transpose2dRK5arrayRK5arrayRKNSt4pairIiiEERKNSt4pairIiiEERKNSt4pairIiiEERKNSt4pairIiiEEi14StreamOrDevice"></span><span id="conv_transpose2d__arrayCR.arrayCR.std::pair:i.i:CR.std::pair:i.i:CR.std::pair:i.i:CR.std::pair:i.i:CR.i.StreamOrDevice"></span><span id="group__ops_1ga3644f52609b78ba0b27548ee503cc34c" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">conv_transpose2d</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">input</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">weight</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4StRK5arraybi14StreamOrDevice"
class="reference internal" title="std"><span class="n"><span
class="pre">std</span></span></a><span class="p"><span class="pre">::</span></span><span class="n"><span class="pre">pair</span></span><span class="p"><span class="pre">\<</span></span><span class="kt"><span class="pre">int</span></span><span class="p"><span class="pre">,</span></span><span class="w"> </span><span class="kt"><span class="pre">int</span></span><span class="p"><span class="pre">\></span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">stride</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="m"><span class="pre">1</span></span><span class="p"><span class="pre">,</span></span><span class="w"> </span><span class="m"><span class="pre">1</span></span><span class="p"><span class="pre">}</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4StRK5arraybi14StreamOrDevice"
class="reference internal" title="std"><span class="n"><span
class="pre">std</span></span></a><span class="p"><span class="pre">::</span></span><span class="n"><span class="pre">pair</span></span><span class="p"><span class="pre">\<</span></span><span class="kt"><span class="pre">int</span></span><span class="p"><span class="pre">,</span></span><span class="w"> </span><span class="kt"><span class="pre">int</span></span><span class="p"><span class="pre">\></span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">padding</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="m"><span class="pre">0</span></span><span class="p"><span class="pre">,</span></span><span class="w"> </span><span class="m"><span class="pre">0</span></span><span class="p"><span class="pre">}</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4StRK5arraybi14StreamOrDevice"
class="reference internal" title="std"><span class="n"><span
class="pre">std</span></span></a><span class="p"><span class="pre">::</span></span><span class="n"><span class="pre">pair</span></span><span class="p"><span class="pre">\<</span></span><span class="kt"><span class="pre">int</span></span><span class="p"><span class="pre">,</span></span><span class="w"> </span><span class="kt"><span class="pre">int</span></span><span class="p"><span class="pre">\></span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">dilation</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="m"><span class="pre">1</span></span><span class="p"><span class="pre">,</span></span><span class="w"> </span><span class="m"><span class="pre">1</span></span><span class="p"><span class="pre">}</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4StRK5arraybi14StreamOrDevice"
class="reference internal" title="std"><span class="n"><span
class="pre">std</span></span></a><span class="p"><span class="pre">::</span></span><span class="n"><span class="pre">pair</span></span><span class="p"><span class="pre">\<</span></span><span class="kt"><span class="pre">int</span></span><span class="p"><span class="pre">,</span></span><span class="w"> </span><span class="kt"><span class="pre">int</span></span><span class="p"><span class="pre">\></span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">output_padding</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="m"><span class="pre">0</span></span><span class="p"><span class="pre">,</span></span><span class="w"> </span><span class="m"><span class="pre">0</span></span><span class="p"><span class="pre">}</span></span>, <span class="kt"><span class="pre">int</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">groups</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="m"><span class="pre">1</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv416conv_transpose2dRK5arrayRK5arrayRKNSt4pairIiiEERKNSt4pairIiiEERKNSt4pairIiiEERKNSt4pairIiiEEi14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
2D transposed convolution with a filter

<!-- -->

<span id="_CPPv316conv_transpose3dRK5arrayRK5arrayRKNSt5tupleIiiiEERKNSt5tupleIiiiEERKNSt5tupleIiiiEERKNSt5tupleIiiiEEi14StreamOrDevice"></span><span id="_CPPv216conv_transpose3dRK5arrayRK5arrayRKNSt5tupleIiiiEERKNSt5tupleIiiiEERKNSt5tupleIiiiEERKNSt5tupleIiiiEEi14StreamOrDevice"></span><span id="conv_transpose3d__arrayCR.arrayCR.std::tuple:i.i.i:CR.std::tuple:i.i.i:CR.std::tuple:i.i.i:CR.std::tuple:i.i.i:CR.i.StreamOrDevice"></span><span id="group__ops_1gaaa0cb6d891287c5b1b8dc87b5c48dd17" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">conv_transpose3d</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">input</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">weight</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4StRK5arraybi14StreamOrDevice"
class="reference internal" title="std"><span class="n"><span
class="pre">std</span></span></a><span class="p"><span class="pre">::</span></span><span class="n"><span class="pre">tuple</span></span><span class="p"><span class="pre">\<</span></span><span class="kt"><span class="pre">int</span></span><span class="p"><span class="pre">,</span></span><span class="w"> </span><span class="kt"><span class="pre">int</span></span><span class="p"><span class="pre">,</span></span><span class="w"> </span><span class="kt"><span class="pre">int</span></span><span class="p"><span class="pre">\></span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">stride</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="m"><span class="pre">1</span></span><span class="p"><span class="pre">,</span></span><span class="w"> </span><span class="m"><span class="pre">1</span></span><span class="p"><span class="pre">,</span></span><span class="w"> </span><span class="m"><span class="pre">1</span></span><span class="p"><span class="pre">}</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4StRK5arraybi14StreamOrDevice"
class="reference internal" title="std"><span class="n"><span
class="pre">std</span></span></a><span class="p"><span class="pre">::</span></span><span class="n"><span class="pre">tuple</span></span><span class="p"><span class="pre">\<</span></span><span class="kt"><span class="pre">int</span></span><span class="p"><span class="pre">,</span></span><span class="w"> </span><span class="kt"><span class="pre">int</span></span><span class="p"><span class="pre">,</span></span><span class="w"> </span><span class="kt"><span class="pre">int</span></span><span class="p"><span class="pre">\></span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">padding</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="m"><span class="pre">0</span></span><span class="p"><span class="pre">,</span></span><span class="w"> </span><span class="m"><span class="pre">0</span></span><span class="p"><span class="pre">,</span></span><span class="w"> </span><span class="m"><span class="pre">0</span></span><span class="p"><span class="pre">}</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4StRK5arraybi14StreamOrDevice"
class="reference internal" title="std"><span class="n"><span
class="pre">std</span></span></a><span class="p"><span class="pre">::</span></span><span class="n"><span class="pre">tuple</span></span><span class="p"><span class="pre">\<</span></span><span class="kt"><span class="pre">int</span></span><span class="p"><span class="pre">,</span></span><span class="w"> </span><span class="kt"><span class="pre">int</span></span><span class="p"><span class="pre">,</span></span><span class="w"> </span><span class="kt"><span class="pre">int</span></span><span class="p"><span class="pre">\></span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">dilation</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="m"><span class="pre">1</span></span><span class="p"><span class="pre">,</span></span><span class="w"> </span><span class="m"><span class="pre">1</span></span><span class="p"><span class="pre">,</span></span><span class="w"> </span><span class="m"><span class="pre">1</span></span><span class="p"><span class="pre">}</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4StRK5arraybi14StreamOrDevice"
class="reference internal" title="std"><span class="n"><span
class="pre">std</span></span></a><span class="p"><span class="pre">::</span></span><span class="n"><span class="pre">tuple</span></span><span class="p"><span class="pre">\<</span></span><span class="kt"><span class="pre">int</span></span><span class="p"><span class="pre">,</span></span><span class="w"> </span><span class="kt"><span class="pre">int</span></span><span class="p"><span class="pre">,</span></span><span class="w"> </span><span class="kt"><span class="pre">int</span></span><span class="p"><span class="pre">\></span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">output_padding</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="m"><span class="pre">0</span></span><span class="p"><span class="pre">,</span></span><span class="w"> </span><span class="m"><span class="pre">0</span></span><span class="p"><span class="pre">,</span></span><span class="w"> </span><span class="m"><span class="pre">0</span></span><span class="p"><span class="pre">}</span></span>, <span class="kt"><span class="pre">int</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">groups</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="m"><span class="pre">1</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv416conv_transpose3dRK5arrayRK5arrayRKNSt5tupleIiiiEERKNSt5tupleIiiiEERKNSt5tupleIiiiEERKNSt5tupleIiiiEEi14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
3D transposed convolution with a filter

<!-- -->

<span id="_CPPv316quantized_matmul5array5array5array5arraybii14StreamOrDevice"></span><span id="_CPPv216quantized_matmul5array5array5array5arraybii14StreamOrDevice"></span><span id="quantized_matmul__array.array.array.array.b.i.i.StreamOrDevice"></span><span id="group__ops_1gabfa4208fb1f9b1cdd0abc563b19175af" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">quantized_matmul</span></span></span><span class="sig-paren">(</span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">x</span></span>, <span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">w</span></span>, <span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">scales</span></span>, <span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">biases</span></span>, <span class="kt"><span class="pre">bool</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">transpose</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="k"><span class="pre">true</span></span>, <span class="kt"><span class="pre">int</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">group_size</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="m"><span class="pre">64</span></span>, <span class="kt"><span class="pre">int</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">bits</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="m"><span class="pre">4</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv416quantized_matmul5array5array5array5arraybii14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Quantized matmul multiplies x with a quantized matrix w.

<!-- -->

<span id="_CPPv38quantizeRK5arrayii14StreamOrDevice"></span><span id="_CPPv28quantizeRK5arrayii14StreamOrDevice"></span><span id="quantize__arrayCR.i.i.StreamOrDevice"></span><span id="group__ops_1gab43cc28690da7cdd43b43065adbd31da" class="target"></span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4StRK5arraybi14StreamOrDevice"
class="reference internal" title="std"><span class="n"><span
class="pre">std</span></span></a><span class="p"><span class="pre">::</span></span><span class="n"><span class="pre">tuple</span></span><span class="p"><span class="pre">\<</span></span><span class="n"><span class="pre">array</span></span><span class="p"><span class="pre">,</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="p"><span class="pre">,</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="p"><span class="pre">\></span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">quantize</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">w</span></span>, <span class="kt"><span class="pre">int</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">group_size</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="m"><span class="pre">64</span></span>, <span class="kt"><span class="pre">int</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">bits</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="m"><span class="pre">4</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv48quantizeRK5arrayii14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Quantize a matrix along its last axis.

<!-- -->

<span id="_CPPv310dequantizeRK5arrayRK5arrayRK5arrayii14StreamOrDevice"></span><span id="_CPPv210dequantizeRK5arrayRK5arrayRK5arrayii14StreamOrDevice"></span><span id="dequantize__arrayCR.arrayCR.arrayCR.i.i.StreamOrDevice"></span><span id="group__ops_1gabff758a5c1ce32ad7e8b78aba0164077" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">dequantize</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">w</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">scales</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">biases</span></span>, <span class="kt"><span class="pre">int</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">group_size</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="m"><span class="pre">64</span></span>, <span class="kt"><span class="pre">int</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">bits</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="m"><span class="pre">4</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv410dequantizeRK5arrayRK5arrayRK5arrayii14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Dequantize a matrix produced by <a
href="https://ml-explore.github.io/mlx/build/html/#group__ops_1gab43cc28690da7cdd43b43065adbd31da"
class="reference internal"><span
class="std std-ref">quantize()</span></a>

<!-- -->

<span id="_CPPv310gather_qmmRK5arrayRK5arrayRK5arrayRK5arrayNSt8optionalI5arrayEENSt8optionalI5arrayEEbiib14StreamOrDevice"></span><span id="_CPPv210gather_qmmRK5arrayRK5arrayRK5arrayRK5arrayNSt8optionalI5arrayEENSt8optionalI5arrayEEbiib14StreamOrDevice"></span><span id="gather_qmm__arrayCR.arrayCR.arrayCR.arrayCR.std::optional:array:.std::optional:array:.b.i.i.b.StreamOrDevice"></span><span id="group__ops_1ga8a358d1d4721a5c3f2b84d42dd42f898" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">gather_qmm</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">x</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">w</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">scales</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">biases</span></span>, <a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4StRK5arraybi14StreamOrDevice"
class="reference internal" title="std"><span class="n"><span
class="pre">std</span></span></a><span class="p"><span class="pre">::</span></span><span class="n"><span class="pre">optional</span></span><span class="p"><span class="pre">\<</span></span><span class="n"><span class="pre">array</span></span><span class="p"><span class="pre">\></span></span><span class="w"> </span><span class="n sig-param"><span class="pre">lhs_indices</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4StRK5arraybi14StreamOrDevice"
class="reference internal" title="std"><span class="n"><span
class="pre">std</span></span></a><span class="p"><span class="pre">::</span></span><span class="n"><span class="pre">nullopt</span></span>, <a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4StRK5arraybi14StreamOrDevice"
class="reference internal" title="std"><span class="n"><span
class="pre">std</span></span></a><span class="p"><span class="pre">::</span></span><span class="n"><span class="pre">optional</span></span><span class="p"><span class="pre">\<</span></span><span class="n"><span class="pre">array</span></span><span class="p"><span class="pre">\></span></span><span class="w"> </span><span class="n sig-param"><span class="pre">rhs_indices</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4StRK5arraybi14StreamOrDevice"
class="reference internal" title="std"><span class="n"><span
class="pre">std</span></span></a><span class="p"><span class="pre">::</span></span><span class="n"><span class="pre">nullopt</span></span>, <span class="kt"><span class="pre">bool</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">transpose</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="k"><span class="pre">true</span></span>, <span class="kt"><span class="pre">int</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">group_size</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="m"><span class="pre">64</span></span>, <span class="kt"><span class="pre">int</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">bits</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="m"><span class="pre">4</span></span>, <span class="kt"><span class="pre">bool</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">sorted_indices</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="k"><span class="pre">false</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv410gather_qmmRK5arrayRK5arrayRK5arrayRK5arrayNSt8optionalI5arrayEENSt8optionalI5arrayEEbiib14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Compute matrix products with matrix-level gather.

<!-- -->

<span id="_CPPv39tensordotRK5arrayRK5arrayKi14StreamOrDevice"></span><span id="_CPPv29tensordotRK5arrayRK5arrayKi14StreamOrDevice"></span><span id="tensordot__arrayCR.arrayCR.iC.StreamOrDevice"></span><span id="group__ops_1gaf5c9735f4690327e1500e04e728fae70" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">tensordot</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">b</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="kt"><span class="pre">int</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">axis</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="m"><span class="pre">2</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv49tensordotRK5arrayRK5arrayKi14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Returns a contraction of a and b over multiple dimensions.

<!-- -->

<span id="_CPPv39tensordotRK5arrayRK5arrayRKNSt6vectorIiEERKNSt6vectorIiEE14StreamOrDevice"></span><span id="_CPPv29tensordotRK5arrayRK5arrayRKNSt6vectorIiEERKNSt6vectorIiEE14StreamOrDevice"></span><span id="tensordot__arrayCR.arrayCR.std::vector:i:CR.std::vector:i:CR.StreamOrDevice"></span><span id="group__ops_1gad7fe00b566f89d607639c1a497cabbc6" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">tensordot</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">b</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4StRK5arraybi14StreamOrDevice"
class="reference internal" title="std"><span class="n"><span
class="pre">std</span></span></a><span class="p"><span class="pre">::</span></span><span class="n"><span class="pre">vector</span></span><span class="p"><span class="pre">\<</span></span><span class="kt"><span class="pre">int</span></span><span class="p"><span class="pre">\></span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">axes_a</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4StRK5arraybi14StreamOrDevice"
class="reference internal" title="std"><span class="n"><span
class="pre">std</span></span></a><span class="p"><span class="pre">::</span></span><span class="n"><span class="pre">vector</span></span><span class="p"><span class="pre">\<</span></span><span class="kt"><span class="pre">int</span></span><span class="p"><span class="pre">\></span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">axes_b</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv49tensordotRK5arrayRK5arrayRKNSt6vectorIiEERKNSt6vectorIiEE14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    

<!-- -->

<span id="_CPPv35outerRK5arrayRK5array14StreamOrDevice"></span><span id="_CPPv25outerRK5arrayRK5array14StreamOrDevice"></span><span id="outer__arrayCR.arrayCR.StreamOrDevice"></span><span id="group__ops_1ga866af24e10db2797e1c5a5986dbf6c0d" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">outer</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">b</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv45outerRK5arrayRK5array14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Compute the outer product of two vectors.

<!-- -->

<span id="_CPPv35innerRK5arrayRK5array14StreamOrDevice"></span><span id="_CPPv25innerRK5arrayRK5array14StreamOrDevice"></span><span id="inner__arrayCR.arrayCR.StreamOrDevice"></span><span id="group__ops_1ga654fec16a9746b390916697a2ab2546e" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">inner</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">b</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv45innerRK5arrayRK5array14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Compute the inner product of two vectors.

<!-- -->

<span id="_CPPv35addmm5array5array5arrayRKfRKf14StreamOrDevice"></span><span id="_CPPv25addmm5array5array5arrayRKfRKf14StreamOrDevice"></span><span id="addmm__array.array.array.floatCR.floatCR.StreamOrDevice"></span><span id="group__ops_1ga82a53e083205a965387b3c3e2463244a" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">addmm</span></span></span><span class="sig-paren">(</span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">c</span></span>, <span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">a</span></span>, <span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">b</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="kt"><span class="pre">float</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">alpha</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="m"><span class="pre">1.f</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="kt"><span class="pre">float</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">beta</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="m"><span class="pre">1.f</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv45addmm5array5array5arrayRKfRKf14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Compute D = beta \* C + alpha \* (A @ B)

<!-- -->

<span id="_CPPv315block_masked_mm5array5arrayiNSt8optionalI5arrayEENSt8optionalI5arrayEENSt8optionalI5arrayEE14StreamOrDevice"></span><span id="_CPPv215block_masked_mm5array5arrayiNSt8optionalI5arrayEENSt8optionalI5arrayEENSt8optionalI5arrayEE14StreamOrDevice"></span><span id="block_masked_mm__array.array.i.std::optional:array:.std::optional:array:.std::optional:array:.StreamOrDevice"></span><span id="group__ops_1ga6b76c8ea46b19e6866af155fa5910be6" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">block_masked_mm</span></span></span><span class="sig-paren">(</span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">a</span></span>, <span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">b</span></span>, <span class="kt"><span class="pre">int</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">block_size</span></span>, <a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4StRK5arraybi14StreamOrDevice"
class="reference internal" title="std"><span class="n"><span
class="pre">std</span></span></a><span class="p"><span class="pre">::</span></span><span class="n"><span class="pre">optional</span></span><span class="p"><span class="pre">\<</span></span><span class="n"><span class="pre">array</span></span><span class="p"><span class="pre">\></span></span><span class="w"> </span><span class="n sig-param"><span class="pre">mask_out</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4StRK5arraybi14StreamOrDevice"
class="reference internal" title="std"><span class="n"><span
class="pre">std</span></span></a><span class="p"><span class="pre">::</span></span><span class="n"><span class="pre">nullopt</span></span>, <a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4StRK5arraybi14StreamOrDevice"
class="reference internal" title="std"><span class="n"><span
class="pre">std</span></span></a><span class="p"><span class="pre">::</span></span><span class="n"><span class="pre">optional</span></span><span class="p"><span class="pre">\<</span></span><span class="n"><span class="pre">array</span></span><span class="p"><span class="pre">\></span></span><span class="w"> </span><span class="n sig-param"><span class="pre">mask_lhs</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4StRK5arraybi14StreamOrDevice"
class="reference internal" title="std"><span class="n"><span
class="pre">std</span></span></a><span class="p"><span class="pre">::</span></span><span class="n"><span class="pre">nullopt</span></span>, <a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4StRK5arraybi14StreamOrDevice"
class="reference internal" title="std"><span class="n"><span
class="pre">std</span></span></a><span class="p"><span class="pre">::</span></span><span class="n"><span class="pre">optional</span></span><span class="p"><span class="pre">\<</span></span><span class="n"><span class="pre">array</span></span><span class="p"><span class="pre">\></span></span><span class="w"> </span><span class="n sig-param"><span class="pre">mask_rhs</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4StRK5arraybi14StreamOrDevice"
class="reference internal" title="std"><span class="n"><span
class="pre">std</span></span></a><span class="p"><span class="pre">::</span></span><span class="n"><span class="pre">nullopt</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv415block_masked_mm5array5arrayiNSt8optionalI5arrayEENSt8optionalI5arrayEENSt8optionalI5arrayEE14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Compute matrix product with block masking.

<!-- -->

<span id="_CPPv39gather_mm5array5arrayNSt8optionalI5arrayEENSt8optionalI5arrayEEb14StreamOrDevice"></span><span id="_CPPv29gather_mm5array5arrayNSt8optionalI5arrayEENSt8optionalI5arrayEEb14StreamOrDevice"></span><span id="gather_mm__array.array.std::optional:array:.std::optional:array:.b.StreamOrDevice"></span><span id="group__ops_1gac610de1eac3aa4fdc577875108f7b59c" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">gather_mm</span></span></span><span class="sig-paren">(</span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">a</span></span>, <span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">b</span></span>, <a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4StRK5arraybi14StreamOrDevice"
class="reference internal" title="std"><span class="n"><span
class="pre">std</span></span></a><span class="p"><span class="pre">::</span></span><span class="n"><span class="pre">optional</span></span><span class="p"><span class="pre">\<</span></span><span class="n"><span class="pre">array</span></span><span class="p"><span class="pre">\></span></span><span class="w"> </span><span class="n sig-param"><span class="pre">lhs_indices</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4StRK5arraybi14StreamOrDevice"
class="reference internal" title="std"><span class="n"><span
class="pre">std</span></span></a><span class="p"><span class="pre">::</span></span><span class="n"><span class="pre">nullopt</span></span>, <a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4StRK5arraybi14StreamOrDevice"
class="reference internal" title="std"><span class="n"><span
class="pre">std</span></span></a><span class="p"><span class="pre">::</span></span><span class="n"><span class="pre">optional</span></span><span class="p"><span class="pre">\<</span></span><span class="n"><span class="pre">array</span></span><span class="p"><span class="pre">\></span></span><span class="w"> </span><span class="n sig-param"><span class="pre">rhs_indices</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4StRK5arraybi14StreamOrDevice"
class="reference internal" title="std"><span class="n"><span
class="pre">std</span></span></a><span class="p"><span class="pre">::</span></span><span class="n"><span class="pre">nullopt</span></span>, <span class="kt"><span class="pre">bool</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">sorted_indices</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="k"><span class="pre">false</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv49gather_mm5array5arrayNSt8optionalI5arrayEENSt8optionalI5arrayEEb14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Compute matrix product with matrix-level gather.

<!-- -->

<span id="_CPPv38diagonalRK5arrayiii14StreamOrDevice"></span><span id="_CPPv28diagonalRK5arrayiii14StreamOrDevice"></span><span id="diagonal__arrayCR.i.i.i.StreamOrDevice"></span><span id="group__ops_1ga9236b085a88ead3128ed8079d009cac6" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">diagonal</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="kt"><span class="pre">int</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">offset</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="m"><span class="pre">0</span></span>, <span class="kt"><span class="pre">int</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">axis1</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="m"><span class="pre">0</span></span>, <span class="kt"><span class="pre">int</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">axis2</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="m"><span class="pre">1</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv48diagonalRK5arrayiii14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Extract a diagonal or construct a diagonal array.

<!-- -->

<span id="_CPPv34diagRK5arrayi14StreamOrDevice"></span><span id="_CPPv24diagRK5arrayi14StreamOrDevice"></span><span id="diag__arrayCR.i.StreamOrDevice"></span><span id="group__ops_1ga11af511875640e1fa88e0ca87e199344" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">diag</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="kt"><span class="pre">int</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">k</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="m"><span class="pre">0</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv44diagRK5arrayi14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Extract diagonal from a 2d array or create a diagonal matrix.

<!-- -->

<span id="_CPPv35traceRK5arrayiii5Dtype14StreamOrDevice"></span><span id="_CPPv25traceRK5arrayiii5Dtype14StreamOrDevice"></span><span id="trace__arrayCR.i.i.i.Dtype.StreamOrDevice"></span><span id="group__ops_1gabf786129c7660ed8d5acb5499bc6fefd" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">trace</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="kt"><span class="pre">int</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">offset</span></span>, <span class="kt"><span class="pre">int</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">axis1</span></span>, <span class="kt"><span class="pre">int</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">axis2</span></span>, <span class="n"><span class="pre">Dtype</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">dtype</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv45traceRK5arrayiii5Dtype14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Return the sum along a specified diagonal in the given array.

<!-- -->

<span id="_CPPv35traceRK5arrayiii14StreamOrDevice"></span><span id="_CPPv25traceRK5arrayiii14StreamOrDevice"></span><span id="trace__arrayCR.i.i.i.StreamOrDevice"></span><span id="group__ops_1ga5ed43c2dbf7d6cbddbaa2fd682deaafd" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">trace</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="kt"><span class="pre">int</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">offset</span></span>, <span class="kt"><span class="pre">int</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">axis1</span></span>, <span class="kt"><span class="pre">int</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">axis2</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv45traceRK5arrayiii14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    

<!-- -->

<span id="_CPPv35traceRK5array14StreamOrDevice"></span><span id="_CPPv25traceRK5array14StreamOrDevice"></span><span id="trace__arrayCR.StreamOrDevice"></span><span id="group__ops_1gaf25c00108feaafaa6350a4434cb0062e" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">trace</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv45traceRK5array14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    

<!-- -->

<span id="_CPPv37dependsRKNSt6vectorI5arrayEERKNSt6vectorI5arrayEE"></span><span id="_CPPv27dependsRKNSt6vectorI5arrayEERKNSt6vectorI5arrayEE"></span><span id="depends__std::vector:array:CR.std::vector:array:CR"></span><span id="group__ops_1gac4a51a68fbe1725436b026d2fbb95759" class="target"></span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4StRK5arraybi14StreamOrDevice"
class="reference internal" title="std"><span class="n"><span
class="pre">std</span></span></a><span class="p"><span class="pre">::</span></span><span class="n"><span class="pre">vector</span></span><span class="p"><span class="pre">\<</span></span><span class="n"><span class="pre">array</span></span><span class="p"><span class="pre">\></span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">depends</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4StRK5arraybi14StreamOrDevice"
class="reference internal" title="std"><span class="n"><span
class="pre">std</span></span></a><span class="p"><span class="pre">::</span></span><span class="n"><span class="pre">vector</span></span><span class="p"><span class="pre">\<</span></span><span class="n"><span class="pre">array</span></span><span class="p"><span class="pre">\></span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">inputs</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4StRK5arraybi14StreamOrDevice"
class="reference internal" title="std"><span class="n"><span
class="pre">std</span></span></a><span class="p"><span class="pre">::</span></span><span class="n"><span class="pre">vector</span></span><span class="p"><span class="pre">\<</span></span><span class="n"><span class="pre">array</span></span><span class="p"><span class="pre">\></span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">dependencies</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv47dependsRKNSt6vectorI5arrayEERKNSt6vectorI5arrayEE"
class="headerlink" title="Link to this definition">#</a>    
Implements the identity function but allows injecting dependencies to
other arrays.

This ensures that these other arrays will have been computed when the
outputs of this function are computed.

<!-- -->

<span id="_CPPv310atleast_1dRK5array14StreamOrDevice"></span><span id="_CPPv210atleast_1dRK5array14StreamOrDevice"></span><span id="atleast_1d__arrayCR.StreamOrDevice"></span><span id="group__ops_1gaba4d25e7a2bf87ba4feb7837ec7fa396" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">atleast_1d</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv410atleast_1dRK5array14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
convert an array to an atleast ndim array

<!-- -->

<span id="_CPPv310atleast_1dRKNSt6vectorI5arrayEE14StreamOrDevice"></span><span id="_CPPv210atleast_1dRKNSt6vectorI5arrayEE14StreamOrDevice"></span><span id="atleast_1d__std::vector:array:CR.StreamOrDevice"></span><span id="group__ops_1ga08ca172ce80157c916c89dd0b45b95c5" class="target"></span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4StRK5arraybi14StreamOrDevice"
class="reference internal" title="std"><span class="n"><span
class="pre">std</span></span></a><span class="p"><span class="pre">::</span></span><span class="n"><span class="pre">vector</span></span><span class="p"><span class="pre">\<</span></span><span class="n"><span class="pre">array</span></span><span class="p"><span class="pre">\></span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">atleast_1d</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4StRK5arraybi14StreamOrDevice"
class="reference internal" title="std"><span class="n"><span
class="pre">std</span></span></a><span class="p"><span class="pre">::</span></span><span class="n"><span class="pre">vector</span></span><span class="p"><span class="pre">\<</span></span><span class="n"><span class="pre">array</span></span><span class="p"><span class="pre">\></span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv410atleast_1dRKNSt6vectorI5arrayEE14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    

<!-- -->

<span id="_CPPv310atleast_2dRK5array14StreamOrDevice"></span><span id="_CPPv210atleast_2dRK5array14StreamOrDevice"></span><span id="atleast_2d__arrayCR.StreamOrDevice"></span><span id="group__ops_1gaeeb7f5bb88aa32a3ac2be1f39c5f8087" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">atleast_2d</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv410atleast_2dRK5array14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    

<!-- -->

<span id="_CPPv310atleast_2dRKNSt6vectorI5arrayEE14StreamOrDevice"></span><span id="_CPPv210atleast_2dRKNSt6vectorI5arrayEE14StreamOrDevice"></span><span id="atleast_2d__std::vector:array:CR.StreamOrDevice"></span><span id="group__ops_1ga9950299a80c2562f13448758f856d1f5" class="target"></span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4StRK5arraybi14StreamOrDevice"
class="reference internal" title="std"><span class="n"><span
class="pre">std</span></span></a><span class="p"><span class="pre">::</span></span><span class="n"><span class="pre">vector</span></span><span class="p"><span class="pre">\<</span></span><span class="n"><span class="pre">array</span></span><span class="p"><span class="pre">\></span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">atleast_2d</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4StRK5arraybi14StreamOrDevice"
class="reference internal" title="std"><span class="n"><span
class="pre">std</span></span></a><span class="p"><span class="pre">::</span></span><span class="n"><span class="pre">vector</span></span><span class="p"><span class="pre">\<</span></span><span class="n"><span class="pre">array</span></span><span class="p"><span class="pre">\></span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv410atleast_2dRKNSt6vectorI5arrayEE14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    

<!-- -->

<span id="_CPPv310atleast_3dRK5array14StreamOrDevice"></span><span id="_CPPv210atleast_3dRK5array14StreamOrDevice"></span><span id="atleast_3d__arrayCR.StreamOrDevice"></span><span id="group__ops_1ga4afd919601e67782ff964465919956a0" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">atleast_3d</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv410atleast_3dRK5array14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    

<!-- -->

<span id="_CPPv310atleast_3dRKNSt6vectorI5arrayEE14StreamOrDevice"></span><span id="_CPPv210atleast_3dRKNSt6vectorI5arrayEE14StreamOrDevice"></span><span id="atleast_3d__std::vector:array:CR.StreamOrDevice"></span><span id="group__ops_1gaffdf742ad79440a60dda40062a8074fe" class="target"></span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4StRK5arraybi14StreamOrDevice"
class="reference internal" title="std"><span class="n"><span
class="pre">std</span></span></a><span class="p"><span class="pre">::</span></span><span class="n"><span class="pre">vector</span></span><span class="p"><span class="pre">\<</span></span><span class="n"><span class="pre">array</span></span><span class="p"><span class="pre">\></span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">atleast_3d</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4StRK5arraybi14StreamOrDevice"
class="reference internal" title="std"><span class="n"><span
class="pre">std</span></span></a><span class="p"><span class="pre">::</span></span><span class="n"><span class="pre">vector</span></span><span class="p"><span class="pre">\<</span></span><span class="n"><span class="pre">array</span></span><span class="p"><span class="pre">\></span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv410atleast_3dRKNSt6vectorI5arrayEE14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    

<!-- -->

<span id="_CPPv318number_of_elementsRK5arrayNSt6vectorIiEEb5Dtype14StreamOrDevice"></span><span id="_CPPv218number_of_elementsRK5arrayNSt6vectorIiEEb5Dtype14StreamOrDevice"></span><span id="number_of_elements__arrayCR.std::vector:i:.b.Dtype.StreamOrDevice"></span><span id="group__ops_1ga6d5f5f72362488b956cdc4615ef6c636" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">number_of_elements</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4StRK5arraybi14StreamOrDevice"
class="reference internal" title="std"><span class="n"><span
class="pre">std</span></span></a><span class="p"><span class="pre">::</span></span><span class="n"><span class="pre">vector</span></span><span class="p"><span class="pre">\<</span></span><span class="kt"><span class="pre">int</span></span><span class="p"><span class="pre">\></span></span><span class="w"> </span><span class="n sig-param"><span class="pre">axes</span></span>, <span class="kt"><span class="pre">bool</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">inverted</span></span>, <span class="n"><span class="pre">Dtype</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">dtype</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="n"><span class="pre">int32</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv418number_of_elementsRK5arrayNSt6vectorIiEEb5Dtype14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Extract the number of elements along some axes as a scalar array.

Used to allow shape dependent shapeless compilation (pun intended).

<!-- -->

<span id="_CPPv39conjugateRK5array14StreamOrDevice"></span><span id="_CPPv29conjugateRK5array14StreamOrDevice"></span><span id="conjugate__arrayCR.StreamOrDevice"></span><span id="group__ops_1ga5b596906bf8cdc8d97ed6ddc9aeb4c23" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">conjugate</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv49conjugateRK5array14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    

<!-- -->

<span id="_CPPv311bitwise_andRK5arrayRK5array14StreamOrDevice"></span><span id="_CPPv211bitwise_andRK5arrayRK5array14StreamOrDevice"></span><span id="bitwise_and__arrayCR.arrayCR.StreamOrDevice"></span><span id="group__ops_1ga752fd2707dabb05d0308ba3d55346ada" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">bitwise_and</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">b</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv411bitwise_andRK5arrayRK5array14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Bitwise and.

<!-- -->

<span id="_CPPv3anRK5arrayRK5array"></span><span id="_CPPv2anRK5arrayRK5array"></span><span id="and-operator__arrayCR.arrayCR"></span><span id="group__ops_1gaf0d232de4cbfffda1e2c838f8afdf6ff" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="k"><span class="pre">operator</span></span><span class="o"><span class="pre">&</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">b</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4anRK5arrayRK5array"
class="headerlink" title="Link to this definition">#</a>    

<!-- -->

<span id="_CPPv310bitwise_orRK5arrayRK5array14StreamOrDevice"></span><span id="_CPPv210bitwise_orRK5arrayRK5array14StreamOrDevice"></span><span id="bitwise_or__arrayCR.arrayCR.StreamOrDevice"></span><span id="group__ops_1ga8af4f22c08c11c4ffab7e3d45e0f3cd6" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">bitwise_or</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">b</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv410bitwise_orRK5arrayRK5array14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Bitwise inclusive or.

<!-- -->

<span id="_CPPv3orRK5arrayRK5array"></span><span id="_CPPv2orRK5arrayRK5array"></span><span id="or-operator__arrayCR.arrayCR"></span><span id="group__ops_1ga52392a2a98f09a80da8d338c4908bd02" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="k"><span class="pre">operator</span></span><span class="o"><span class="pre">\|</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">b</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4orRK5arrayRK5array"
class="headerlink" title="Link to this definition">#</a>    

<!-- -->

<span id="_CPPv311bitwise_xorRK5arrayRK5array14StreamOrDevice"></span><span id="_CPPv211bitwise_xorRK5arrayRK5array14StreamOrDevice"></span><span id="bitwise_xor__arrayCR.arrayCR.StreamOrDevice"></span><span id="group__ops_1ga3188638fba3a60e264baf69956a1e08b" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">bitwise_xor</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">b</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv411bitwise_xorRK5arrayRK5array14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Bitwise exclusive or.

<!-- -->

<span id="_CPPv3eoRK5arrayRK5array"></span><span id="_CPPv2eoRK5arrayRK5array"></span><span id="xor-operator__arrayCR.arrayCR"></span><span id="group__ops_1gac3a6fe18694e84b3d63458e9553ac181" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="k"><span class="pre">operator</span></span><span class="o"><span class="pre">^</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">b</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4eoRK5arrayRK5array"
class="headerlink" title="Link to this definition">#</a>    

<!-- -->

<span id="_CPPv310left_shiftRK5arrayRK5array14StreamOrDevice"></span><span id="_CPPv210left_shiftRK5arrayRK5array14StreamOrDevice"></span><span id="left_shift__arrayCR.arrayCR.StreamOrDevice"></span><span id="group__ops_1ga89682bf78491761e062d4ee7bef0c829" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">left_shift</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">b</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv410left_shiftRK5arrayRK5array14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Shift bits to the left.

<!-- -->

<span id="_CPPv3lsRK5arrayRK5array"></span><span id="_CPPv2lsRK5arrayRK5array"></span><span id="lshift-operator__arrayCR.arrayCR"></span><span id="group__ops_1gad656c30f9fd7d9467e405657b325aa7e" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="k"><span class="pre">operator</span></span><span class="o"><span class="pre">\<\<</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">b</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4lsRK5arrayRK5array"
class="headerlink" title="Link to this definition">#</a>    

<!-- -->

<span id="_CPPv311right_shiftRK5arrayRK5array14StreamOrDevice"></span><span id="_CPPv211right_shiftRK5arrayRK5array14StreamOrDevice"></span><span id="right_shift__arrayCR.arrayCR.StreamOrDevice"></span><span id="group__ops_1gafa376ad57d38ba87378f0272dc379b23" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">right_shift</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">b</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv411right_shiftRK5arrayRK5array14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Shift bits to the right.

<!-- -->

<span id="_CPPv3rsRK5arrayRK5array"></span><span id="_CPPv2rsRK5arrayRK5array"></span><span id="rshift-operator__arrayCR.arrayCR"></span><span id="group__ops_1ga498b61f7e8f056ae00297fa0dc17303a" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="k"><span class="pre">operator</span></span><span class="o"><span class="pre">\>\></span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">b</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4rsRK5arrayRK5array"
class="headerlink" title="Link to this definition">#</a>    

<!-- -->

<span id="_CPPv314bitwise_invertRK5array14StreamOrDevice"></span><span id="_CPPv214bitwise_invertRK5array14StreamOrDevice"></span><span id="bitwise_invert__arrayCR.StreamOrDevice"></span><span id="group__ops_1gaf1182ae7c049fbc9ee190f3e0fffbf83" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">bitwise_invert</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv414bitwise_invertRK5array14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Invert the bits.

<!-- -->

<span id="_CPPv3coRK5array"></span><span id="_CPPv2coRK5array"></span><span id="inv-operator__arrayCR"></span><span id="group__ops_1ga849365a62878579a33b3d3ad09bbc7be" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="k"><span class="pre">operator</span></span><span class="o"><span class="pre">~</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span><span class="sig-paren">)</span><a href="https://ml-explore.github.io/mlx/build/html/#_CPPv4coRK5array"
class="headerlink" title="Link to this definition">#</a>    

<!-- -->

<span id="_CPPv34viewRK5arrayRK5Dtype14StreamOrDevice"></span><span id="_CPPv24viewRK5arrayRK5Dtype14StreamOrDevice"></span><span id="view__arrayCR.DtypeCR.StreamOrDevice"></span><span id="group__ops_1ga3602aa91b7b124a0b41ec1b2137a1b02" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">view</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">Dtype</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">dtype</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv44viewRK5arrayRK5Dtype14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    

<!-- -->

<span id="_CPPv34rollRK5arrayi14StreamOrDevice"></span><span id="_CPPv24rollRK5arrayi14StreamOrDevice"></span><span id="roll__arrayCR.i.StreamOrDevice"></span><span id="group__ops_1gac40e48c69f9c715a767912c30836e75c" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">roll</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="kt"><span class="pre">int</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">shift</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv44rollRK5arrayi14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    
Roll elements along an axis and introduce them on the other side.

<!-- -->

<span id="_CPPv34rollRK5arrayRK5Shape14StreamOrDevice"></span><span id="_CPPv24rollRK5arrayRK5Shape14StreamOrDevice"></span><span id="roll__arrayCR.ShapeCR.StreamOrDevice"></span><span id="group__ops_1ga5011d1a5735c64e5b91afa56c7e2cc02" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">roll</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">Shape</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">shift</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv44rollRK5arrayRK5Shape14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    

<!-- -->

<span id="_CPPv34rollRK5arrayii14StreamOrDevice"></span><span id="_CPPv24rollRK5arrayii14StreamOrDevice"></span><span id="roll__arrayCR.i.i.StreamOrDevice"></span><span id="group__ops_1ga8694ec137165752cb6d8a36a6b7c3436" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">roll</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="kt"><span class="pre">int</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">shift</span></span>, <span class="kt"><span class="pre">int</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">axis</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv44rollRK5arrayii14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    

<!-- -->

<span id="_CPPv34rollRK5arrayiRKNSt6vectorIiEE14StreamOrDevice"></span><span id="_CPPv24rollRK5arrayiRKNSt6vectorIiEE14StreamOrDevice"></span><span id="roll__arrayCR.i.std::vector:i:CR.StreamOrDevice"></span><span id="group__ops_1ga665f502ecc96f1f4467556b784abf9ae" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">roll</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="kt"><span class="pre">int</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">shift</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4StRK5arraybi14StreamOrDevice"
class="reference internal" title="std"><span class="n"><span
class="pre">std</span></span></a><span class="p"><span class="pre">::</span></span><span class="n"><span class="pre">vector</span></span><span class="p"><span class="pre">\<</span></span><span class="kt"><span class="pre">int</span></span><span class="p"><span class="pre">\></span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">axes</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv44rollRK5arrayiRKNSt6vectorIiEE14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    

<!-- -->

<span id="_CPPv34rollRK5arrayRK5Shapei14StreamOrDevice"></span><span id="_CPPv24rollRK5arrayRK5Shapei14StreamOrDevice"></span><span id="roll__arrayCR.ShapeCR.i.StreamOrDevice"></span><span id="group__ops_1ga79137f90bc44ac9e35f408c012701df9" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">roll</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">Shape</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">shift</span></span>, <span class="kt"><span class="pre">int</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">axis</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv44rollRK5arrayRK5Shapei14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    

<!-- -->

<span id="_CPPv34rollRK5arrayRK5ShapeRKNSt6vectorIiEE14StreamOrDevice"></span><span id="_CPPv24rollRK5arrayRK5ShapeRKNSt6vectorIiEE14StreamOrDevice"></span><span id="roll__arrayCR.ShapeCR.std::vector:i:CR.StreamOrDevice"></span><span id="group__ops_1ga9d76930fb567a7d459ff96fb851abe36" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">roll</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">Shape</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">shift</span></span>, <span class="k"><span class="pre">const</span></span><span class="w"> </span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv4StRK5arraybi14StreamOrDevice"
class="reference internal" title="std"><span class="n"><span
class="pre">std</span></span></a><span class="p"><span class="pre">::</span></span><span class="n"><span class="pre">vector</span></span><span class="p"><span class="pre">\<</span></span><span class="kt"><span class="pre">int</span></span><span class="p"><span class="pre">\></span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">axes</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv44rollRK5arrayRK5ShapeRKNSt6vectorIiEE14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    

<!-- -->

<span id="_CPPv34realRK5array14StreamOrDevice"></span><span id="_CPPv24realRK5array14StreamOrDevice"></span><span id="real__arrayCR.StreamOrDevice"></span><span id="group__ops_1gaf8913cabeb9fb193ba687aaeb2087764" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">real</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv44realRK5array14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    

<!-- -->

<span id="_CPPv34imagRK5array14StreamOrDevice"></span><span id="_CPPv24imagRK5array14StreamOrDevice"></span><span id="imag__arrayCR.StreamOrDevice"></span><span id="group__ops_1ga7ff592a64d528f0cf4f3d098465da029" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">imag</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv44imagRK5array14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    

<!-- -->

<span id="_CPPv310contiguousRK5arrayb14StreamOrDevice"></span><span id="_CPPv210contiguousRK5arrayb14StreamOrDevice"></span><span id="contiguous__arrayCR.b.StreamOrDevice"></span><span id="group__ops_1ga8ab10aa6c41416d739791164a52b25d5" class="target"></span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="sig-name descname"><span class="n"><span class="pre">contiguous</span></span></span><span class="sig-paren">(</span><span class="k"><span class="pre">const</span></span><span class="w"> </span><span class="n"><span class="pre">array</span></span><span class="w"> </span><span class="p"><span class="pre">&</span></span><span class="n sig-param"><span class="pre">a</span></span>, <span class="kt"><span class="pre">bool</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">allow_col_major</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="k"><span class="pre">false</span></span>, <span class="n"><span class="pre">StreamOrDevice</span></span><span class="w"> </span><span class="n sig-param"><span class="pre">s</span></span><span class="w"> </span><span class="p"><span class="pre">=</span></span><span class="w"> </span><span class="p"><span class="pre">{</span></span><span class="p"><span class="pre">}</span></span><span class="sig-paren">)</span><a
href="https://ml-explore.github.io/mlx/build/html/#_CPPv410contiguousRK5arrayb14StreamOrDevice"
class="headerlink" title="Link to this definition">#</a>    

</div>

<div class="prev-next-area">

<a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.utils.tree_reduce.html"
class="left-prev" title="previous page"><em></em></a>

<div class="prev-next-info">

previous

mlx.utils.tree_reduce

</div>

<a
href="https://ml-explore.github.io/mlx/build/html/dev/extensions.html"
class="right-next" title="next page"></a>

<div class="prev-next-info">

next

Custom Extensions in MLX

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
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv46arangeddd5Dtype14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">arange()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv46arangeddd14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">arange()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv46arangedd5Dtype14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">arange()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv46arangedd14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">arange()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv46aranged5Dtype14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">arange()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv46aranged14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">arange()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv46arangeiii14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">arange()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv46arangeii14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">arange()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv46arangei14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">arange()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv48linspaceddi5Dtype14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">linspace()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv46astype5array5Dtype14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">astype()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv410as_strided5array5Shape7Strides6size_t14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">as_strided()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv44copy5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">copy()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv44full5Shape5array5Dtype14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">full()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv44full5Shape5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">full()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv4I0E4full5array5Shape1T5Dtype14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">full()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv4I0E4full5array5Shape1T14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">full()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv45zerosRK5Shape5Dtype14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">zeros()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv45zerosRK5Shape14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">zeros()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv410zeros_likeRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">zeros_like()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv44onesRK5Shape5Dtype14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">ones()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv44onesRK5Shape14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">ones()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv49ones_likeRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">ones_like()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv43eyeiii5Dtype14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">eye()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv43eyei5Dtype14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">eye()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv43eyeii14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">eye()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv43eyeiii14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">eye()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv43eyei14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">eye()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv48identityi5Dtype14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">identity()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv48identityi14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">identity()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv43triiii5Dtype14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">tri()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv43trii5Dtype14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">tri()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv44tril5arrayi14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">tril()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv44triu5arrayi14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">triu()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv47reshapeRK5array5Shape14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">reshape()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv49unflattenRK5arrayi5Shape14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">unflatten()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv47flattenRK5arrayii14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">flatten()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv47flattenRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">flatten()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv418hadamard_transformRK5arrayNSt8optionalIfEE14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">hadamard_transform()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv47squeezeRK5arrayRKNSt6vectorIiEE14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">squeeze()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv47squeezeRK5arrayi14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">squeeze()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv47squeezeRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">squeeze()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv411expand_dimsRK5arrayRKNSt6vectorIiEE14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">expand_dims()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv411expand_dimsRK5arrayi14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">expand_dims()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv45sliceRK5array5Shape5Shape5Shape14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">slice()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv45sliceRK5arrayNSt16initializer_listIiEE5Shape5Shape14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">slice()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv45sliceRK5array5Shape5Shape14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">slice()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv45sliceRK5arrayRK5arrayNSt6vectorIiEE5Shape14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">slice()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv412slice_updateRK5arrayRK5array5Shape5Shape5Shape14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">slice_update()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv412slice_updateRK5arrayRK5array5Shape5Shape14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">slice_update()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv412slice_updateRK5arrayRK5arrayRK5arrayNSt6vectorIiEE14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">slice_update()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv45splitRK5arrayii14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">split()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv45splitRK5arrayi14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">split()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv45splitRK5arrayRK5Shapei14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">split()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv45splitRK5arrayRK5Shape14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">split()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv48meshgridRKNSt6vectorI5arrayEEbRKNSt6stringE14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">meshgrid()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv44clipRK5arrayRKNSt8optionalI5arrayEERKNSt8optionalI5arrayEE14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">clip()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv411concatenateNSt6vectorI5arrayEEi14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">concatenate()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv411concatenateNSt6vectorI5arrayEE14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">concatenate()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv45stackRKNSt6vectorI5arrayEEi14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">stack()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv45stackRKNSt6vectorI5arrayEE14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">stack()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv46repeatRK5arrayii14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">repeat()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv46repeatRK5arrayi14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">repeat()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv44tileRK5arrayNSt6vectorIiEE14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">tile()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv49transposeRK5arrayNSt6vectorIiEE14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">transpose()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv49transposeRK5arrayNSt16initializer_listIiEE14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">transpose()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv48swapaxesRK5arrayii14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">swapaxes()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv48moveaxisRK5arrayii14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">moveaxis()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv43padRK5arrayRKNSt6vectorIiEERK5ShapeRK5ShapeRK5arrayRKNSt6stringE14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">pad()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv43padRK5arrayRKNSt6vectorINSt4pairIiiEEEERK5arrayRKNSt6stringE14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">pad()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv43padRK5arrayRKNSt4pairIiiEERK5arrayRKNSt6stringE14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">pad()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv43padRK5arrayiRK5arrayRKNSt6stringE14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">pad()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv49transposeRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">transpose()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv412broadcast_toRK5arrayRK5Shape14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">broadcast_to()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv416broadcast_arraysRKNSt6vectorI5arrayEE14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">broadcast_arrays()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv45equalRK5arrayRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">equal()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv4eqRK5arrayRK5array"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">operator==()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv4I0Eeq5array1TRK5array"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">operator==()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv4I0Eeq5arrayRK5array1T"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">operator==()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv49not_equalRK5arrayRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">not_equal()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv4neRK5arrayRK5array"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">operator!=()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv4I0Ene5array1TRK5array"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">operator!=()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv4I0Ene5arrayRK5array1T"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">operator!=()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv47greaterRK5arrayRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">greater()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv4gtRK5arrayRK5array"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">operator&gt;()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv4I0Egt5array1TRK5array"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">operator&gt;()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv4I0Egt5arrayRK5array1T"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">operator&gt;()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv413greater_equalRK5arrayRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">greater_equal()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv4geRK5arrayRK5array"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">operator&gt;=()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv4I0Ege5array1TRK5array"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">operator&gt;=()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv4I0Ege5arrayRK5array1T"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">operator&gt;=()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv44lessRK5arrayRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">less()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv4ltRK5arrayRK5array"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">operator&lt;()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv4I0Elt5array1TRK5array"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">operator&lt;()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv4I0Elt5arrayRK5array1T"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">operator&lt;()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv410less_equalRK5arrayRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">less_equal()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv4leRK5arrayRK5array"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">operator&lt;=()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv4I0Ele5array1TRK5array"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">operator&lt;=()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv4I0Ele5arrayRK5array1T"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">operator&lt;=()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv411array_equalRK5arrayRK5arrayb14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">array_equal()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv411array_equalRK5arrayRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">array_equal()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv45isnanRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">isnan()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv45isinfRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">isinf()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv48isfiniteRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">isfinite()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv48isposinfRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">isposinf()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv48isneginfRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">isneginf()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv45whereRK5arrayRK5arrayRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">where()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv410nan_to_numRK5arrayfKNSt8optionalIfEEKNSt8optionalIfEE14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">nan_to_num()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv43allRK5arrayb14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">all()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv43allRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">all()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv48allcloseRK5arrayRK5arrayddb14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">allclose()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv47iscloseRK5arrayRK5arrayddb14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">isclose()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv43allRK5arrayRKNSt6vectorIiEEb14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">all()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv43allRK5arrayib14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">all()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv43anyRK5arrayb14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">any()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv43anyRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">any()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv43anyRK5arrayRKNSt6vectorIiEEb14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">any()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv43anyRK5arrayib14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">any()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv43sumRK5arrayb14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">sum()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv43sumRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">sum()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv43sumRK5arrayRKNSt6vectorIiEEb14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">sum()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv43sumRK5arrayib14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">sum()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv44meanRK5arrayb14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">mean()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv44meanRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">mean()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv44meanRK5arrayRKNSt6vectorIiEEb14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">mean()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv44meanRK5arrayib14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">mean()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv43varRK5arraybi14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">var()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv43varRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">var()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv43varRK5arrayRKNSt6vectorIiEEbi14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">var()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv43varRK5arrayibi14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">var()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv4StRK5arraybi14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">std()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv4StRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">std()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv4StRK5arrayRKNSt6vectorIiEEbi14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">std()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv4StRK5arrayibi14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">std()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv44prodRK5arrayb14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">prod()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv44prodRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">prod()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv44prodRK5arrayRKNSt6vectorIiEEb14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">prod()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv44prodRK5arrayib14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">prod()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv43maxRK5arrayb14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">max()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv43maxRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">max()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv43maxRK5arrayRKNSt6vectorIiEEb14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">max()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv43maxRK5arrayib14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">max()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv43minRK5arrayb14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">min()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv43minRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">min()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv43minRK5arrayRKNSt6vectorIiEEb14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">min()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv43minRK5arrayib14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">min()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv46argminRK5arrayb14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">argmin()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv46argminRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">argmin()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv46argminRK5arrayib14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">argmin()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv46argmaxRK5arrayb14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">argmax()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv46argmaxRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">argmax()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv46argmaxRK5arrayib14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">argmax()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv44sortRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">sort()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv44sortRK5arrayi14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">sort()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv47argsortRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">argsort()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv47argsortRK5arrayi14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">argsort()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv49partitionRK5arrayi14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">partition()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv49partitionRK5arrayii14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">partition()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv412argpartitionRK5arrayi14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">argpartition()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv412argpartitionRK5arrayii14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">argpartition()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv44topkRK5arrayi14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">topk()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv44topkRK5arrayii14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">topk()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv412logcumsumexpRK5arrayibb14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">logcumsumexp()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv49logsumexpRK5arrayb14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">logsumexp()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv49logsumexpRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">logsumexp()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv49logsumexpRK5arrayRKNSt6vectorIiEEb14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">logsumexp()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv49logsumexpRK5arrayib14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">logsumexp()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv43absRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">abs()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv48negativeRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">negative()</code></span></a>
- <a href="https://ml-explore.github.io/mlx/build/html/#_CPPv4miRK5array"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">operator-()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv44signRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">sign()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv411logical_notRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">logical_not()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv411logical_andRK5arrayRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">logical_and()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv4aaRK5arrayRK5array"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">operator&amp;&amp;()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv410logical_orRK5arrayRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">logical_or()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv4ooRK5arrayRK5array"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">operator||()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv410reciprocalRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">reciprocal()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv43addRK5arrayRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">add()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv4plRK5arrayRK5array"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">operator+()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv4I0Epl5array1TRK5array"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">operator+()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv4I0Epl5arrayRK5array1T"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">operator+()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv48subtractRK5arrayRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">subtract()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv4miRK5arrayRK5array"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">operator-()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv4I0Emi5array1TRK5array"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">operator-()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv4I0Emi5arrayRK5array1T"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">operator-()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv48multiplyRK5arrayRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">multiply()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv4mlRK5arrayRK5array"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">operator*()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv4I0Eml5array1TRK5array"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">operator*()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv4I0Eml5arrayRK5array1T"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">operator*()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv46divideRK5arrayRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">divide()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv4dvRK5arrayRK5array"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">operator/()</code></span></a>
- <a href="https://ml-explore.github.io/mlx/build/html/#_CPPv4dvdRK5array"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">operator/()</code></span></a>
- <a href="https://ml-explore.github.io/mlx/build/html/#_CPPv4dvRK5arrayd"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">operator/()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv46divmodRK5arrayRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">divmod()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv412floor_divideRK5arrayRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">floor_divide()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv49remainderRK5arrayRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">remainder()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv4rmRK5arrayRK5array"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">operator%()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv4I0Erm5array1TRK5array"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">operator%()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv4I0Erm5arrayRK5array1T"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">operator%()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv47maximumRK5arrayRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">maximum()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv47minimumRK5arrayRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">minimum()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv45floorRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">floor()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv44ceilRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">ceil()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv46squareRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">square()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv43expRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">exp()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv43sinRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">sin()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv43cosRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">cos()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv43tanRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">tan()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv46arcsinRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">arcsin()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv46arccosRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">arccos()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv46arctanRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">arctan()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv47arctan2RK5arrayRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">arctan2()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv44sinhRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">sinh()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv44coshRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">cosh()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv44tanhRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">tanh()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv47arcsinhRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">arcsinh()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv47arccoshRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">arccosh()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv47arctanhRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">arctanh()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv47degreesRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">degrees()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv47radiansRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">radians()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv43logRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">log()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv44log2RK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">log2()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv45log10RK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">log10()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv45log1pRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">log1p()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv49logaddexpRK5arrayRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">logaddexp()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv47sigmoidRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">sigmoid()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv43erfRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">erf()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv46erfinvRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">erfinv()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv45expm1RK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">expm1()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv413stop_gradientRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">stop_gradient()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv45roundRK5arrayi14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">round()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv45roundRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">round()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv46matmulRK5arrayRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">matmul()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv46gatherRK5arrayRKNSt6vectorI5arrayEERKNSt6vectorIiEERK5Shape14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">gather()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv46gatherRK5arrayRK5arrayiRK5Shape14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">gather()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv44kronRK5arrayRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">kron()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv44takeRK5arrayRK5arrayi14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">take()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv44takeRK5arrayii14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">take()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv44takeRK5arrayRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">take()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv44takeRK5arrayi14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">take()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv415take_along_axisRK5arrayRK5arrayi14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">take_along_axis()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv414put_along_axisRK5arrayRK5arrayRK5arrayi14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">put_along_axis()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv416scatter_add_axisRK5arrayRK5arrayRK5arrayi14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">scatter_add_axis()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv47scatterRK5arrayRKNSt6vectorI5arrayEERK5arrayRKNSt6vectorIiEE14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">scatter()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv47scatterRK5arrayRK5arrayRK5arrayi14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">scatter()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv411scatter_addRK5arrayRKNSt6vectorI5arrayEERK5arrayRKNSt6vectorIiEE14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">scatter_add()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv411scatter_addRK5arrayRK5arrayRK5arrayi14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">scatter_add()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv412scatter_prodRK5arrayRKNSt6vectorI5arrayEERK5arrayRKNSt6vectorIiEE14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">scatter_prod()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv412scatter_prodRK5arrayRK5arrayRK5arrayi14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">scatter_prod()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv411scatter_maxRK5arrayRKNSt6vectorI5arrayEERK5arrayRKNSt6vectorIiEE14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">scatter_max()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv411scatter_maxRK5arrayRK5arrayRK5arrayi14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">scatter_max()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv411scatter_minRK5arrayRKNSt6vectorI5arrayEERK5arrayRKNSt6vectorIiEE14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">scatter_min()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv411scatter_minRK5arrayRK5arrayRK5arrayi14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">scatter_min()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv44sqrtRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">sqrt()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv45rsqrtRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">rsqrt()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv47softmaxRK5arrayRKNSt6vectorIiEEb14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">softmax()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv47softmaxRK5arrayb14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">softmax()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv47softmaxRK5arrayib14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">softmax()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv45powerRK5arrayRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">power()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv46cumsumRK5arrayibb14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">cumsum()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv47cumprodRK5arrayibb14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">cumprod()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv46cummaxRK5arrayibb14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">cummax()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv46cumminRK5arrayibb14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">cummin()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv412conv_general5array5arrayNSt6vectorIiEENSt6vectorIiEENSt6vectorIiEENSt6vectorIiEENSt6vectorIiEEib14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">conv_general()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv412conv_generalRK5arrayRK5arrayNSt6vectorIiEENSt6vectorIiEENSt6vectorIiEENSt6vectorIiEEib14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">conv_general()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv46conv1dRK5arrayRK5arrayiiii14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">conv1d()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv46conv2dRK5arrayRK5arrayRKNSt4pairIiiEERKNSt4pairIiiEERKNSt4pairIiiEEi14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">conv2d()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv46conv3dRK5arrayRK5arrayRKNSt5tupleIiiiEERKNSt5tupleIiiiEERKNSt5tupleIiiiEEi14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">conv3d()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv416conv_transpose1dRK5arrayRK5arrayiiiii14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">conv_transpose1d()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv416conv_transpose2dRK5arrayRK5arrayRKNSt4pairIiiEERKNSt4pairIiiEERKNSt4pairIiiEERKNSt4pairIiiEEi14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">conv_transpose2d()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv416conv_transpose3dRK5arrayRK5arrayRKNSt5tupleIiiiEERKNSt5tupleIiiiEERKNSt5tupleIiiiEERKNSt5tupleIiiiEEi14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">conv_transpose3d()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv416quantized_matmul5array5array5array5arraybii14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">quantized_matmul()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv48quantizeRK5arrayii14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">quantize()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv410dequantizeRK5arrayRK5arrayRK5arrayii14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">dequantize()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv410gather_qmmRK5arrayRK5arrayRK5arrayRK5arrayNSt8optionalI5arrayEENSt8optionalI5arrayEEbiib14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">gather_qmm()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv49tensordotRK5arrayRK5arrayKi14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">tensordot()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv49tensordotRK5arrayRK5arrayRKNSt6vectorIiEERKNSt6vectorIiEE14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">tensordot()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv45outerRK5arrayRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">outer()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv45innerRK5arrayRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">inner()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv45addmm5array5array5arrayRKfRKf14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">addmm()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv415block_masked_mm5array5arrayiNSt8optionalI5arrayEENSt8optionalI5arrayEENSt8optionalI5arrayEE14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">block_masked_mm()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv49gather_mm5array5arrayNSt8optionalI5arrayEENSt8optionalI5arrayEEb14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">gather_mm()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv48diagonalRK5arrayiii14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">diagonal()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv44diagRK5arrayi14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">diag()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv45traceRK5arrayiii5Dtype14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">trace()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv45traceRK5arrayiii14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">trace()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv45traceRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">trace()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv47dependsRKNSt6vectorI5arrayEERKNSt6vectorI5arrayEE"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">depends()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv410atleast_1dRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">atleast_1d()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv410atleast_1dRKNSt6vectorI5arrayEE14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">atleast_1d()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv410atleast_2dRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">atleast_2d()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv410atleast_2dRKNSt6vectorI5arrayEE14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">atleast_2d()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv410atleast_3dRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">atleast_3d()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv410atleast_3dRKNSt6vectorI5arrayEE14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">atleast_3d()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv418number_of_elementsRK5arrayNSt6vectorIiEEb5Dtype14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">number_of_elements()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv49conjugateRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">conjugate()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv411bitwise_andRK5arrayRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">bitwise_and()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv4anRK5arrayRK5array"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">operator&amp;()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv410bitwise_orRK5arrayRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">bitwise_or()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv4orRK5arrayRK5array"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">operator|()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv411bitwise_xorRK5arrayRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">bitwise_xor()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv4eoRK5arrayRK5array"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">operator^()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv410left_shiftRK5arrayRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">left_shift()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv4lsRK5arrayRK5array"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">operator&lt;&lt;()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv411right_shiftRK5arrayRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">right_shift()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv4rsRK5arrayRK5array"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">operator&gt;&gt;()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv414bitwise_invertRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">bitwise_invert()</code></span></a>
- <a href="https://ml-explore.github.io/mlx/build/html/#_CPPv4coRK5array"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">operator~()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv44viewRK5arrayRK5Dtype14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">view()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv44rollRK5arrayi14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">roll()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv44rollRK5arrayRK5Shape14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">roll()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv44rollRK5arrayii14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">roll()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv44rollRK5arrayiRKNSt6vectorIiEE14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">roll()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv44rollRK5arrayRK5Shapei14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">roll()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv44rollRK5arrayRK5ShapeRKNSt6vectorIiEE14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">roll()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv44realRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">real()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv44imagRK5array14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">imag()</code></span></a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#_CPPv410contiguousRK5arrayb14StreamOrDevice"
  class="reference internal nav-link"><span class="pre"><code
  class="docutils literal notranslate">contiguous()</code></span></a>

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
