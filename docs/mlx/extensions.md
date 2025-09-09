Curator's note: Prefer the human-authored MLX guides for clarity.
- ../docs_curated/README.md
- ../docs_curated/PYTORCH_DISSONANCE.md
- ../docs_curated/NUMPY_USERS.md
- ../docs_curated/COMMON_PITFALLS.md

<!--
Per-file analysis (extensions.md):
- Developer doc for adding custom primitives/ops with CPU/GPU backends.
- The highest value is build guidance and minimal validation harness.
-->

## Curated Notes

- Start with a Python shim that checks dtypes/shapes, then dispatches to your C++ op; fail early with clear errors.
- Provide both CPU and GPU paths; feature‑gate GPU behind `mx.metal.is_available()`.
- Add a tiny correctness test: compare to a NumPy/MLX reference on small inputs, including non‑contiguous views.


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
  href="https://ml-explore.github.io/mlx/build/html/_sources/dev/extensions.rst"
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

# Custom Extensions in MLX

<div id="print-main-content">

<div id="jb-print-toc">

<div>

## Contents

</div>

- <a
  href="https://ml-explore.github.io/mlx/build/html/#introducing-the-example"
  class="reference internal nav-link">Introducing the Example</a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#operations-and-primitives"
  class="reference internal nav-link">Operations and Primitives</a>
  - <a href="https://ml-explore.github.io/mlx/build/html/#operations"
    class="reference internal nav-link">Operations</a>
  - <a href="https://ml-explore.github.io/mlx/build/html/#primitives"
    class="reference internal nav-link">Primitives</a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/#using-the-primitive"
    class="reference internal nav-link">Using the Primitive</a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#implementing-the-primitive"
  class="reference internal nav-link">Implementing the Primitive</a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/#implementing-the-cpu-back-end"
    class="reference internal nav-link">Implementing the CPU Back-end</a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/#implementing-the-gpu-back-end"
    class="reference internal nav-link">Implementing the GPU Back-end</a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/#primitive-transforms"
    class="reference internal nav-link">Primitive Transforms</a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#building-and-binding"
  class="reference internal nav-link">Building and Binding</a>
  - <a href="https://ml-explore.github.io/mlx/build/html/#binding-to-python"
    class="reference internal nav-link">Binding to Python</a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/#building-with-cmake"
    class="reference internal nav-link">Building with CMake</a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/#building-with-setuptools"
    class="reference internal nav-link">Building with <span
    class="pre"><code
    class="docutils literal notranslate">setuptools</code></span></a>
- <a href="https://ml-explore.github.io/mlx/build/html/#usage"
  class="reference internal nav-link">Usage</a>
  - <a href="https://ml-explore.github.io/mlx/build/html/#results"
    class="reference internal nav-link">Results</a>
- <a href="https://ml-explore.github.io/mlx/build/html/#scripts"
  class="reference internal nav-link">Scripts</a>

</div>

</div>

</div>

<div id="searchbox">

</div>

<div id="custom-extensions-in-mlx" class="section">

# Custom Extensions in MLX<a
href="https://ml-explore.github.io/mlx/build/html/#custom-extensions-in-mlx"
class="headerlink" title="Link to this heading">#</a>

You can extend MLX with custom operations on the CPU or GPU. This guide
explains how to do that with a simple example.

<div id="introducing-the-example" class="section">

## Introducing the Example<a
href="https://ml-explore.github.io/mlx/build/html/#introducing-the-example"
class="headerlink" title="Link to this heading">#</a>

Let’s say you would like an operation that takes in two arrays,
<span class="pre">`x`</span> and <span class="pre">`y`</span>, scales
them both by coefficients <span class="pre">`alpha`</span> and
<span class="pre">`beta`</span> respectively, and then adds them
together to get the result
<span class="pre">`z`</span>` `<span class="pre">`=`</span>` `<span class="pre">`alpha`</span>` `<span class="pre">`*`</span>` `<span class="pre">`x`</span>` `<span class="pre">`+`</span>` `<span class="pre">`beta`</span>` `<span class="pre">`*`</span>` `<span class="pre">`y`</span>.
You can do that in MLX directly:

<div class="highlight-python notranslate">

<div class="highlight">

    import mlx.core as mx

    def simple_axpby(x: mx.array, y: mx.array, alpha: float, beta: float) -> mx.array:
        return alpha * x + beta * y

</div>

</div>

This function performs that operation while leaving the implementation
and function transformations to MLX.

However, you may want to customize the underlying implementation,
perhaps to make it faster. In this tutorial we will go through adding
custom extensions. It will cover:

- The structure of the MLX library.

- Implementing a CPU operation.

- Implementing a GPU operation using metal.

- Adding the <span class="pre">`vjp`</span> and
  <span class="pre">`jvp`</span> function transformation.

- Building a custom extension and binding it to python.

</div>

<div id="operations-and-primitives" class="section">

## Operations and Primitives<a
href="https://ml-explore.github.io/mlx/build/html/#operations-and-primitives"
class="headerlink" title="Link to this heading">#</a>

Operations in MLX build the computation graph. Primitives provide the
rules for evaluating and transforming the graph. Let’s start by
discussing operations in more detail.

<div id="operations" class="section">

### Operations<a href="https://ml-explore.github.io/mlx/build/html/#operations"
class="headerlink" title="Link to this heading">#</a>

Operations are the front-end functions that operate on arrays. They are
defined in the C++ API (<a
href="https://ml-explore.github.io/mlx/build/html/cpp/ops.html#cpp-ops"
class="reference internal"><span
class="std std-ref">Operations</span></a>), and the Python API (<a
href="https://ml-explore.github.io/mlx/build/html/python/ops.html#ops"
class="reference internal"><span
class="std std-ref">Operations</span></a>) binds them.

We would like an operation <span class="pre">`axpby()`</span> that takes
in two arrays, <span class="pre">`x`</span> and
<span class="pre">`y`</span>, and two scalars,
<span class="pre">`alpha`</span> and <span class="pre">`beta`</span>.
This is how to define it in C++:

<div class="highlight-C++ notranslate">

<div class="highlight">

    /**
    *  Scale and sum two vectors element-wise
    *  z = alpha * x + beta * y
    *
    *  Use NumPy-style broadcasting between x and y
    *  Inputs are upcasted to floats if needed
    **/
    array axpby(
        const array& x, // Input array x
        const array& y, // Input array y
        const float alpha, // Scaling factor for x
        const float beta, // Scaling factor for y
        StreamOrDevice s = {} // Stream on which to schedule the operation
    );

</div>

</div>

The simplest way to implement this is with existing operations:

<div class="highlight-C++ notranslate">

<div class="highlight">

    array axpby(
        const array& x, // Input array x
        const array& y, // Input array y
        const float alpha, // Scaling factor for x
        const float beta, // Scaling factor for y
        StreamOrDevice s /* = {} */ // Stream on which to schedule the operation
    ) {
        // Scale x and y on the provided stream
        auto ax = multiply(array(alpha), x, s);
        auto by = multiply(array(beta), y, s);

        // Add and return
        return add(ax, by, s);
    }

</div>

</div>

The operations themselves do not contain the implementations that act on
the data, nor do they contain the rules of transformations. Rather, they
are an easy to use interface that use
<span class="pre">`Primitive`</span> building blocks.

</div>

<div id="primitives" class="section">

### Primitives<a href="https://ml-explore.github.io/mlx/build/html/#primitives"
class="headerlink" title="Link to this heading">#</a>

A <span class="pre">`Primitive`</span> is part of the computation graph
of an <span class="pre">`array`</span>. It defines how to create output
arrays given input arrays. Further, a
<span class="pre">`Primitive`</span> has methods to run on the CPU or
GPU and for function transformations such as
<span class="pre">`vjp`</span> and <span class="pre">`jvp`</span>. Let’s
go back to our example to be more concrete:

<div class="highlight-C++ notranslate">

<div class="highlight">

    class Axpby : public Primitive {
      public:
        explicit Axpby(Stream stream, float alpha, float beta)
            : Primitive(stream), alpha_(alpha), beta_(beta){};

        /**
        * A primitive must know how to evaluate itself on the CPU/GPU
        * for the given inputs and populate the output array.
        *
        * To avoid unnecessary allocations, the evaluation function
        * is responsible for allocating space for the array.
        */
        void eval_cpu(
            const std::vector<array>& inputs,
            std::vector<array>& outputs) override;
        void eval_gpu(
            const std::vector<array>& inputs,
            std::vector<array>& outputs) override;

        /** The Jacobian-vector product. */
        std::vector<array> jvp(
            const std::vector<array>& primals,
            const std::vector<array>& tangents,
            const std::vector<int>& argnums) override;

        /** The vector-Jacobian product. */
        std::vector<array> vjp(
            const std::vector<array>& primals,
            const std::vector<array>& cotangents,
            const std::vector<int>& argnums,
            const std::vector<array>& outputs) override;

        /**
        * The primitive must know how to vectorize itself across
        * the given axes. The output is a pair containing the array
        * representing the vectorized computation and the axis which
        * corresponds to the output vectorized dimension.
        */
        virtual std::pair<std::vector<array>, std::vector<int>> vmap(
            const std::vector<array>& inputs,
            const std::vector<int>& axes) override;

        /** Print the primitive. */
        void print(std::ostream& os) override {
            os << "Axpby";
        }

        /** Equivalence check **/
        bool is_equivalent(const Primitive& other) const override;

      private:
        float alpha_;
        float beta_;
    };

</div>

</div>

The <span class="pre">`Axpby`</span> class derives from the base
<span class="pre">`Primitive`</span> class. The
<span class="pre">`Axpby`</span> treats <span class="pre">`alpha`</span>
and <span class="pre">`beta`</span> as parameters. It then provides
implementations of how the output array is produced given the inputs
through <span class="pre">`Axpby::eval_cpu()`</span> and
<span class="pre">`Axpby::eval_gpu()`</span>. It also provides rules of
transformations in <span class="pre">`Axpby::jvp()`</span>,
<span class="pre">`Axpby::vjp()`</span>, and
<span class="pre">`Axpby::vmap()`</span>.

</div>

<div id="using-the-primitive" class="section">

### Using the Primitive<a
href="https://ml-explore.github.io/mlx/build/html/#using-the-primitive"
class="headerlink" title="Link to this heading">#</a>

Operations can use this <span class="pre">`Primitive`</span> to add a
new <span class="pre">`array`</span> to the computation graph. An
<span class="pre">`array`</span> can be constructed by providing its
data type, shape, the <span class="pre">`Primitive`</span> that computes
it, and the <span class="pre">`array`</span> inputs that are passed to
the primitive.

Let’s reimplement our operation now in terms of our
<span class="pre">`Axpby`</span> primitive.

<div class="highlight-C++ notranslate">

<div class="highlight">

    array axpby(
        const array& x, // Input array x
        const array& y, // Input array y
        const float alpha, // Scaling factor for x
        const float beta, // Scaling factor for y
        StreamOrDevice s /* = {} */ // Stream on which to schedule the operation
    ) {
        // Promote dtypes between x and y as needed
        auto promoted_dtype = promote_types(x.dtype(), y.dtype());

        // Upcast to float32 for non-floating point inputs x and y
        auto out_dtype = issubdtype(promoted_dtype, float32)
            ? promoted_dtype
            : promote_types(promoted_dtype, float32);

        // Cast x and y up to the determined dtype (on the same stream s)
        auto x_casted = astype(x, out_dtype, s);
        auto y_casted = astype(y, out_dtype, s);

        // Broadcast the shapes of x and y (on the same stream s)
        auto broadcasted_inputs = broadcast_arrays({x_casted, y_casted}, s);
        auto out_shape = broadcasted_inputs[0].shape();

        // Construct the array as the output of the Axpby primitive
        // with the broadcasted and upcasted arrays as inputs
        return array(
            /* const std::vector<int>& shape = */ out_shape,
            /* Dtype dtype = */ out_dtype,
            /* std::unique_ptr<Primitive> primitive = */
            std::make_shared<Axpby>(to_stream(s), alpha, beta),
            /* const std::vector<array>& inputs = */ broadcasted_inputs);
    }

</div>

</div>

This operation now handles the following:

1.  Upcast inputs and resolve the output data type.

2.  Broadcast the inputs and resolve the output shape.

3.  Construct the primitive <span class="pre">`Axpby`</span> using the
    given stream, <span class="pre">`alpha`</span>, and
    <span class="pre">`beta`</span>.

4.  Construct the output <span class="pre">`array`</span> using the
    primitive and the inputs.

</div>

</div>

<div id="implementing-the-primitive" class="section">

## Implementing the Primitive<a
href="https://ml-explore.github.io/mlx/build/html/#implementing-the-primitive"
class="headerlink" title="Link to this heading">#</a>

No computation happens when we call the operation alone. The operation
only builds the computation graph. When we evaluate the output array,
MLX schedules the execution of the computation graph, and calls
<span class="pre">`Axpby::eval_cpu()`</span> or
<span class="pre">`Axpby::eval_gpu()`</span> depending on the
stream/device specified by the user.

<div class="admonition warning">

Warning

When <span class="pre">`Primitive::eval_cpu()`</span> or
<span class="pre">`Primitive::eval_gpu()`</span> are called, no memory
has been allocated for the output array. It falls on the implementation
of these functions to allocate memory as needed.

</div>

<div id="implementing-the-cpu-back-end" class="section">

### Implementing the CPU Back-end<a
href="https://ml-explore.github.io/mlx/build/html/#implementing-the-cpu-back-end"
class="headerlink" title="Link to this heading">#</a>

Let’s start by implementing
<span class="pre">`Axpby::eval_cpu()`</span>.

The method will go over each element of the output array, find the
corresponding input elements of <span class="pre">`x`</span> and
<span class="pre">`y`</span> and perform the operation point-wise. This
is captured in the templated function
<span class="pre">`axpby_impl()`</span>.

<div class="highlight-C++ notranslate">

<div class="highlight">

    template <typename T>
    void axpby_impl(
        const mx::array& x,
        const mx::array& y,
        mx::array& out,
        float alpha_,
        float beta_,
        mx::Stream stream) {
      out.set_data(mx::allocator::malloc(out.nbytes()));

      // Get the CPU command encoder and register input and output arrays
      auto& encoder = mx::cpu::get_command_encoder(stream);
      encoder.set_input_array(x);
      encoder.set_input_array(y);
      encoder.set_output_array(out);

      // Launch the CPU kernel
      encoder.dispatch([x_ptr = x.data<T>(),
                        y_ptr = y.data<T>(),
                        out_ptr = out.data<T>(),
                        size = out.size(),
                        shape = out.shape(),
                        x_strides = x.strides(),
                        y_strides = y.strides(),
                        alpha_,
                        beta_]() {

        // Cast alpha and beta to the relevant types
        T alpha = static_cast<T>(alpha_);
        T beta = static_cast<T>(beta_);

        // Do the element-wise operation for each output
        for (size_t out_idx = 0; out_idx < size; out_idx++) {
          // Map linear indices to offsets in x and y
          auto x_offset = mx::elem_to_loc(out_idx, shape, x_strides);
          auto y_offset = mx::elem_to_loc(out_idx, shape, y_strides);

          // We allocate the output to be contiguous and regularly strided
          // (defaults to row major) and hence it doesn't need additional mapping
          out_ptr[out_idx] = alpha * x_ptr[x_offset] + beta * y_ptr[y_offset];
        }
      });
    }

</div>

</div>

Our implementation should work for all incoming floating point arrays.
Accordingly, we add dispatches for <span class="pre">`float32`</span>,
<span class="pre">`float16`</span>, <span class="pre">`bfloat16`</span>
and <span class="pre">`complex64`</span>. We throw an error if we
encounter an unexpected type.

<div class="highlight-C++ notranslate">

<div class="highlight">

    void Axpby::eval_cpu(
        const std::vector<mx::array>& inputs,
        std::vector<mx::array>& outputs) {
      auto& x = inputs[0];
      auto& y = inputs[1];
      auto& out = outputs[0];

      // Dispatch to the correct dtype
      if (out.dtype() == mx::float32) {
        return axpby_impl<float>(x, y, out, alpha_, beta_, stream());
      } else if (out.dtype() == mx::float16) {
        return axpby_impl<mx::float16_t>(x, y, out, alpha_, beta_, stream());
      } else if (out.dtype() == mx::bfloat16) {
        return axpby_impl<mx::bfloat16_t>(x, y, out, alpha_, beta_, stream());
      } else if (out.dtype() == mx::complex64) {
        return axpby_impl<mx::complex64_t>(x, y, out, alpha_, beta_, stream());
      } else {
        throw std::runtime_error(
            "Axpby is only supported for floating point types.");
      }
    }

</div>

</div>

Just this much is enough to run the operation
<span class="pre">`axpby()`</span> on a CPU stream! If you do not plan
on running the operation on the GPU or using transforms on computation
graphs that contain <span class="pre">`Axpby`</span>, you can stop
implementing the primitive here.

</div>

<div id="implementing-the-gpu-back-end" class="section">

### Implementing the GPU Back-end<a
href="https://ml-explore.github.io/mlx/build/html/#implementing-the-gpu-back-end"
class="headerlink" title="Link to this heading">#</a>

Apple silicon devices address their GPUs using the
<a href="https://developer.apple.com/documentation/metal?language=objc"
class="reference external">Metal</a> shading language, and GPU kernels
in MLX are written using Metal.

<div class="admonition note">

Note

Here are some helpful resources if you are new to Metal:

- A walkthrough of the metal compute pipeline: <a
  href="https://developer.apple.com/documentation/metal/performing_calculations_on_a_gpu?language=objc"
  class="reference external">Metal Example</a>

- Documentation for metal shading language: <a
  href="https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf"
  class="reference external">Metal Specification</a>

- Using metal from C++: <a href="https://developer.apple.com/metal/cpp/"
  class="reference external">Metal-cpp</a>

</div>

Let’s keep the GPU kernel simple. We will launch exactly as many threads
as there are elements in the output. Each thread will pick the element
it needs from <span class="pre">`x`</span> and
<span class="pre">`y`</span>, do the point-wise operation, and update
its assigned element in the output.

<div class="highlight-C++ notranslate">

<div class="highlight">

    template <typename T>
    [[kernel]] void axpby_general(
            device const T* x [[buffer(0)]],
            device const T* y [[buffer(1)]],
            device T* out [[buffer(2)]],
            constant const float& alpha [[buffer(3)]],
            constant const float& beta [[buffer(4)]],
            constant const int* shape [[buffer(5)]],
            constant const int64_t* x_strides [[buffer(6)]],
            constant const int64_t* y_strides [[buffer(7)]],
            constant const int& ndim [[buffer(8)]],
            uint index [[thread_position_in_grid]]) {
        // Convert linear indices to offsets in array
        auto x_offset = elem_to_loc(index, shape, x_strides, ndim);
        auto y_offset = elem_to_loc(index, shape, y_strides, ndim);

        // Do the operation and update the output
        out[index] =
            static_cast<T>(alpha) * x[x_offset] + static_cast<T>(beta) * y[y_offset];
    }

</div>

</div>

We then need to instantiate this template for all floating point types
and give each instantiation a unique host name so we can identify it.

<div class="highlight-C++ notranslate">

<div class="highlight">

    instantiate_kernel("axpby_general_float32", axpby_general, float)
    instantiate_kernel("axpby_general_float16", axpby_general, float16_t)
    instantiate_kernel("axpby_general_bfloat16", axpby_general, bfloat16_t)
    instantiate_kernel("axpby_general_complex64", axpby_general, complex64_t)

</div>

</div>

The logic to determine the kernel, set the inputs, resolve the grid
dimensions, and dispatch to the GPU are contained in
<span class="pre">`Axpby::eval_gpu()`</span> as shown below.

<div class="highlight-C++ notranslate">

<div class="highlight">

    /** Evaluate primitive on GPU */
    void Axpby::eval_gpu(
      const std::vector<array>& inputs,
      std::vector<array>& outputs) {
        // Prepare inputs
        assert(inputs.size() == 2);
        auto& x = inputs[0];
        auto& y = inputs[1];
        auto& out = outputs[0];

        // Each primitive carries the stream it should execute on
        // and each stream carries its device identifiers
        auto& s = stream();
        // We get the needed metal device using the stream
        auto& d = metal::device(s.device);

        // Allocate output memory
        out.set_data(allocator::malloc(out.nbytes()));

        // Resolve name of kernel
        std::ostringstream kname;
        kname << "axpby_" << "general_" << type_to_name(out);

        // Make sure the metal library is available
        d.register_library("mlx_ext");

        // Make a kernel from this metal library
        auto kernel = d.get_kernel(kname.str(), "mlx_ext");

        // Prepare to encode kernel
        auto& compute_encoder = d.get_command_encoder(s.index);
        compute_encoder.set_compute_pipeline_state(kernel);

        // Kernel parameters are registered with buffer indices corresponding to
        // those in the kernel declaration at axpby.metal
        int ndim = out.ndim();
        size_t nelem = out.size();

        // Encode input arrays to kernel
        compute_encoder.set_input_array(x, 0);
        compute_encoder.set_input_array(y, 1);

        // Encode output arrays to kernel
        compute_encoder.set_output_array(out, 2);

        // Encode alpha and beta
        compute_encoder.set_bytes(alpha_, 3);
        compute_encoder.set_bytes(beta_, 4);

        // Encode shape, strides and ndim
        compute_encoder.set_vector_bytes(x.shape(), 5);
        compute_encoder.set_vector_bytes(x.strides(), 6);
        compute_encoder.set_bytes(y.strides(), 7);
        compute_encoder.set_bytes(ndim, 8);

        // We launch 1 thread for each input and make sure that the number of
        // threads in any given threadgroup is not higher than the max allowed
        size_t tgp_size = std::min(nelem, kernel->maxTotalThreadsPerThreadgroup());

        // Fix the 3D size of each threadgroup (in terms of threads)
        MTL::Size group_dims = MTL::Size(tgp_size, 1, 1);

        // Fix the 3D size of the launch grid (in terms of threads)
        MTL::Size grid_dims = MTL::Size(nelem, 1, 1);

        // Launch the grid with the given number of threads divided among
        // the given threadgroups
        compute_encoder.dispatch_threads(grid_dims, group_dims);
    }

</div>

</div>

We can now call the <span class="pre">`axpby()`</span> operation on both
the CPU and the GPU!

A few things to note about MLX and Metal before moving on. MLX keeps
track of the active <span class="pre">`command_buffer`</span> and the
<span class="pre">`MTLCommandBuffer`</span> to which it is associated.
We rely on <span class="pre">`d.get_command_encoder()`</span> to give us
the active metal compute command encoder instead of building a new one
and calling <span class="pre">`compute_encoder->end_encoding()`</span>
at the end. MLX adds kernels (compute pipelines) to the active command
buffer until some specified limit is hit or the command buffer needs to
be flushed for synchronization.

</div>

<div id="primitive-transforms" class="section">

### Primitive Transforms<a
href="https://ml-explore.github.io/mlx/build/html/#primitive-transforms"
class="headerlink" title="Link to this heading">#</a>

Next, let’s add implementations for transformations in a
<span class="pre">`Primitive`</span>. These transformations can be built
on top of other operations, including the one we just defined:

<div class="highlight-C++ notranslate">

<div class="highlight">

    /** The Jacobian-vector product. */
    std::vector<array> Axpby::jvp(
            const std::vector<array>& primals,
            const std::vector<array>& tangents,
            const std::vector<int>& argnums) {
        // Forward mode diff that pushes along the tangents
        // The jvp transform on the primitive can be built with ops
        // that are scheduled on the same stream as the primitive

        // If argnums = {0}, we only push along x in which case the
        // jvp is just the tangent scaled by alpha
        // Similarly, if argnums = {1}, the jvp is just the tangent
        // scaled by beta
        if (argnums.size() > 1) {
            auto scale = argnums[0] == 0 ? alpha_ : beta_;
            auto scale_arr = array(scale, tangents[0].dtype());
            return {multiply(scale_arr, tangents[0], stream())};
        }
        // If argnums = {0, 1}, we take contributions from both
        // which gives us jvp = tangent_x * alpha + tangent_y * beta
        else {
            return {axpby(tangents[0], tangents[1], alpha_, beta_, stream())};
        }
    }

</div>

</div>

<div class="highlight-C++ notranslate">

<div class="highlight">

    /** The vector-Jacobian product. */
    std::vector<array> Axpby::vjp(
            const std::vector<array>& primals,
            const std::vector<array>& cotangents,
            const std::vector<int>& argnums,
            const std::vector<int>& /* unused */) {
        // Reverse mode diff
        std::vector<array> vjps;
        for (auto arg : argnums) {
            auto scale = arg == 0 ? alpha_ : beta_;
            auto scale_arr = array(scale, cotangents[0].dtype());
            vjps.push_back(multiply(scale_arr, cotangents[0], stream()));
        }
        return vjps;
    }

</div>

</div>

Note, a transformation does not need to be fully defined to start using
the <span class="pre">`Primitive`</span>.

<div class="highlight-C++ notranslate">

<div class="highlight">

    /** Vectorize primitive along given axis */
    std::pair<std::vector<array>, std::vector<int>> Axpby::vmap(
            const std::vector<array>& inputs,
            const std::vector<int>& axes) {
        throw std::runtime_error("[Axpby] vmap not implemented.");
    }

</div>

</div>

</div>

</div>

<div id="building-and-binding" class="section">

## Building and Binding<a
href="https://ml-explore.github.io/mlx/build/html/#building-and-binding"
class="headerlink" title="Link to this heading">#</a>

Let’s look at the overall directory structure first.

<div class="line">

extensions

</div>

<div class="line">

├── axpby

</div>

<div class="line">

│ ├── axpby.cpp

</div>

<div class="line">

│ ├── axpby.h

</div>

<div class="line">

│ └── axpby.metal

</div>

<div class="line">

├── mlx_sample_extensions

</div>

<div class="line">

│ └── \_\_init\_\_.py

</div>

<div class="line">

├── bindings.cpp

</div>

<div class="line">

├── CMakeLists.txt

</div>

<div class="line">

└── setup.py

</div>

- <span class="pre">`extensions/axpby/`</span> defines the C++ extension
  library

- <span class="pre">`extensions/mlx_sample_extensions`</span> sets out
  the structure for the associated Python package

- <span class="pre">`extensions/bindings.cpp`</span> provides Python
  bindings for our operation

- <span class="pre">`extensions/CMakeLists.txt`</span> holds CMake rules
  to build the library and Python bindings

- <span class="pre">`extensions/setup.py`</span> holds the
  <span class="pre">`setuptools`</span> rules to build and install the
  Python package

<div id="binding-to-python" class="section">

### Binding to Python<a href="https://ml-explore.github.io/mlx/build/html/#binding-to-python"
class="headerlink" title="Link to this heading">#</a>

We use <a href="https://nanobind.readthedocs.io/en/latest/"
class="reference external">nanobind</a> to build a Python API for the
C++ library. Since bindings for components such as <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.html#mlx.core.array"
class="reference internal" title="mlx.core.array"><span
class="pre"><code
class="sourceCode python">mlx.core.array</code></span></a>, <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.stream.html#mlx.core.stream"
class="reference internal" title="mlx.core.stream"><span
class="pre"><code
class="sourceCode python">mlx.core.stream</code></span></a>, etc. are
already provided, adding our <span class="pre">`axpby()`</span> is
simple.

<div class="highlight-C++ notranslate">

<div class="highlight">

    NB_MODULE(_ext, m) {
         m.doc() = "Sample extension for MLX";

         m.def(
             "axpby",
             &axpby,
             "x"_a,
             "y"_a,
             "alpha"_a,
             "beta"_a,
             nb::kw_only(),
             "stream"_a = nb::none(),
             R"(
                 Scale and sum two vectors element-wise
                 ``z = alpha * x + beta * y``

                 Follows numpy style broadcasting between ``x`` and ``y``
                 Inputs are upcasted to floats if needed

                 Args:
                     x (array): Input array.
                     y (array): Input array.
                     alpha (float): Scaling factor for ``x``.
                     beta (float): Scaling factor for ``y``.

                 Returns:
                     array: ``alpha * x + beta * y``
             )");
     }

</div>

</div>

Most of the complexity in the above example comes from additional bells
and whistles such as the literal names and doc-strings.

<div class="admonition warning">

Warning

<span class="pre">`mlx.core`</span> must be imported before importing
<span class="pre">`mlx_sample_extensions`</span> as defined by the
nanobind module above to ensure that the casters for
<span class="pre">`mlx.core`</span> components like <a
href="https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.array.html#mlx.core.array"
class="reference internal" title="mlx.core.array"><span
class="pre"><code
class="sourceCode python">mlx.core.array</code></span></a> are
available.

</div>

</div>

<div id="building-with-cmake" class="section">

<span id="id1"></span>

### Building with CMake<a
href="https://ml-explore.github.io/mlx/build/html/#building-with-cmake"
class="headerlink" title="Link to this heading">#</a>

Building the C++ extension library only requires that you
<span class="pre">`find_package(MLX`</span>` `<span class="pre">`CONFIG)`</span>
and then link it to your library.

<div class="highlight-cmake notranslate">

<div class="highlight">

    # Add library
    add_library(mlx_ext)

    # Add sources
    target_sources(
        mlx_ext
        PUBLIC
        ${CMAKE_CURRENT_LIST_DIR}/axpby/axpby.cpp
    )

    # Add include headers
    target_include_directories(
        mlx_ext PUBLIC ${CMAKE_CURRENT_LIST_DIR}
    )

    # Link to mlx
    target_link_libraries(mlx_ext PUBLIC mlx)

</div>

</div>

We also need to build the attached Metal library. For convenience, we
provide a <span class="pre">`mlx_build_metallib()`</span> function that
builds a <span class="pre">`.metallib`</span> target given sources,
headers, destinations, etc. (defined in
<span class="pre">`cmake/extension.cmake`</span> and automatically
imported with MLX package).

Here is what that looks like in practice:

<div class="highlight-cmake notranslate">

<div class="highlight">

    # Build metallib
    if(MLX_BUILD_METAL)

    mlx_build_metallib(
        TARGET mlx_ext_metallib
        TITLE mlx_ext
        SOURCES ${CMAKE_CURRENT_LIST_DIR}/axpby/axpby.metal
        INCLUDE_DIRS ${PROJECT_SOURCE_DIR} ${MLX_INCLUDE_DIRS}
        OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}
    )

    add_dependencies(
        mlx_ext
        mlx_ext_metallib
    )

    endif()

</div>

</div>

Finally, we build the
<a href="https://nanobind.readthedocs.io/en/latest/"
class="reference external">nanobind</a> bindings

<div class="highlight-cmake notranslate">

<div class="highlight">

    nanobind_add_module(
      _ext
      NB_STATIC STABLE_ABI LTO NOMINSIZE
      NB_DOMAIN mlx
      ${CMAKE_CURRENT_LIST_DIR}/bindings.cpp
    )
    target_link_libraries(_ext PRIVATE mlx_ext)

    if(BUILD_SHARED_LIBS)
      target_link_options(_ext PRIVATE -Wl,-rpath,@loader_path)
    endif()

</div>

</div>

</div>

<div id="building-with-setuptools" class="section">

### Building with <span class="pre">`setuptools`</span><a
href="https://ml-explore.github.io/mlx/build/html/#building-with-setuptools"
class="headerlink" title="Link to this heading">#</a>

Once we have set out the CMake build rules as described above, we can
use the build utilities defined in
<span class="pre">`mlx.extension`</span>:

<div class="highlight-python notranslate">

<div class="highlight">

    from mlx import extension
    from setuptools import setup

    if __name__ == "__main__":
        setup(
            name="mlx_sample_extensions",
            version="0.0.0",
            description="Sample C++ and Metal extensions for MLX primitives.",
            ext_modules=[extension.CMakeExtension("mlx_sample_extensions._ext")],
            cmdclass={"build_ext": extension.CMakeBuild},
            packages=["mlx_sample_extensions"],
            package_data={"mlx_sample_extensions": ["*.so", "*.dylib", "*.metallib"]},
            extras_require={"dev":[]},
            zip_safe=False,
            python_requires=">=3.8",
        )

</div>

</div>

<div class="admonition note">

Note

We treat <span class="pre">`extensions/mlx_sample_extensions`</span> as
the package directory even though it only contains a
<span class="pre">`__init__.py`</span> to ensure the following:

- <span class="pre">`mlx.core`</span> must be imported before importing
  <span class="pre">`_ext`</span>

- The C++ extension library and the metal library are co-located with
  the python bindings and copied together if the package is installed

</div>

To build the package, first install the build dependencies with
<span class="pre">`pip`</span>` `<span class="pre">`install`</span>` `<span class="pre">`-r`</span>` `<span class="pre">`requirements.txt`</span>.
You can then build inplace for development using
<span class="pre">`python`</span>` `<span class="pre">`setup.py`</span>` `<span class="pre">`build_ext`</span>` `<span class="pre">`-j8`</span>` `<span class="pre">`--inplace`</span>
(in <span class="pre">`extensions/`</span>)

This results in the directory structure:

<div class="line">

extensions

</div>

<div class="line">

├── mlx_sample_extensions

</div>

<div class="line">

│ ├── \_\_init\_\_.py

</div>

<div class="line">

│ ├── libmlx_ext.dylib \# C++ extension library

</div>

<div class="line">

│ ├── mlx_ext.metallib \# Metal library

</div>

<div class="line">

│ └── \_ext.cpython-3x-darwin.so \# Python Binding

</div>

<div class="line">

…

</div>

When you try to install using the command
<span class="pre">`python`</span>` `<span class="pre">`-m`</span>` `<span class="pre">`pip`</span>` `<span class="pre">`install`</span>` `<span class="pre">`.`</span>
(in <span class="pre">`extensions/`</span>), the package will be
installed with the same structure as
<span class="pre">`extensions/mlx_sample_extensions`</span> and the C++
and Metal library will be copied along with the Python binding since
they are specified as <span class="pre">`package_data`</span>.

</div>

</div>

<div id="usage" class="section">

## Usage<a href="https://ml-explore.github.io/mlx/build/html/#usage"
class="headerlink" title="Link to this heading">#</a>

After installing the extension as described above, you should be able to
simply import the Python package and play with it as you would any other
MLX operation.

Let’s look at a simple script and its results:

<div class="highlight-python notranslate">

<div class="highlight">

    import mlx.core as mx
    from mlx_sample_extensions import axpby

    a = mx.ones((3, 4))
    b = mx.ones((3, 4))
    c = axpby(a, b, 4.0, 2.0, stream=mx.cpu)

    print(f"c shape: {c.shape}")
    print(f"c dtype: {c.dtype}")
    print(f"c is correct: {mx.all(c == 6.0).item()}")

</div>

</div>

Output:

<div class="highlight-python notranslate">

<div class="highlight">

    c shape: [3, 4]
    c dtype: float32
    c is correct: True

</div>

</div>

<div id="results" class="section">

### Results<a href="https://ml-explore.github.io/mlx/build/html/#results"
class="headerlink" title="Link to this heading">#</a>

Let’s run a quick benchmark and see how our new
<span class="pre">`axpby`</span> operation compares with the naive
<span class="pre">`simple_axpby()`</span> we first defined.

<div class="highlight-python notranslate">

<div class="highlight">

    import mlx.core as mx
    from mlx_sample_extensions import axpby
    import time

    def simple_axpby(x: mx.array, y: mx.array, alpha: float, beta: float) -> mx.array:
        return alpha * x + beta * y

    M = 4096
    N = 4096

    x = mx.random.normal((M, N))
    y = mx.random.normal((M, N))
    alpha = 4.0
    beta = 2.0

    mx.eval(x, y)

    def bench(f):
        # Warm up
        for i in range(5):
            z = f(x, y, alpha, beta)
            mx.eval(z)

        # Timed run
        s = time.time()
        for i in range(100):
            z = f(x, y, alpha, beta)
            mx.eval(z)
        e = time.time()
        return 1000 * (e - s) / 100

    simple_time = bench(simple_axpby)
    custom_time = bench(axpby)

    print(f"Simple axpby: {simple_time:.3f} ms | Custom axpby: {custom_time:.3f} ms")

</div>

</div>

The results are
<span class="pre">`Simple`</span>` `<span class="pre">`axpby:`</span>` `<span class="pre">`1.559`</span>` `<span class="pre">`ms`</span>` `<span class="pre">`|`</span>` `<span class="pre">`Custom`</span>` `<span class="pre">`axpby:`</span>` `<span class="pre">`0.774`</span>` `<span class="pre">`ms`</span>.
We see modest improvements right away!

This operation is now good to be used to build other operations, in <a
href="https://ml-explore.github.io/mlx/build/html/python/nn/module.html#mlx.nn.Module"
class="reference internal" title="mlx.nn.Module"><span class="pre"><code
class="sourceCode python">mlx.nn.Module</code></span></a> calls, and
also as a part of graph transformations like
<span class="pre">`grad()`</span>.

</div>

</div>

<div id="scripts" class="section">

## Scripts<a href="https://ml-explore.github.io/mlx/build/html/#scripts"
class="headerlink" title="Link to this heading">#</a>

<div class="admonition-download-the-code admonition">

Download the code

The full example code is available in <a
href="https://github.com/ml-explore/mlx/tree/main/examples/extensions/"
class="reference external">mlx</a>.

</div>

</div>

</div>

<div class="prev-next-area">

<a href="https://ml-explore.github.io/mlx/build/html/cpp/ops.html"
class="left-prev" title="previous page"><em></em></a>

<div class="prev-next-info">

previous

Operations

</div>

<a
href="https://ml-explore.github.io/mlx/build/html/dev/metal_debugger.html"
class="right-next" title="next page"></a>

<div class="prev-next-info">

next

Metal Debugger

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
  href="https://ml-explore.github.io/mlx/build/html/#introducing-the-example"
  class="reference internal nav-link">Introducing the Example</a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#operations-and-primitives"
  class="reference internal nav-link">Operations and Primitives</a>
  - <a href="https://ml-explore.github.io/mlx/build/html/#operations"
    class="reference internal nav-link">Operations</a>
  - <a href="https://ml-explore.github.io/mlx/build/html/#primitives"
    class="reference internal nav-link">Primitives</a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/#using-the-primitive"
    class="reference internal nav-link">Using the Primitive</a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#implementing-the-primitive"
  class="reference internal nav-link">Implementing the Primitive</a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/#implementing-the-cpu-back-end"
    class="reference internal nav-link">Implementing the CPU Back-end</a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/#implementing-the-gpu-back-end"
    class="reference internal nav-link">Implementing the GPU Back-end</a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/#primitive-transforms"
    class="reference internal nav-link">Primitive Transforms</a>
- <a
  href="https://ml-explore.github.io/mlx/build/html/#building-and-binding"
  class="reference internal nav-link">Building and Binding</a>
  - <a href="https://ml-explore.github.io/mlx/build/html/#binding-to-python"
    class="reference internal nav-link">Binding to Python</a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/#building-with-cmake"
    class="reference internal nav-link">Building with CMake</a>
  - <a
    href="https://ml-explore.github.io/mlx/build/html/#building-with-setuptools"
    class="reference internal nav-link">Building with <span
    class="pre"><code
    class="docutils literal notranslate">setuptools</code></span></a>
- <a href="https://ml-explore.github.io/mlx/build/html/#usage"
  class="reference internal nav-link">Usage</a>
  - <a href="https://ml-explore.github.io/mlx/build/html/#results"
    class="reference internal nav-link">Results</a>
- <a href="https://ml-explore.github.io/mlx/build/html/#scripts"
  class="reference internal nav-link">Scripts</a>

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
