# Metal Shader Optimization Tips

This document summarizes key optimization strategies for Metal shaders, based on best practices and hardware architecture considerations.

## 1. Shader Performance Fundamentals

### 1.1. Address Spaces

Choosing the correct address space for your buffer arguments is crucial for performance. GPUs have multiple memory paths, and selecting the right one for the access pattern is key.

*   **`device` memory:**
    *   Read-write.
    *   No size restrictions.
    *   Flexible alignment.
    *   Use for data that is large, variable in size, or written to by the shader.
    *   **Example:** A variable number of vertex positions.
      ```cpp
      vertex float4 simpleVertex(uint vid [[ vertex_id ]],
                                const device float4 *positions [[ buffer(0) ]])
      {
          return positions[vid];
      }
      ```

*   **`constant` memory:**
    *   Read-only.
    *   Limited in size.
    *   Has alignment restrictions.
    *   Optimized for high-reuse, where many threads access the same data. This allows the GPU to cache it effectively.
    *   **Example:** A single projection matrix used by all vertices.
      ```cpp
      vertex float4 transformedVertex(uint vid [[ vertex_id ]],
                                      const device float4 *positions [[ buffer(0) ]],
                                      constant matrix_float4x4 &transform [[ buffer(1) ]])
      {
          return transform * positions[vid];
      }
      ```

**Rule of thumb:** Start with `device`. If the data is fixed-size and read many times by a group of threads, move it to `constant`.

### 1.2. Buffer Preloading

The GPU has dedicated hardware to preload buffers, which can significantly improve performance by hiding memory latency.

*   **Constant Buffer Preloading:**
    *   To enable preloading, accesses must be statically bound (the compiler must know the size and access range).
    *   Pass single struct arguments by reference (`constant MyData &myData`).
    *   Use bounded arrays within structs instead of raw pointers. This is a common and effective pattern.
    *   **Example (Incorrect):** Using raw pointers prevents the compiler from knowing the data size, disabling preloading.
      ```cpp
      fragment float4 litFragment(const device Light *l [[ buffer(0) ]],
                                 const device uint *count [[ buffer(1) ]],
                                 LitVertex vertex [[ stage_in ]]);
      ```
    *   **Example (Correct):** A struct with a bounded array allows for preloading.
      ```cpp
      typedef struct {
          uint count;
          Light data[MAX_LIGHTS];
      } LightData;

      fragment float4 litFragment(constant LightData &lights [[ buffer(0) ]],
                                 LitVertex vertex [[ stage_in ]]);
      ```

*   **Vertex Buffer Preloading:**
    *   The fixed-function vertex fetcher uses dedicated hardware.
    *   Ensure buffer access is indexed directly by `[[vertex_id]]` or `[[instance_id]]`.
    *   Use vertex descriptors (`MTLVertexDescriptor`) where possible to formalize your data layout.

Why it matters (teacher’s note)
- Preloading lets dedicated units hide memory latency for data the compiler knows is bounded and repeatedly accessed. By putting small, reused inputs into `constant` structs and indexing vertex buffers via the built-in IDs, you let hardware fetchers and caches do their job.
- Exercise: Switch a fragment parameter from `const device T*` + separate `count` to a bounded `constant struct`. Compare GPU time and Xcode counters (buffer read transactions).

### 1.3. Fragment Function Resource Writes

*   Writing to resources (buffers, textures) from a fragment shader can partially disable **hidden surface removal**, a key optimization where the GPU avoids shading pixels that aren't visible. (This capability was introduced around the WWDC16 timeframe.)
*   A fragment performing a write cannot be occluded by later fragments.
*   To mitigate this, use the `[[early_fragment_tests]]` attribute in your fragment function signature. This allows the depth/stencil test to reject fragments *before* they execute, which can restore some of the lost performance.
*   **Rendering Order:** To maximize early rejection, draw objects with fragment writes **after** opaque objects. If they also update depth, sort them front-to-back.

Why it matters (teacher’s note)
- Hidden surface removal saves a lot of fragment work by skipping occluded pixels. A fragment that writes can’t be trivially discarded later because the write may have side effects. `[[early_fragment_tests]]` restores depth/stencil rejection before execution, rescuing most of the lost benefit when your test rejects.
- Exercise: Render two overlapping quads where the top quad writes to a storage texture. Measure GPU time with and without `[[early_fragment_tests]]` and when drawing the writer before/after opaque geometry.

### 1.4. Compute Kernel Organization

*   **Amortize Launch Overhead:** Each thread has a launch cost. Process multiple work items per compute thread to reduce this overhead. Reuse values between work items where possible.
    *   **Example:** In an image filter like Sobel, a single thread can process multiple adjacent pixels, reusing texture reads.
      ```cpp
      // Initial: 1 thread = 1 pixel
      kernel void sobel_1_1(ushort2 tid [[ thread_position_in_grid ]]) { ... }

      // Better: 1 thread = 2 pixels, reusing texture samples
      kernel void sobel_2_1(ushort2 tid [[ thread_position_in_grid ]]) {
          // process pixel at tid.x * 2
          // ...
          // reuse some texture data and process pixel at tid.x * 2 + 1
      }
      ```
*   **Barriers:** Use barriers with the smallest possible scope.
    *   `simdgroup_barrier` is faster than `threadgroup_barrier` for threadgroups smaller than or equal to the SIMD group size. A `threadgroup_barrier` is not needed for SIMD-width threadgroups, making it a more efficient choice when applicable.

Teaching tip
- Use `simdgroup_barrier` only when communication stays within a single SIMD group and you aren’t relying on threadgroup memory written by other SIMD groups. If any data crosses SIMD-group boundaries (typical for 16×16 tiles), use `threadgroup_barrier`.

## 2. Tuning Shader Code

### 2.1. Data Types

*   A8 and later GPUs use 16-bit register units.
*   **Use the smallest possible data type.** `half` and `short` are faster and use fewer registers, improving thread occupancy.
*   Energy efficiency: `half` < `float` < `short` < `int`.
*   Use `half` for texture reads, interpolants, and general math where its precision (`6.1 x 10^-5` to `65504`) is sufficient.
    *   **Warning:** Be mindful of the limitations. For example, writing `65535` as a `half` will result in infinity.
*   Avoid `float` literals (e.g., `2.0`) in `half` precision operations; use `half` literals (`2.0h`).
    ```cpp
    // Bad: promotes a and b to float
    half foo(half a, half b) { return clamp(a, b, -2.0, 5.0); }

    // Good: uses half-precision literals
    half foo(half a, half b) { return clamp(a, b, -2.0h, 5.0h); }
    ```
*   `char` is not natively supported for arithmetic and may result in extra instructions.

Pattern: half for I/O, float to accumulate
- Read/store in `half` to reduce bandwidth, but accumulate in `float` for stability. Convert once at the boundary.
  ```cpp
  // Read inputs as half, widen to float for math
  half ha = tex.sample(s, uv).x;
  float a = float(ha);
  float acc = fma(a, b, acc);
  // ... after reduction
  out[i] = half(acc); // only if the result tolerates half
  ```

### 2.2. Arithmetic

*   **Built-ins:** Use built-in functions like `abs()`, `saturate()`, and `fma()` (fused multiply-add). Modifiers like negate (`-x`) are often free.
*   **Scalar Architecture:** A8 and later GPUs are scalar. The compiler splits vector operations. Do not waste time manually vectorizing code that isn't naturally vector-based.
*   **Ternary Operators:** The `select` instruction (ternary operator `? :`) is very fast on A8 and later GPUs. Prefer it over creative solutions like multiplying by 0 or 1.
    ```cpp
    // Slow: fakes a ternary op
    if (foo) m = 0.0h; else m = 1.0h;
    half p = v * m;

    // Fast: uses native ternary op
    half p = foo ? 0.0h : v; // Note: logic inverted from source for clarity
    ```
*   **Integer Division:** Avoid division or modulus by denominators that are not known at compile time. This can be extremely slow.
    ```cpp
    // Extremely slow: constInputs.width is not a compile-time constant
    int onPos0 = vertexIn[vertex_id] / constInputs.width;

    // Fast: 256 is a literal
    int onPos1 = vertexIn[vertex_id] / 256;

    // Fast: 'width' is a function constant, known at compile time
    constant int width [[ function_constant(0) ]];
    int onPos2 = vertexIn[vertex_id] / width;
    ```
*   **Fast Math:** Enabled by default in Metal. It provides significant performance gains by using faster, less precise built-ins. If you must disable it, use `fma()` to regain some performance, as disabling fast-math prohibits many compiler optimizations.

Teaching note: what fast‑math changes
- Fast‑math relaxes IEEE constraints, enabling instruction selection and reassociation that often fuses ops (`fma`) and simplifies divisions. If you must disable it for numerical reasons, isolate those regions and keep the rest of the kernel under fast‑math.

### 2.3. Control Flow

*   Control flow that is uniform across a SIMD group is fast.
*   Divergence (when threads in a SIMD group take different paths) is expensive as both paths must be executed.
*   Avoid `switch` statement fall-throughs, as they can create unstructured control flow and lead to code duplication.

### 2.4. Memory Access

*   **Stack Access:** Avoid dynamically indexed stack arrays where the array itself is not a compile-time constant. The performance cost can be "catastrophic," with one real-world app seeing a 30% slowdown from a single 32-byte array.
    ```cpp
    // Bad: 'tmp' is not a compile-time constant array, and 'c' is a dynamic index.
    // This can have a catastrophic performance impact.
    int foo(int a, int b, int c) {
        int tmp[2] = { a, b };
        return tmp[c]; // 'c' makes this a dynamic index
    }
    ```

*   **Loads and Stores:**
    *   One large vector load/store is faster than multiple scalar ones.
    *   Arrange data in structs to be adjacent to allow the compiler to vectorize loads.
    *   Use `int` or smaller types for device memory addressing, not `uint`.
    
    Before/after: struct layout for vectorized loads
    ```cpp
    // Bad: forces two scalar loads (a and c are far apart)
    struct Foo { float a; float b[7]; float c; };
    float sum_mul(device const Foo* x, uint n) {
        float s = 0;
        for (uint i = 0; i < n; ++i) s += x[i].a * x[i].c;
        return s;
    }
    // Better: pack to enable a single vector load
    struct Foo2 { float2 a; float b[7]; };
    float sum_mul2(device const Foo2* x, uint n) {
        float s = 0;
        for (uint i = 0; i < n; ++i) s += x[i].a.x * x[i].a.y;
        return s;
    }
    ```
    
    Measure it
    - Use Xcode GPU capture to compare memory transactions and shader time for the two layouts on the same workload. Expect fewer scalar loads and improved throughput for the packed layout.
*   **Latency Hiding:**
    *   GPUs hide memory latency by switching to other threads. Higher register usage reduces the number of active threads (occupancy), making the shader more sensitive to latency.
    *   Initiate long-latency operations (like texture samples) as early as possible. This allows the GPU to issue the memory requests and switch to other threads, hiding the latency. Avoid structuring code in a way that creates unnecessary dependencies between these operations.
    ```cpp
    // BAD: The fetch for 'b' is dependent on the 'if' condition. The GPU must
    // wait for the result of 'a' before it can even start fetching 'b'.
    // This serializes the two texture fetches and creates a long dependency chain.
    half a = tex0.sample(s0, c0);
    half res = 0.0h;
    if (a >= 0.0h) {
        half b = tex1.sample(s1, c1); // Stalls until 'a' is ready
        res = a * b;
    }
    // GOOD: 'a' and 'b' are fetched independently at the start.
    // The GPU can issue both sample operations in parallel.
    half a = tex0.sample(s0, c0);
    half b = tex1.sample(s1, c1);
    half res = 0.0h;
    if (a >= 0.0h) {
        res = a * b; // The calculation waits for both, but they were fetched in parallel.
    }
    ```
