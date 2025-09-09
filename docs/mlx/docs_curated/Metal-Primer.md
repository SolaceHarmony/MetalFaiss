Metal Primer (Apple GPUs) for MLX Kernel Authors

This primer covers the minimal MSL concepts you need to implement fast kernels for MLX.

Indices and Sizes

- Global/threadgroup/local indices:
  - `thread_position_in_grid` (global thread id)
  - `threadgroup_position_in_grid` (block index)
  - `thread_position_in_threadgroup` (local index inside block)
- Sizes:
  - `threads_per_threadgroup`, `grid_size` — helpful for reductions/circuit breakers.

Shared Memory and Barriers

- Stage tiles in `threadgroup` arrays; insert `threadgroup_barrier(mem_flags::mem_threadgroup)` between load/compute phases.
- For multi‑phase global writes/reads, use `threadgroup_barrier(mem_flags::mem_device)`.

SIMD Reductions

- Apple execution width ~32 lanes; use `simd_sum(x)`, `simd_max(x)`.
- Write one partial per warp to a `threadgroup` scratch array; have thread 0 combine and broadcast.

Branchless Guards

- Prefer clamp/ternary to avoid divergence:
  - `x = fmin(fmax(x, -cap), cap);`
  - `x = (fabs(x) < eps) ? 0.0f : x;`
- Pass flags via a small `uint32` buffer and compute booleans: `(flags & 1u) != 0`.

Sizing Kernels

- ≤ 1024 threads per threadgroup; align x/y dims to 32.
- Defaults: 16×16 (256 threads) for GEMM‑like; try 32×8 or 8×32 per device.

No Global Barrier

- If you need a two‑phase algorithm, split into two kernels or keep within a single threadgroup.

Parameter Passing in MLX

- Avoid templated recompiles at call time; pass small parameter arrays:
  - `shape=[m,n,k] (uint32)`, `flags=[bits] (uint32)`, `eps=[1e-6] (float32)`
- Bind as inputs; read inside the kernel.

Streams

- Use `s = mx.new_stream(mx.default_device())` and `with mx.stream(s): ...` to queue work; `mx.synchronize()` to join.
- Be mindful of peak memory when launching many bands concurrently.

References

- See: Comprehensive-MLX-Metal-Guide.md, MetalKernel-Patterns.md

