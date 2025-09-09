Comprehensive Guide to Using Metal in MLX (2025 Addendum)

This guide consolidates working, production‑tested patterns for MLX’s `mx.fast.metal_kernel` on Apple GPUs. It complements existing pages by showing the exact MLX API contracts that compile and run, with kernel sizing, parameter passing, reductions, and integration advice. See also: MetalKernel-Patterns.md and Tiled-LinearAlgebra.md.

1) Kernel Contract (Header vs Body)

- header: `#include <metal_stdlib>` + `using namespace metal;` and any inline helpers (branchless guards, reductions). Avoid duplicating includes in `source`.
- source: body‑only statements — do not write a kernel signature; MLX generates it and binds buffers by `input_names`/`output_names`.
- Build kernels once and reuse the handle; pass shapes/flags/eps as small MLX buffers (uint32/float32) to avoid recompiling.

2) Binding and Launch

```python
kernel = mx.fast.metal_kernel(
    name="gemm_av",
    input_names=["A","V","shape"],  # shape=[m,n,k] (uint32)
    output_names=["C"],                # C=(m,k)
    header="#include <metal_stdlib>\nusing namespace metal;\n",
    source=r"""
        const uint TM=16, TN=16, TK=16;
        threadgroup float Asub[TM][TN];
        threadgroup float Vsub[TN][TK];
        uint m=shape[0], n=shape[1], k=shape[2];
        uint lx=thread_position_in_threadgroup.x; // 0..TK-1
        uint ly=thread_position_in_threadgroup.y; // 0..TM-1
        uint bx=threadgroup_position_in_grid.x;   // tile in k
        uint by=threadgroup_position_in_grid.y;   // tile in m
        uint row = by*TM + ly, col = bx*TK + lx;
        float acc=0.0f; uint tiles=(n+TN-1)/TN;
        for (uint t=0;t<tiles;++t){
           uint a_col=t*TN+lx; uint v_row=t*TN+ly;
           Asub[ly][lx]= (row<m && a_col<n)? A[row*n+a_col]:0.0f;
           Vsub[ly][lx]= (v_row<n && col<k)? V[v_row*k+col]:0.0f;
           threadgroup_barrier(mem_flags::mem_threadgroup);
           for (uint p=0;p<TN;++p) acc += Asub[ly][p]*Vsub[p][lx];
           threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (row<m && col<k) C[row*k+col]=acc;
    """,
    ensure_row_contiguous=True,
)
m,n,k = map(int, A.shape+(V.shape[1],))
shape = mx.array([m,n,k], dtype=mx.uint32)
grid  = ((k+15)//16, (m+15)//16, 1)
tg    = (16,16,1)
(C,)  = kernel(inputs=[A,V,shape], output_shapes=[(m,k)], output_dtypes=[A.dtype], grid=grid, threadgroup=tg)
```

3) Sizing, Streams, and Barriers

- Threadgroup sizes ≤ 1024; align x/y to 32 (Apple execution width). Try 16×16, 32×8, or 8×32.
- Synchronize shared memory with `threadgroup_barrier(mem_flags::mem_threadgroup)`; for multi‑phase global writes, use `mem_device` barriers.
- No global barrier exists; structure multi‑phase algorithms as multiple kernels (e.g., dot then update).
- Use `mx.new_stream(device)` + `with mx.stream(s): ...` to fan‑out independent tiles; call `mx.synchronize()` to join. Measure memory before enabling many streams.

4) Branchless Guards (the “where” analogue in MSL)

- Avoid warp divergence in hot loops:
  - `x = (fabs(x) < eps) ? 0.0f : x;`
  - `x = fmin(fmax(x, -cap), cap);`
- Toggle behavior via a small flags buffer (uint32 bit mask) rather than JIT templating.

5) QR and SVD Patterns

- QR (Modified Gram–Schmidt):
  - `c = Q^T v` (column‑parallel dot) and `v ← v − Q c` (row‑parallel update) as two kernels; keep numerics stable with two re‑orth steps.
  - Per‑column norms can use reductions (simd_sum + TG scratch) when needed.
- SVD (subspace iteration):
  - Prefer two GEMM‑like kernels: `B=A@V`, `Z=Aᵀ@B`. Tile and coalesce; avoid a giant monolithic kernel.
  - Banded processing (split k) can reduce peak memory; multi‑stream dispatch may help only at larger sizes.

6) Compile and MLX Graphs

- `mx.compile` fuses pure‑MLX graphs (e.g., MLX matmul path for SVD); shapes must be stable.
- Compiling does not change kernel internals, but a compiled wrapper can reduce Python/graph overhead when driving many kernel calls.

7) Debug and Diagnostics (Optional)

- Add tiny `dbg` buffers (float) under a debug gate to record grid/tg sizes and early exits; keep off hot paths by default.
- Use `mlx.core.metal.start_capture()` / `stop_capture()` around a single iteration to inspect scheduling in Xcode.

8) Pitfalls and Best Practices

- Don’t rebuild kernels per call; pass parameters via buffers.
- Keep host scalar pulls (`.numpy()`, `.item()`, `float(int)`) out of hot paths; they force sync to CPU.
- Mind Metal’s argument count limits; pack many scalars into a small array.

References

- See also: MetalKernel-Patterns.md, Tiled-LinearAlgebra.md, Autoswitching-Design.md, Common-Pitfalls-Addendum.md
- Working examples in this repo (QR/SVD kernels) mirror these patterns.

