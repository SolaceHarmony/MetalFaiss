WWDC16-Inspired Optimization Patterns For MLX + Metal

This document adapts Apple’s WWDC16 “Advanced Metal Shader Optimization” to MLX `mx.fast.metal_kernel`. It covers address spaces, compute kernel organization, barrier scope, data types, arithmetic, control flow, memory access, occupancy, and how we apply them in linear algebra kernels.

Highlights

- Address spaces: keep small, read‑only params (shape/flags/eps) in tiny buffers and load once into registers at kernel start. MLX doesn’t expose constant bindings directly; emulate by minimizing repeated loads.
- Kernel organization: do enough work per thread (e.g., 2 outputs) if register usage allows; split across kernels for multi‑phase algorithms (no global barrier in MSL).
- Barriers: use the smallest scope; prefer TG barriers; consider simdgroup barriers for warp‑sized groups.
- Data types: evaluate `half` for intermediates (accumulate in `float`), use `ushort`/`int` appropriately; avoid mixing float literals in half math.
- Arithmetic: rely on fast‑math and `fma(a,b,c)`; avoid runtime integer division in hot loops.
- Control flow: favor ternary/select over “multiply by mask”; keep divergence out of the hot loops.
- Memory access: coalesce; stage tiles; avoid dynamic stack indexing; use `int` addressing.
- Occupancy: watch register/TG usage; measure medians; prune losers.

Examples

- GEMM tiling (A@V; Aᵀ@B): 16×16 TG tiles, shared memory staging, `fma` inner loops, int indices.
- QR helpers: separate dot (c = Qᵀv) and update (v ← v − Qc) kernels; `fma` in the inner accumulation; two‑pass MGS retains stability.
- SVD Z‑step: two GEMM‑like kernels preferred over monolithic; banded processing reduces peak; streams may help only at large sizes.

See also

- Comprehensive‑MLX‑Metal‑Guide.md, Metal‑Primer.md, Tiled‑LinearAlgebra.md, Autoswitching‑Design.md, Benchmarks‑and‑Pruning.md

Applied In Practice (MetalFaiss)

- Fused multiply‑add in hot loops:
  - Added explicit `fma(a,b,acc)` in GEMM tiles and QR update to leverage the hardware FMA path.
  - Files: `python/metalfaiss/faissmlx/kernels/gemm_kernels.py`, `python/metalfaiss/faissmlx/kernels/qr_kernels.py`.

- Integer addressing:
  - Switched shape/index math to `int` inside kernels to avoid extra instructions from unsigned arithmetic.
  - Files: same as above.

- Two‑phase kernels (no global barrier):
  - SVD Z‑step implemented as two tiled kernels (`A@V` then `Aᵀ@B`) instead of a monolith; easier to tune and aligns with barrier guidance.
  - File: `python/metalfaiss/faissmlx/kernels/gemm_kernels.py`; orchestrated in `python/metalfaiss/faissmlx/svd.py`.

- QR helper split:
  - Separate dot and update kernels for MGS (`c = Qᵀv`, `v ← v − Qc`); numerically stable with two re‑orth steps.
  - Files: `python/metalfaiss/faissmlx/kernels/qr_kernels.py`, `python/metalfaiss/faissmlx/qr.py`.

- Autoswitch thresholds by measurement:
  - Prefer MLX/compiled MLX at small/medium sizes; switch to kernels for larger work (current heuristic ~16M MACs/iter).
  - File: `python/metalfaiss/faissmlx/dispatch.py`.

- Variant test harnesses (to guide tuning):
  - QR: MLX vs kernel‑assisted with orthonormality + reconstruction checks.
  - SVD: MLX, MLX+compile, kernel mono/banded/streams with residual checks; prints timings.
  - Files: `python/metalfaiss/unittest/test_qr_variants.py`, `python/metalfaiss/unittest/test_svd_variants.py`.

- Extended precision scaffolding (guarded):
  - Added MLX‑only compensated helpers (`kahan_sum`, `safe_norm2`) for numerically hard cases; to be wired under autoswitch as needed.
  - File: `python/metalfaiss/faissmlx/hpc16x8.py`.
