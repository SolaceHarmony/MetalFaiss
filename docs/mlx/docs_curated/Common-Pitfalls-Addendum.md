Common Pitfalls (Addendum)

Real errors and symptoms we fixed along the way.

1) “expected expression” during kernel compile

Cause: Included full Metal function signatures in `source`. Fix: keep `source` body-only; move includes to `header`.

2) Duplicate include/module errors

Cause: Includes placed both in body and header or fed per-call templates. Fix: includes only in `header`, compile once, reuse.

3) NumPy mocks hide bugs

CPU code paths with NumPy won’t exercise GPU kernels or MLX shapes/dtypes, masking errors. Fix: stay MLX-only for GPU paths; remove mocks.

4) JIT thrash

Compiling kernels per call with string-templated shapes. Fix: pass shapes via small buffers; build kernel once and reuse.

5) Dtype drift

Silent casts between fp16/fp32 can destabilize QR/SVD. Fix: standardize on fp32; add HPC16x8 limbs for sensitive accumulations.

References

- Curated: `COMMON_PITFALLS.md`
- This repo: `docs/mlx/Kernel-Guide.md` and `docs/research/Journal.md`

