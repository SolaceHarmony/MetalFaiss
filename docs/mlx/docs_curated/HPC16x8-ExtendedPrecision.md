HPC16x8: Extended Precision Without fp64 on GPU

Motivation

Apple GPUs don’t offer fast fp64. Accumulating long dot products and norms in fp32 can drift for ill-conditioned inputs. HPC16x8 represents a sum as 8 limbs × 16 bits each, accumulating integer partials and carrying at the end.

Pattern

- Split each product `a*b` into high/low halves, accumulate into 8 uint limbs.
- Reduce limbs across a threadgroup, then carry-propagate from limb 0→7.
- Convert back to float by radix expansion.

Use cases

- QR projections `qᵀv` and norms `vᵀv` for tough columns.
- Gram updates in block algorithms.

Sketch (conceptual)

```c
// Per-thread accumulators
uint limbs[8] = {0};
for (uint i = start; i < end; ++i) {
  float p = a[i] * b[i];
  // reinterpret as fixed-point, accumulate across limbs...
}
// reduce limbs across the warp/tg, then carry-propagate
```

References

- Curated doc: `HPC16x8.md`
- Ember ML: limb-accumulating dot/norm helpers used in QR/SVD/eigen paths.

