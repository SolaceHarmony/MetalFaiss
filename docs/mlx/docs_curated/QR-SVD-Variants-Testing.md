QR & SVD Variants: Testing and Benchmarking Patterns

Purpose

- Incubate multiple implementations (MLX vs custom kernels; banded/streamed variants) while keeping correctness tight and observing performance.

SVD Variant Harness

- Variants:
  - MLX GEMM path
  - MLX + compile (optional)
  - Kernel tiled (monolithic)
  - Kernel banded (serial; fixed band size)
  - Kernel banded + streams (S streams)
- Residual ratio for correctness:
  - r = ||A V − U S||_F / ||A||_F ≤ 0.6 (loose enough for subspace iteration)
- Warmup and medians recommended for timings; print results, don’t assert perf.

QR Variant Harness

- Compare MLX two‑pass MGS vs kernel‑assisted MGS (projection + update kernels).
- Checks:
  - Orthonormality: Qkᵀ Qk ≈ I (e.g., 1e‑3 tolerance)
  - Reconstruction: ||Q R − A||_F / ||A||_F ≤ 1e‑2

Recommended Shapes

- Small: (256×128, k=16)
- Medium: (1024×256, k=32)
- Large: (2048×512, k=64)

Autoswitch Tuning

- Use medians from the suite to decide thresholds (device + size aware).
- Prefer MLX for small/medium; switch to kernels at larger work sizes.

References

- See also: Autoswitching-Design.md and Benchmarks-and-Pruning.md

