# MLX Curated Documentation (Human‑Authored)

This folder contains a readable, practical rewrite of key MLX docs. It focuses on clarity, examples, and day‑to‑day workflows while preserving the core facts of the API.

Contents:

- Getting Started: install, first arrays, devices, performance tips
- Arrays: creation, dtype/shape, indexing, math, reductions, random
- Neural Networks: `mlx.nn` Modules, layers, activations, training loop
- Optimizers: `mlx.optimizers` usage and configuration
- Devices & Streams: CPUs/GPUs, default device, streams, sync
- Saving & Loading: arrays, tensors, weights, formats
- Linear Algebra: solve, SVD, Cholesky, eigen, tips
- FFT: 1D/2D/N‑D FFTs and common patterns
- Distributed: multi‑process basics, collectives, send/recv
- Porting from PyTorch: mental model, code patterns, RNG, devices
- Shapes & Broadcasting: explicit shape discipline to avoid confusion
- Common Pitfalls: sharp edges and how to avoid them
 - Training Patterns: loops, eval, accumulation, checkpoints
 - Conv and Shapes: NCHW, output math, depthwise/groups
- Optimizers Deep Dive: state, schedules, troubleshooting
- PyTorch Dissonance: one‑page checklist of key differences
- In‑Place vs Functional: how MLX avoids hidden mutation
- NumPy Users: quick mapping and gotchas
- Backend Support: CPU vs GPU notes and fallbacks
- Module Primer: MLX Module vs PyTorch Module in practice
- Liquid Autoencoder Snippets: end‑to‑end patterns (compile, attention, trees)
 - Custom Metal Kernels: MX Fast patterns and VJP tips
- HPC16x8 Extended Precision: limb‑based double‑double pattern
- The Apple Way (TL;DR): devices, immutability, RNG, lazy eval

If you prefer a single entry point, start with Getting Started.
