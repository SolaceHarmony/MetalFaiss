"""
Autoswitching helpers for MLX vs. Metal‑kernel implementations

Purpose
- Choose between MLX vectorized ops and custom Metal kernels based on device
  and problem size. Prefer MLX for small/medium shapes; use kernels for larger
  tiles. Allow env overrides for experimentation and benchmarking.

Environment Overrides
- `METALFAISS_FORCE_QR=MLX|KERNEL`
- `METALFAISS_FORCE_SVD=MLX|KERNEL`

References
- docs/mlx/Comprehensive-MLX-Metal-Guide.md:1
- docs/research/Journal.md:1
"""

from __future__ import annotations
from typing import Literal, Dict
import os
import mlx.core as mx
from .flags import qr_kernels_enabled, svd_kernels_enabled


def _device_type() -> str:
    # Best-effort device detection. Keep simple and robust.
    try:
        # If MLX exposes metal module, we assume GPU is available
        import mlx.metal as _mx_metal  # type: ignore
        return "gpu"
    except Exception:
        return "cpu"


def choose_qr_impl(m: int, k: int) -> Literal["MLX_MGS", "KERNEL_PROJ"]:
    """Select QR projection path.

    Heuristic
    - On GPU, choose kernel projections when `m*k` is large (long vectors and/or
      many columns). Else, MLX ops are competitive and simpler.

    Env override
    - `METALFAISS_FORCE_QR=MLX|KERNEL` to force a choice.
    """
    # Env overrides
    if os.environ.get("METALFAISS_FORCE_QR", "").upper() == "MLX":
        return "MLX_MGS"
    if os.environ.get("METALFAISS_FORCE_QR", "").upper() == "KERNEL":
        return "KERNEL_PROJ"

    # Global gating: keep research kernels off unless explicitly enabled
    if not qr_kernels_enabled():
        return "MLX_MGS"
    dev = _device_type()
    # Threshold heuristic: when there are many columns or long vectors,
    # using a kernel for c = Q^T v becomes beneficial.
    if dev == "gpu" and (m * k) >= (256 * 1024):  # ~256k elements
        return "KERNEL_PROJ"
    return "MLX_MGS"


def choose_svd_impl(m: int, n: int, k: int) -> Literal["MLX_MATMUL", "KERNEL_TILED"]:
    """Select SVD Z‑step implementation.

    Heuristic
    - Prefer MLX for small/medium sizes; switch to tiled kernels for large work
      (roughly `m*n*k >= 16M`).

    Env override
    - `METALFAISS_FORCE_SVD=MLX|KERNEL` to force a choice.
    """
    # Env overrides
    if os.environ.get("METALFAISS_FORCE_SVD", "").upper() == "MLX":
        return "MLX_MATMUL"
    if os.environ.get("METALFAISS_FORCE_SVD", "").upper() == "KERNEL":
        return "KERNEL_TILED"

    # Global gating: keep research kernels off unless explicitly enabled
    if not svd_kernels_enabled():
        return "MLX_MATMUL"
    dev = _device_type()
    # Use tiled kernels for larger problems; MLX matmul is very strong for small sizes.
    work = m * n * k
    # Prefer MLX for small/medium; switch to kernel at larger work sizes.
    if dev == "gpu" and work >= (16 * 1024 * 1024):  # ~16M MACs per iteration step
        return "KERNEL_TILED"
    return "MLX_MATMUL"
