"""
dispatch.py - Autoswitching helpers for implementation selection

Select MLX vs. Metal-kernel implementations based on device and size.
Follows the pattern used elsewhere: prefer MLX vectorized ops for
small/medium problems; use kernels for larger tiles; allow env overrides.
"""

from __future__ import annotations
from typing import Literal, Dict
import os
import mlx.core as mx


def _device_type() -> str:
    # Best-effort device detection. Keep simple and robust.
    try:
        # If MLX exposes metal module, we assume GPU is available
        import mlx.metal as _mx_metal  # type: ignore
        return "gpu"
    except Exception:
        return "cpu"


def choose_qr_impl(m: int, k: int) -> Literal["MLX_MGS", "KERNEL_PROJ"]:
    # Env overrides
    if os.environ.get("METALFAISS_FORCE_QR", "").upper() == "MLX":
        return "MLX_MGS"
    if os.environ.get("METALFAISS_FORCE_QR", "").upper() == "KERNEL":
        return "KERNEL_PROJ"

    dev = _device_type()
    # Threshold heuristic: when there are many columns or long vectors,
    # using a kernel for c = Q^T v becomes beneficial.
    if dev == "gpu" and (m * k) >= (256 * 1024):  # ~256k elements
        return "KERNEL_PROJ"
    return "MLX_MGS"


def choose_svd_impl(m: int, n: int, k: int) -> Literal["MLX_MATMUL", "KERNEL_TILED"]:
    # Env overrides
    if os.environ.get("METALFAISS_FORCE_SVD", "").upper() == "MLX":
        return "MLX_MATMUL"
    if os.environ.get("METALFAISS_FORCE_SVD", "").upper() == "KERNEL":
        return "KERNEL_TILED"

    dev = _device_type()
    # Use tiled kernels for larger problems; MLX matmul is very strong for small sizes.
    work = m * n * k
    # Prefer MLX for small/medium; switch to kernel at larger work sizes.
    if dev == "gpu" and work >= (16 * 1024 * 1024):  # ~16M MACs per iteration step
        return "KERNEL_TILED"
    return "MLX_MATMUL"
