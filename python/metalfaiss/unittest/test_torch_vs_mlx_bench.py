"""
PyTorch vs MLX micro-benchmarks (GEMM)

Compares:
- A@V (GEMM)
- Z-step: A^T (A V)

Across MLX (matmul), MLX with Metal kernels (if enabled), and PyTorch (MPS/CPU).

Usage
  METALFAISS_USE_GEMM_KERNEL=1 python -m python.metalfaiss.unittest.test_torch_vs_mlx_bench
"""

import os
import time
import unittest
from typing import Tuple

import mlx.core as mx

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore

try:  # optional torch
    import torch
    _HAVE_TORCH = True
except Exception:
    _HAVE_TORCH = False

from ..faissmlx.kernels import gemm_kernels as gk


def _median_time(fn, warmup=1, repeats=5):
    for _ in range(warmup):
        out = fn()
        if isinstance(out, tuple):
            mx.eval(*out)
        else:
            mx.eval(out)
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        if isinstance(out, tuple):
            mx.eval(*out)
        else:
            mx.eval(out)
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[len(ts)//2]


def _torch_device():
    if not _HAVE_TORCH:
        return None
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _make_inputs(m: int, n: int, k: int) -> Tuple[mx.array, mx.array, torch.Tensor | None, torch.Tensor | None]:
    """Create A (m,n), V (n,k) as MLX arrays and optional Torch tensors with same data."""
    if np is not None and _HAVE_TORCH:
        rng = np.random.default_rng(123)
        A_np = rng.standard_normal(size=(m, n), dtype=np.float32)
        V_np = rng.standard_normal(size=(n, k), dtype=np.float32)
        A_mx = mx.array(A_np)
        V_mx = mx.array(V_np)
        dev = _torch_device()
        if dev is not None:
            A_t = torch.from_numpy(A_np.copy()).to(dev)
            V_t = torch.from_numpy(V_np.copy()).to(dev)
        else:
            A_t = None; V_t = None
        return A_mx, V_mx, A_t, V_t
    # Fallback: generate separately
    A_mx = mx.random.normal(shape=(m, n)).astype(mx.float32)
    V_mx = mx.random.normal(shape=(n, k)).astype(mx.float32)
    A_t = V_t = None
    if _HAVE_TORCH:
        dev = _torch_device()
        if dev is not None:
            A_t = torch.randn((m, n), dtype=torch.float32, device=dev)
            V_t = torch.randn((n, k), dtype=torch.float32, device=dev)
    return A_mx, V_mx, A_t, V_t


def _torch_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    out = A @ B
    if A.device.type == "mps":  # ensure op completed
        torch.mps.synchronize()
    return out


class TestTorchVsMLXBench(unittest.TestCase):
    def test_compare_gemm(self):
        shapes = [
            (256, 128, 16),
            (512, 256, 32),
            (1024, 512, 64),
        ]
        use_kernel = os.environ.get("METALFAISS_USE_GEMM_KERNEL", "0") == "1"
        dev = _torch_device()
        print(f"\n[PyTorch vs MLX] device={dev} use_kernel={use_kernel}")
        for (m, n, k) in shapes:
            print(f"\n[GEMM] shape=({m}x{n}, k={k})")
            A_mx, V_mx, A_t, V_t = _make_inputs(m, n, k)

            # MLX matmul
            def run_mlx_gemm():
                return mx.matmul(A_mx, V_mx)
            t_mlx = _median_time(run_mlx_gemm)
            print(f"  MLX matmul:  {t_mlx:.4f}s")

            # MLX kernel path (if requested)
            if use_kernel:
                def run_mlx_kernel():
                    return gk.gemm_av(A_mx, V_mx)
                t_kernel = _median_time(run_mlx_kernel)
                print(f"  MLX kernel:  {t_kernel:.4f}s")

            # Torch matmul, if available
            if _HAVE_TORCH and A_t is not None and V_t is not None:
                # warmup
                _torch_matmul(A_t, V_t)
                def run_torch():
                    return _torch_matmul(A_t, V_t)
                ts = []
                for _ in range(5):
                    t0 = time.perf_counter()
                    _ = run_torch()
                    ts.append(time.perf_counter() - t0)
                ts.sort(); t_torch = ts[len(ts)//2]
                print(f"  Torch @:     {t_torch:.4f}s")
            else:
                print("  Torch @:     (torch not available)")

    def test_compare_zstep(self):
        shapes = [
            (256, 128, 16),
            (512, 256, 32),
        ]
        use_kernel = os.environ.get("METALFAISS_USE_GEMM_KERNEL", "0") == "1"
        dev = _torch_device()
        print(f"\n[PyTorch vs MLX Z-step] device={dev} use_kernel={use_kernel}")
        for (m, n, k) in shapes:
            print(f"\n[Z-step] shape=({m}x{n}, k={k})")
            A_mx, V_mx, A_t, V_t = _make_inputs(m, n, k)

            # MLX baseline
            def run_mlx_z():
                AV = mx.matmul(A_mx, V_mx)
                return mx.matmul(mx.transpose(A_mx), AV)
            t_mlx = _median_time(run_mlx_z)
            print(f"  MLX (mx.matmul): {t_mlx:.4f}s")

            if use_kernel:
                def run_kernel_z():
                    B = gk.gemm_av(A_mx, V_mx)
                    return gk.gemm_at_b(A_mx, B)
                t_kernel = _median_time(run_kernel_z)
                print(f"  MLX (kernels):  {t_kernel:.4f}s")

            if _HAVE_TORCH and A_t is not None and V_t is not None:
                if dev and dev.type == "mps":
                    torch.mps.synchronize()
                def run_torch_z():
                    AV = A_t @ V_t
                    Z = A_t.t() @ AV
                    if dev and dev.type == "mps":
                        torch.mps.synchronize()
                    return Z
                ts = []
                for _ in range(5):
                    t0 = time.perf_counter(); _ = run_torch_z(); ts.append(time.perf_counter()-t0)
                ts.sort(); t_torch = ts[len(ts)//2]
                print(f"  Torch (A^T A V): {t_torch:.4f}s")
            else:
                print("  Torch (A^T A V): (torch not available)")


if __name__ == "__main__":
    unittest.main()

