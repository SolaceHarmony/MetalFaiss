"""
svd.py - MLX-only tiled SVD via block power iteration

Provides a GPU-friendly top-k SVD using subspace (block) power iteration.
This uses only MLX ops (matmul, norms, our MLX QR) and keeps compute on GPU.
"""

from typing import Tuple
import mlx.core as mx
from .qr import pure_mlx_qr
from .dispatch import choose_svd_impl
from .kernels.gemm_kernels import gemm_av, gemm_at_b
import os

_COMPILED_MLX_ITER = None

def _mlx_power_iter_once(A: mx.array, V: mx.array) -> mx.array:
    """One power-iteration step using MLX GEMMs; returns re-orthonormalized V.

    Shapes: A (m,n), V (n,k) -> V_next (n,k)
    """
    AV = mx.matmul(A, V)
    Z = mx.matmul(mx.transpose(A), AV)
    Qz, _ = pure_mlx_qr(Z)
    return Qz


def _kernel_zstep_banded(
    A: mx.array,
    V: mx.array,
    band_size: int,
    streams: int | None = None,
) -> mx.array:
    """Compute Z = A^T (A V) by column bands, optionally across multiple streams.

    Args:
        A: (m, n)
        V: (n, k)
        band_size: number of columns per band
        streams: if None or <=1, run serially; otherwise create this many streams and
                 round-robin bands across them.
    Returns:
        Z: (n, k)
    """
    n, k = int(V.shape[0]), int(V.shape[1])
    if band_size <= 0 or band_size >= k:
        B = gemm_av(A, V)
        return gemm_at_b(A, B)

    bands = [(s, min(s + band_size, k)) for s in range(0, k, band_size)]
    outs = [None] * len(bands)

    if streams is None or streams <= 1:
        for idx, (s, e) in enumerate(bands):
            Vb = V[:, s:e]
            Bb = gemm_av(A, Vb)
            Zb = gemm_at_b(A, Bb)
            outs[idx] = Zb
        mx.eval(*outs)  # ensure completion before concat
        return mx.concatenate(outs, axis=1)

    # Multi-stream execution
    S = max(1, int(streams))
    dev = mx.default_device()
    stream_objs = [mx.new_stream(dev) for _ in range(S)]
    for idx, (s, e) in enumerate(bands):
        st = stream_objs[idx % S]
        with mx.stream(st):
            Vb = V[:, s:e]
            Bb = gemm_av(A, Vb)
            Zb = gemm_at_b(A, Bb)
            outs[idx] = Zb
    # Fan-in barrier before concatenation
    mx.synchronize()
    return mx.concatenate(outs, axis=1)


def topk_svd(
    A: mx.array,
    k: int,
    iters: int = 8,
    use_kernel: bool = False,
    use_compile: bool = False,
    band_size: int | None = None,
) -> Tuple[mx.array, mx.array, mx.array]:
    """Approximate top-k SVD of A using block power iteration.

    Args:
        A: (m, n) matrix
        k: number of leading singular vectors/values to compute (k <= min(m,n))
        iters: number of subspace iteration steps

    Returns:
        U: (m, k), S: (k,), Vt: (k, n)
    """
    m, n = int(A.shape[0]), int(A.shape[1])
    k = min(k, m, n)

    # Initialize V with random normal and orthonormalize
    V0 = mx.random.normal(shape=(n, k)).astype(A.dtype)
    Qv, _ = pure_mlx_qr(V0)
    V = Qv[:, :k]  # (n, k)

    # Subspace iteration: repeatedly apply (A^T A) to V and re-orthonormalize
    for _ in range(max(1, iters)):
        impl = None
        if use_kernel or os.environ.get("METALFAISS_USE_SVD_KERNEL", "0") == "1":
            impl = "KERNEL_TILED"
        else:
            impl = choose_svd_impl(m, n, k)

        if impl == "KERNEL_TILED":
            # Kernel path (optional banding + streams)
            env_band = os.environ.get("METALFAISS_SVD_BAND")
            bsz = band_size or (int(env_band) if env_band else None)
            env_streams = os.environ.get("METALFAISS_SVD_STREAMS")
            nstreams = int(env_streams) if env_streams else None
            if bsz is not None and bsz > 0 and bsz < k:
                Z = _kernel_zstep_banded(A, V, bsz, streams=nstreams)
            else:
                B = gemm_av(A, V)
                Z = gemm_at_b(A, B)
            Qz, _ = pure_mlx_qr(Z)
            V = Qz[:, :k]
        else:
            if use_compile or os.environ.get("METALFAISS_USE_COMPILE", "0") == "1":
                global _COMPILED_MLX_ITER
                if _COMPILED_MLX_ITER is None:
                    if hasattr(mx, "compile"):
                        _COMPILED_MLX_ITER = mx.compile(_mlx_power_iter_once)
                    else:
                        _COMPILED_MLX_ITER = _mlx_power_iter_once
                Qz = _COMPILED_MLX_ITER(A, V)
                V = Qz[:, :k]
            else:
                AV = mx.matmul(A, V)         # (m, k)
                Z = mx.matmul(mx.transpose(A), AV)  # (n, k)
                Qz, _ = pure_mlx_qr(Z)
                V = Qz[:, :k]

    # Ritz values/vectors: compute U = A V, then singular values as norms
    AV = mx.matmul(A, V)    # (m, k)
    # Column norms for singular values
    s = mx.sqrt(mx.sum(AV * AV, axis=0))  # (k,)
    # Avoid div by zero
    inv = mx.where(s > 0, 1.0 / s, 0.0)
    U = AV * inv.reshape((1, -1))
    Vt = mx.transpose(V)
    return U, s, Vt
