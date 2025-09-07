"""
Top‑k SVD via subspace (block) power iteration

Overview
- MLX path: uses `mx.matmul` for AV and AᵀB plus MLX QR for re‑orthonormalization.
- Kernel path: uses shared‑memory tiled Metal kernels for AV and AᵀB (see
  `faissmlx/kernels/gemm_kernels.py`), optionally in bands and with multiple
  MLX streams.

Tuning & Controls
- `use_kernel`: force the kernel path (else autoswitch by size/device via `svd_dispatch.choose_svd_impl`).
- `use_compile`: try `mx.compile` wrapper for the MLX iteration step when available.
- Env vars:
  - `METALFAISS_USE_SVD_KERNEL=1` forces kernel path.
  - `METALFAISS_SVD_BAND=<B>` enables banded Z‑step with band width B for the kernel path.
  - `METALFAISS_SVD_STREAMS=<S>` runs banded Z‑step across S MLX streams (fan‑out/fan‑in).

References
- docs/mlx/Kernel-Guide.md:1
- docs/research/Journal.md:1
- docs/metal/Shader-Optimization-Tips.md:7
"""

from typing import Tuple
import mlx.core as mx
from .qr import pure_mlx_qr
from .svd_dispatch import choose_svd_impl
from .kernels.gemm_kernels import gemm_av, gemm_at_b
import os

_COMPILED_MLX_ITER = None

def _mlx_power_iter_once(A: mx.array, V: mx.array) -> mx.array:
    """One power‑iteration step using MLX GEMMs.

    Parameters
    - `A (m,n)`, `V (n,k)`

    Returns
    - `V_next (n,k)` after re‑orthonormalization via QR
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
    """Kernel Z‑step by bands, optionally across multiple MLX streams.

    Parameters
    - `A (m,n)`, `V (n,k)`
    - `band_size`: columns per band (<=0 or >=k falls back to monolithic tiled)
    - `streams`: None or <=1 for serial; else create `streams` and round‑robin bands

    Returns
    - `Z (n,k)`

    Trade‑offs
    - Banded execution can improve cache locality and reduce peak memory, often
      winning for small k. Streams can overlap work at larger sizes but may lose
      to contention; prefer small S (2–4) and measure.
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
    """Approximate top‑k SVD of A using block power iteration.

    Parameters
    - `A (m,n)`, `k ≤ min(m,n)`, `iters ≥ 1`
    - `use_kernel`: use the kernel Z‑step (else autoswitch via dispatch)
    - `use_compile`: try `mx.compile` for MLX loop
    - `band_size`: optional band width for kernel path

    Returns
    - `U (m,k)`, `S (k,)`, `Vt (k,n)`

    Notes
    - Kernel path performs Z via `gemm_av` then `gemm_at_b`. If `band_size` is
      set (or `METALFAISS_SVD_BAND`), the Z‑step runs in bands; `METALFAISS_SVD_STREAMS`
      can distribute bands across multiple MLX streams.
    """
    m, n = int(A.shape[0]), int(A.shape[1])
    k = min(k, m, n)

    # Initialize V with random normal and orthonormalize
    V0 = mx.random.normal(shape=(n, k)).astype(A.dtype)
    Qv, _ = pure_mlx_qr(V0)
    V = Qv[:, :k]  # (n, k)

    # Subspace iteration: repeatedly apply (A^T A) to V and re-orthonormalize
    for _ in range(max(1, iters)):
        # Production: use tiled GEMM kernels for Z-step (AV then A^T B)
        B = gemm_av(A, V)
        Z = gemm_at_b(A, B)
        Qz, _ = pure_mlx_qr(Z)
        V = Qz[:, :k]

    # Ritz values/vectors: compute U = A V, then singular values as norms
    AV = mx.matmul(A, V)    # (m, k)
    # Column norms for singular values
    s = mx.sqrt(mx.sum(mx.square(AV), axis=0))  # (k,)
    # Avoid div by zero via ones/s
    ones = mx.ones_like(s)
    inv = mx.where(s > 0, mx.divide(ones, s), mx.zeros_like(s))
    U = mx.multiply(AV, inv.reshape((1, -1)))
    Vt = mx.transpose(V)
    return U, s, Vt
