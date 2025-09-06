"""
gemm_kernels.py - Tiled GEMM kernels (MLX JIT/Metal)

Provides shared-memory tiled kernels for:
- B = A (m,n) x V (n,k) -> (m,k)
- Z = A^T (n,m) x B (m,k) -> (n,k)

These are specialized for row-contiguous float32 arrays and use explicit
grid/threadgroup sizing with body-only kernel sources and a header that
contains includes and namespace — matching MLX fast.metal_kernel contract.

Notes:
- Tile sizes chosen for Apple GPU execution width (32) and ≤1024 threads/tg.
- Default tiles: TM=16, TN=16, TK=16 (256 threads/tg). We may tune later.
"""

from __future__ import annotations
from typing import Tuple
import mlx.core as mx

_HEADER = """#include <metal_stdlib>\nusing namespace metal;\n"""

# B = A (m,n) * V (n,k) -> (m,k)
_BODY_AV = r"""
    // Threadgroup-tiled GEMM: C = A * B, here C=B, A=A, B=V
    // Shapes via shape buffer: [m, n, k]
    const uint TM = 16; // tile size along m (rows of A / rows of C)
    const uint TN = 16; // tile size along n (shared dimension)
    const uint TK = 16; // tile size along k (cols of V / cols of C)

    threadgroup float Asub[TM][TN];
    threadgroup float Vsub[TN][TK];

    int m = int(shape[0]);
    int n = int(shape[1]);
    int k = int(shape[2]);

    uint local_x = thread_position_in_threadgroup.x; // 0..TK-1 -> col in tile
    uint local_y = thread_position_in_threadgroup.y; // 0..TM-1 -> row in tile

    int block_x = int(threadgroup_position_in_grid.x); // tile col index
    int block_y = int(threadgroup_position_in_grid.y); // tile row index

    int row = block_y * int(TM) + int(local_y); // output row in [0,m)
    int col = block_x * int(TK) + int(local_x); // output col in [0,k)

    float acc = 0.0f;

    // Iterate over tiles of the shared dimension n
    int ntiles = (n + int(TN) - 1) / int(TN);
    for (int t = 0; t < ntiles; ++t) {
        // Global col in A / row in V tile
        int a_col = t * int(TN) + int(local_x); // reuse local_x for Asub column load
        int v_row = t * int(TN) + int(local_y); // reuse local_y for Vsub row load

        // Load Asub[rowTile, p] and Vsub[p, colTile]
        float a_val = 0.0f;
        if (row < m && a_col < n) {
            a_val = A[row * n + a_col];
        }
        Asub[local_y][local_x] = a_val;

        float v_val = 0.0f;
        if (v_row < n && col < k) {
            v_val = V[v_row * k + col];
        }
        Vsub[local_y][local_x] = v_val;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Accumulate over TN
        for (uint p = 0; p < TN; ++p) {
            acc = fma(Asub[local_y][p], Vsub[p][local_x], acc);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < m && col < k) {
        C[row * k + col] = acc;
    }
"""

# Z = A^T (n,m) * B (m,k) -> (n,k)
_BODY_AT_B = r"""
    // Threadgroup-tiled GEMM for Z = A^T * B
    // Shapes: A (m,n), B (m,k), Z (n,k), shape=[m,n,k]
    const uint TN = 16; // tile size along n (rows of Z)
    const uint TI = 16; // tile size along m (shared dimension)
    const uint TK = 16; // tile size along k (cols of Z)

    threadgroup float Atile[TI][TN]; // A^T tile: [i in m][n in n]
    threadgroup float Btile[TI][TK];

    int m = int(shape[0]);
    int n = int(shape[1]);
    int k = int(shape[2]);

    uint local_x = thread_position_in_threadgroup.x; // 0..TK-1 -> col in tile
    uint local_y = thread_position_in_threadgroup.y; // 0..TN-1 -> row in tile

    int block_x = int(threadgroup_position_in_grid.x); // tile col index in k
    int block_y = int(threadgroup_position_in_grid.y); // tile row index in n

    int rowN = block_y * int(TN) + int(local_y); // output row in n
    int colK = block_x * int(TK) + int(local_x); // output col in k

    float acc = 0.0f;

    int itiles = (m + int(TI) - 1) / int(TI); // walk shared dimension m
    for (int t = 0; t < itiles; ++t) {
        int i = t * int(TI) + int(local_y); // row in m for loads

        // Load tiles
        float a_val = 0.0f;
        if (i < m && rowN < n) {
            // A^T[rowN, i] = A[i, rowN]
            a_val = A[i * n + rowN];
        }
        Atile[local_y][local_x] = a_val; // reuse local_x as n-index within tile

        float b_val = 0.0f;
        int i2 = t * int(TI) + int(local_y); // same i for B load
        if (i2 < m && colK < k) {
            b_val = B[i2 * k + colK];
        }
        Btile[local_y][local_x] = b_val;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Accumulate over TI
        for (uint p = 0; p < TI; ++p) {
            acc = fma(Atile[p][local_y], Btile[p][local_x], acc);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (rowN < n && colK < k) {
        Z[rowN * k + colK] = acc;
    }
"""

_KERNEL_AV = None
_KERNEL_AT_B = None


def _build_av_kernel():
    return mx.fast.metal_kernel(
        name="gemm_av_tiled",
        input_names=["A", "V", "shape"],
        output_names=["C"],
        header=_HEADER,
        source=_BODY_AV,
        ensure_row_contiguous=True,
    )


def _build_at_b_kernel():
    return mx.fast.metal_kernel(
        name="gemm_at_b_tiled",
        input_names=["A", "B", "shape"],
        output_names=["Z"],
        header=_HEADER,
        source=_BODY_AT_B,
        ensure_row_contiguous=True,
    )


def gemm_av(A: mx.array, V: mx.array) -> mx.array:
    """Compute B = A @ V with tiled Metal kernel.

    Args:
        A: (m, n)
        V: (n, k)
    Returns:
        B: (m, k)
    """
    global _KERNEL_AV
    if _KERNEL_AV is None:
        _KERNEL_AV = _build_av_kernel()

    m, n = int(A.shape[0]), int(A.shape[1])
    k = int(V.shape[1])
    shape = mx.array([m, n, k], dtype=mx.uint32)

    # Tile sizes (TM,TK) = (16,16) -> threadgroup (16,16,1)
    grid_x = (k + 16 - 1) // 16
    grid_y = (m + 16 - 1) // 16
    grid = (grid_x, grid_y, 1)
    threadgroup = (16, 16, 1)

    (B,) = _KERNEL_AV(
        inputs=[A, V, shape],
        output_shapes=[(m, k)],
        output_dtypes=[A.dtype],
        grid=grid,
        threadgroup=threadgroup,
    )
    return B


def gemm_at_b(A: mx.array, B: mx.array) -> mx.array:
    """Compute Z = A.T @ B with tiled Metal kernel.

    Args:
        A: (m, n)
        B: (m, k)
    Returns:
        Z: (n, k)
    """
    global _KERNEL_AT_B
    if _KERNEL_AT_B is None:
        _KERNEL_AT_B = _build_at_b_kernel()

    m, n = int(A.shape[0]), int(A.shape[1])
    k = int(B.shape[1])
    shape = mx.array([m, n, k], dtype=mx.uint32)

    # Tile sizes (TN,TK) = (16,16)
    grid_x = (k + 16 - 1) // 16
    grid_y = (n + 16 - 1) // 16
    grid = (grid_x, grid_y, 1)
    threadgroup = (16, 16, 1)

    (Z,) = _KERNEL_AT_B(
        inputs=[A, B, shape],
        output_shapes=[(n, k)],
        output_dtypes=[A.dtype],
        grid=grid,
        threadgroup=threadgroup,
    )
    return Z
