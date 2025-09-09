# MetalFaiss - A pure Python implementation of FAISS using MLX for Metal acceleration
# Copyright (c) 2024 Sydney Bach, The Solace Project
# Licensed under the Apache License, Version 2.0 (see LICENSE file)
#
# Original Swift implementation by Jan Krukowski used as reference for Python translation

"""
mlx/distances.py

This module contains basic vector-to-vector distance functions adapted
from the FAISS C++ API. All operations use MLX’s array type (mx.array)
to support lazy evaluation and GPU–acceleration. The functions provided here
mirror the “fvec_…” functions in FAISS, such as fvec_L2sqr, fvec_inner_product,
fvec_L1, and fvec_Linf. Also included is a pairwise L2–squared distance
function that computes the full distance matrix between a query set and a database.
"""

import math

import mlx.core as mx
from .faissmlx.device_guard import require_gpu

# Compile wrapper: MX compile works best when functions are defined once
# at import time and reused (avoid per-call creation/destruction of callables).
try:
    compile_fn = mx.compile  # type: ignore[attr-defined]
except Exception:  # pragma: no cover
    def compile_fn(f):
        return f

###############################################################################
# Basic Distance Functions
###############################################################################

@compile_fn
def fvec_L2sqr(x: mx.array, y: mx.array) -> mx.array:
    """
    Compute the squared L2 distance between two vectors.
    
    Args:
        x: MLX array of shape (d,).
        y: MLX array of shape (d,).
        
    Returns:
        MLX scalar (0-d array) with the squared L2 distance.
    """
    require_gpu("fvec_L2sqr")
    diff = mx.subtract(x, y)
    # mx.dot returns an MLX scalar (0-d array); keep it on device.
    return mx.dot(diff, diff)


@compile_fn
def fvec_inner_product(x: mx.array, y: mx.array) -> mx.array:
    """
    Compute the inner product between two vectors.
    
    Args:
        x: MLX array of shape (d,).
        y: MLX array of shape (d,).
        
    Returns:
        MLX scalar (0-d array) with the inner product.
    """
    require_gpu("fvec_inner_product")
    return mx.dot(x, y)


@compile_fn
def fvec_L1(x: mx.array, y: mx.array) -> mx.array:
    """
    Compute the L1 (Manhattan) distance between two vectors.
    
    Args:
        x: MLX array of shape (d,).
        y: MLX array of shape (d,).
        
    Returns:
        MLX scalar (0-d array) with the L1 distance.
    """
    require_gpu("fvec_L1")
    return mx.sum(mx.abs(mx.subtract(x, y)))


@compile_fn
def fvec_Linf(x: mx.array, y: mx.array) -> mx.array:
    """
    Compute the Chebyshev (L∞) distance between two vectors.
    
    Args:
        x: MLX array of shape (d,).
        y: MLX array of shape (d,).
        
    Returns:
        MLX scalar (0-d array) with the L∞ distance.
    """
    require_gpu("fvec_Linf")
    return mx.max(mx.abs(mx.subtract(x, y)))


###############################################################################
# Additional Distance Functions (specialized domains)
###############################################################################

@compile_fn
def fvec_canberra(x: mx.array, y: mx.array) -> mx.array:
    """Compute Canberra distance between two vectors."""
    require_gpu("fvec_canberra")
    num = mx.abs(mx.subtract(x, y))
    den = mx.add(mx.abs(x), mx.abs(y))
    den = mx.where(mx.greater(den, mx.zeros_like(den)), den, mx.ones_like(den))
    return mx.sum(mx.divide(num, den))


@compile_fn
def fvec_bray_curtis(x: mx.array, y: mx.array) -> mx.array:
    """Compute Bray-Curtis distance between two vectors."""
    require_gpu("fvec_bray_curtis")
    num = mx.sum(mx.abs(mx.subtract(x, y)))
    den = mx.sum(mx.abs(mx.add(x, y)))
    return mx.divide(num, mx.maximum(den, mx.array(1e-20, dtype=num.dtype)))


@compile_fn
def fvec_jensen_shannon(x: mx.array, y: mx.array) -> mx.array:
    """Compute Jensen-Shannon divergence (symmetric KL) between two histograms."""
    require_gpu("fvec_jensen_shannon")
    m = mx.multiply(mx.array(0.5, dtype=x.dtype), mx.add(x, y))
    # Avoid log(0); 0 * log(0) treated as 0
    tiny = mx.array(1e-20, dtype=x.dtype)
    kl1 = mx.where(x > 0, mx.multiply(x, mx.subtract(mx.log(x), mx.log(mx.maximum(m, tiny)))), mx.zeros_like(x))
    kl2 = mx.where(y > 0, mx.multiply(y, mx.subtract(mx.log(y), mx.log(mx.maximum(m, tiny)))), mx.zeros_like(y))
    return mx.multiply(mx.array(0.5, dtype=x.dtype), mx.add(mx.sum(kl1), mx.sum(kl2)))


###############################################################################
# Batch Distance Computations
###############################################################################

@compile_fn
def pairwise_L2sqr(xq: mx.array, xb: mx.array) -> mx.array:
    """
    Compute pairwise squared L2 distances between two sets of vectors.
    
    Given a query matrix xq of shape (nq, d) and a database matrix xb of shape (nb, d),
    this function returns an MLX array D of shape (nq, nb) where:
    
        D[i, j] = || xq[i] - xb[j] ||^2
    
    This is computed using the identity:
    
        || a - b ||^2 = ||a||^2 + ||b||^2 - 2 * <a, b>
    
    Args:
        xq: MLX array of query vectors (shape: [nq, d]).
        xb: MLX array of database vectors (shape: [nb, d]).
    
    Returns:
        MLX array of squared distances with shape (nq, nb).
    """
    # Compute squared norms for queries and database vectors.
    require_gpu("pairwise_L2sqr")
    norms_q = mx.sum(mx.square(xq), axis=1, keepdims=True)  # shape: (nq, 1)
    norms_b = mx.sum(mx.square(xb), axis=1, keepdims=True)  # shape: (nb, 1)
    
    # Compute dot products: (nq, d) x (d, nb) = (nq, nb)
    dot_products = mx.matmul(xq, mx.transpose(xb))
    
    # Combine: D = norms_q + norms_b.T - 2 * dot_products
    D = mx.subtract(mx.add(norms_q, mx.transpose(norms_b)), mx.add(dot_products, dot_products))
    
    # Due to rounding, some distances may be slightly negative.
    return mx.maximum(D, mx.zeros_like(D))


@compile_fn
def pairwise_extra_distances(xq: mx.array, xb: mx.array, metric: str) -> mx.array:
    """Pairwise distances for extended metrics: Canberra, Bray-Curtis, Jensen-Shannon.

    Args:
        xq: (nq, d) queries
        xb: (nb, d) database
        metric: one of {"Canberra", "BrayCurtis", "JensenShannon"}
    Returns:
        (nq, nb) distances
    """
    require_gpu("pairwise_extra_distances")
    nq, d = xq.shape
    nb = xb.shape[0]
    if metric == "Canberra":
        num = mx.abs(mx.subtract(xq[:, None, :], xb[None, :, :]))
        den = mx.add(mx.abs(xq[:, None, :]), mx.abs(xb[None, :, :]))
        den = mx.where(mx.greater(den, mx.zeros_like(den)), den, mx.ones_like(den))
        return mx.sum(mx.divide(num, den), axis=2)
    if metric == "BrayCurtis":
        num = mx.sum(mx.abs(mx.subtract(xq[:, None, :], xb[None, :, :])), axis=2)
        den = mx.sum(mx.abs(mx.add(xq[:, None, :], xb[None, :, :])), axis=2)
        return mx.divide(num, mx.maximum(den, mx.array(1e-20, dtype=num.dtype)))
    if metric == "JensenShannon":
        # Fully vectorized per pair (nq, nb, d)
        # p = xq[:, None, :], q = xb[None, :, :], m = 0.5*(p+q)
        p = xq[:, None, :]
        q = xb[None, :, :]
        m = mx.multiply(mx.array(0.5, dtype=p.dtype), mx.add(p, q))
        tiny = mx.array(1e-20, dtype=p.dtype)
        kl1 = mx.where(mx.greater(p, mx.zeros_like(p)),
                       mx.multiply(p, mx.subtract(mx.log(p), mx.log(mx.maximum(m, tiny)))),
                       mx.zeros_like(p))
        kl2 = mx.where(mx.greater(q, mx.zeros_like(q)),
                       mx.multiply(q, mx.subtract(mx.log(q), mx.log(mx.maximum(m, tiny)))),
                       mx.zeros_like(q))
        return mx.multiply(mx.array(0.5, dtype=p.dtype), mx.add(mx.sum(kl1, axis=2), mx.sum(kl2, axis=2)))
    raise ValueError(f"Unsupported extra metric: {metric}")


###############################################################################
# Norm and Renormalization Functions
###############################################################################

@compile_fn
def fvec_norms_L2(x: mx.array) -> mx.array:
    """
    Compute the L2 norms of each vector (row) in the input matrix.
    
    Args:
        x: MLX array of shape (n, d).
    
    Returns:
        MLX array of shape (n,) with the L2 norm of each vector.
    """
    require_gpu("fvec_norms_L2")
    return mx.sqrt(mx.sum(mx.square(x), axis=1))


@compile_fn
def fvec_norms_L2sqr(x: mx.array) -> mx.array:
    """
    Compute the squared L2 norms for each vector (row) in the input matrix.
    
    Args:
        x: MLX array of shape (n, d).
    
    Returns:
        MLX array of shape (n,) with the squared L2 norm of each vector.
    """
    require_gpu("fvec_norms_L2sqr")
    return mx.sum(mx.square(x), axis=1)


@compile_fn
def fvec_renorm_L2(x: mx.array) -> mx.array:
    """
    L2-renormalize each vector (row) in the input matrix.
    
    Args:
        x: MLX array of shape (n, d).
    
    Returns:
        MLX array of shape (n, d) where each row has been normalized to unit norm.
    """
    require_gpu("fvec_renorm_L2")
    norms = mx.sqrt(mx.sum(mx.square(x), axis=1, keepdims=True))
    norms = mx.where(mx.greater(norms, mx.zeros_like(norms)), norms, mx.ones_like(norms))
    return mx.divide(x, norms)


###############################################################################
# Functions for Single–Vector to Batch Operations
###############################################################################

@compile_fn
def fvec_inner_products_ny(x: mx.array, y: mx.array) -> mx.array:
    """
    Compute the inner product between a single vector x and each row of y.
    
    Args:
        x: MLX array of shape (d,).
        y: MLX array of shape (ny, d).
    
    Returns:
        MLX array of shape (ny,) containing the inner products.
    """
    # Reshape x to (d,1) so that mx.matmul yields shape (ny,1), then reshape.
    require_gpu("fvec_inner_products_ny")
    return mx.matmul(y, x.reshape((-1, 1))).reshape(-1)


@compile_fn
def fvec_L2sqr_ny(x: mx.array, y: mx.array) -> mx.array:
    """
    Compute squared L2 distances between a single vector x and each row of y.
    
    Args:
        x: MLX array of shape (d,).
        y: MLX array of shape (ny, d).
    
    Returns:
        MLX array of shape (ny,) with the squared distances.
    """
    require_gpu("fvec_L2sqr_ny")
    diff = mx.subtract(y, x)
    return mx.sum(mx.square(diff), axis=1)


@compile_fn
def pairwise_L1(xq: mx.array, xb: mx.array) -> mx.array:
    """Pairwise L1 distances between rows of xq (nq,d) and xb (nb,d)."""
    require_gpu("pairwise_L1")
    return mx.sum(mx.abs(mx.subtract(xq[:, None, :], xb[None, :, :])), axis=2)


@compile_fn
def pairwise_Linf(xq: mx.array, xb: mx.array) -> mx.array:
    """Pairwise Linf distances between rows of xq (nq,d) and xb (nb,d)."""
    require_gpu("pairwise_Linf")
    return mx.max(mx.abs(mx.subtract(xq[:, None, :], xb[None, :, :])), axis=2)


@compile_fn
def pairwise_jaccard(xq: mx.array, xb: mx.array) -> mx.array:
    """Pairwise Jaccard distances (non-negative vectors) between xq and xb.

    J(x,y) = 1 - sum(min(x,y)) / sum(max(x,y))
    """
    require_gpu("pairwise_jaccard")
    minimum = mx.minimum(xq[:, None, :], xb[None, :, :])
    maximum = mx.maximum(xq[:, None, :], xb[None, :, :])
    num = mx.sum(minimum, axis=2)
    den = mx.sum(maximum, axis=2)
    ones = mx.ones_like(num)
    frac = mx.divide(num, mx.maximum(den, mx.array(1e-20, dtype=num.dtype)))
    return mx.subtract(ones, frac)


###############################################################################
# End of File
###############################################################################
