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

###############################################################################
# Basic Distance Functions
###############################################################################

def fvec_L2sqr(x: mx.array, y: mx.array) -> mx.array:
    """
    Compute the squared L2 distance between two vectors.
    
    Args:
        x: MLX array of shape (d,).
        y: MLX array of shape (d,).
        
    Returns:
        MLX scalar (0-d array) with the squared L2 distance.
    """
    diff = x - y
    # mx.dot returns an MLX scalar (0-d array); keep it on device.
    return mx.dot(diff, diff)


def fvec_inner_product(x: mx.array, y: mx.array) -> mx.array:
    """
    Compute the inner product between two vectors.
    
    Args:
        x: MLX array of shape (d,).
        y: MLX array of shape (d,).
        
    Returns:
        MLX scalar (0-d array) with the inner product.
    """
    return mx.dot(x, y)


def fvec_L1(x: mx.array, y: mx.array) -> mx.array:
    """
    Compute the L1 (Manhattan) distance between two vectors.
    
    Args:
        x: MLX array of shape (d,).
        y: MLX array of shape (d,).
        
    Returns:
        MLX scalar (0-d array) with the L1 distance.
    """
    return mx.sum(mx.abs(x - y))


def fvec_Linf(x: mx.array, y: mx.array) -> mx.array:
    """
    Compute the Chebyshev (L∞) distance between two vectors.
    
    Args:
        x: MLX array of shape (d,).
        y: MLX array of shape (d,).
        
    Returns:
        MLX scalar (0-d array) with the L∞ distance.
    """
    return mx.max(mx.abs(x - y))


###############################################################################
# Additional Distance Functions (specialized domains)
###############################################################################

def fvec_canberra(x: mx.array, y: mx.array) -> mx.array:
    """Compute Canberra distance between two vectors."""
    num = mx.abs(x - y)
    den = mx.abs(x) + mx.abs(y)
    den = mx.where(den > 0, den, mx.ones_like(den))
    return mx.sum(mx.divide(num, den))


def fvec_bray_curtis(x: mx.array, y: mx.array) -> mx.array:
    """Compute Bray-Curtis distance between two vectors."""
    num = mx.sum(mx.abs(x - y))
    den = mx.sum(mx.abs(x + y))
    return mx.divide(num, mx.maximum(den, mx.array(1e-20, dtype=num.dtype)))


def fvec_jensen_shannon(x: mx.array, y: mx.array) -> mx.array:
    """Compute Jensen-Shannon divergence (symmetric KL) between two histograms."""
    m = mx.multiply(mx.array(0.5, dtype=x.dtype), mx.add(x, y))
    # Avoid log(0); 0 * log(0) treated as 0
    tiny = mx.array(1e-20, dtype=x.dtype)
    kl1 = mx.where(x > 0, mx.multiply(x, (mx.log(x) - mx.log(mx.maximum(m, tiny)))), mx.zeros_like(x))
    kl2 = mx.where(y > 0, mx.multiply(y, (mx.log(y) - mx.log(mx.maximum(m, tiny)))), mx.zeros_like(y))
    return mx.multiply(mx.array(0.5, dtype=x.dtype), mx.add(mx.sum(kl1), mx.sum(kl2)))


###############################################################################
# Batch Distance Computations
###############################################################################

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
    norms_q = mx.sum(mx.square(xq), axis=1, keepdims=True)  # shape: (nq, 1)
    norms_b = mx.sum(mx.square(xb), axis=1, keepdims=True)  # shape: (nb, 1)
    
    # Compute dot products: (nq, d) x (d, nb) = (nq, nb)
    dot_products = mx.matmul(xq, xb.T)
    
    # Combine: D = norms_q + norms_b.T - 2 * dot_products
    D = mx.subtract(mx.add(norms_q, mx.transpose(norms_b)), mx.add(dot_products, dot_products))
    
    # Due to rounding, some distances may be slightly negative.
    return mx.maximum(D, mx.zeros_like(D))


def pairwise_extra_distances(xq: mx.array, xb: mx.array, metric: str) -> mx.array:
    """Pairwise distances for extended metrics: Canberra, Bray-Curtis, Jensen-Shannon.

    Args:
        xq: (nq, d) queries
        xb: (nb, d) database
        metric: one of {"Canberra", "BrayCurtis", "JensenShannon"}
    Returns:
        (nq, nb) distances
    """
    nq, d = xq.shape
    nb = xb.shape[0]
    if metric == "Canberra":
        num = mx.abs(xq[:, None, :] - xb[None, :, :])
        den = mx.abs(xq[:, None, :]) + mx.abs(xb[None, :, :])
        den = mx.where(den > 0, den, mx.ones_like(den))
        return mx.sum(mx.divide(num, den), axis=2)
    if metric == "BrayCurtis":
        num = mx.sum(mx.abs(xq[:, None, :] - xb[None, :, :]), axis=2)
        den = mx.sum(mx.abs(xq[:, None, :] + xb[None, :, :]), axis=2)
        return mx.divide(num, mx.maximum(den, mx.array(1e-20, dtype=num.dtype)))
    if metric == "JensenShannon":
        # Compute per pair JSD; simple loop over nq for clarity (moderate sizes)
        out = mx.zeros((nq, nb), dtype=mx.float32)
        for i in range(nq):
            xi = xq[i]
            m = 0.5 * (xi[None, :] + xb)
            kl1 = mx.where(xi[None, :] > 0, xi[None, :] * (mx.log(xi[None, :]) - mx.log(mx.maximum(m, 1e-20))), 0.0)
            kl2 = mx.where(xb > 0, xb * (mx.log(xb) - mx.log(mx.maximum(m, 1e-20))), 0.0)
            out[i] = 0.5 * (mx.sum(kl1, axis=1) + mx.sum(kl2, axis=1))
        return out
    raise ValueError(f"Unsupported extra metric: {metric}")


###############################################################################
# Norm and Renormalization Functions
###############################################################################

def fvec_norms_L2(x: mx.array) -> mx.array:
    """
    Compute the L2 norms of each vector (row) in the input matrix.
    
    Args:
        x: MLX array of shape (n, d).
    
    Returns:
        MLX array of shape (n,) with the L2 norm of each vector.
    """
    return mx.sqrt(mx.sum(mx.square(x), axis=1))


def fvec_norms_L2sqr(x: mx.array) -> mx.array:
    """
    Compute the squared L2 norms for each vector (row) in the input matrix.
    
    Args:
        x: MLX array of shape (n, d).
    
    Returns:
        MLX array of shape (n,) with the squared L2 norm of each vector.
    """
    return mx.sum(mx.square(x), axis=1)


def fvec_renorm_L2(x: mx.array) -> mx.array:
    """
    L2-renormalize each vector (row) in the input matrix.
    
    Args:
        x: MLX array of shape (n, d).
    
    Returns:
        MLX array of shape (n, d) where each row has been normalized to unit norm.
    """
    norms = mx.sqrt(mx.sum(mx.square(x), axis=1, keepdims=True))
    norms = mx.where(norms > 0, norms, mx.ones_like(norms))
    return mx.divide(x, norms)


###############################################################################
# Functions for Single–Vector to Batch Operations
###############################################################################

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
    return mx.matmul(y, x.reshape((-1, 1))).reshape(-1)


def fvec_L2sqr_ny(x: mx.array, y: mx.array) -> mx.array:
    """
    Compute squared L2 distances between a single vector x and each row of y.
    
    Args:
        x: MLX array of shape (d,).
        y: MLX array of shape (ny, d).
    
    Returns:
        MLX array of shape (ny,) with the squared distances.
    """
    diff = y - x
    return mx.sum(mx.square(diff), axis=1)


def pairwise_L1(xq: mx.array, xb: mx.array) -> mx.array:
    """Pairwise L1 distances between rows of xq (nq,d) and xb (nb,d)."""
    return mx.sum(mx.abs(xq[:, None, :] - xb[None, :, :]), axis=2)


def pairwise_Linf(xq: mx.array, xb: mx.array) -> mx.array:
    """Pairwise Linf distances between rows of xq (nq,d) and xb (nb,d)."""
    return mx.max(mx.abs(xq[:, None, :] - xb[None, :, :]), axis=2)


def pairwise_jaccard(xq: mx.array, xb: mx.array) -> mx.array:
    """Pairwise Jaccard distances (non-negative vectors) between xq and xb.

    J(x,y) = 1 - sum(min(x,y)) / sum(max(x,y))
    """
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
