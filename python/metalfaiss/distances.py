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
    norms_q = mx.sum(xq * xq, axis=1, keepdims=True)  # shape: (nq, 1)
    norms_b = mx.sum(xb * xb, axis=1, keepdims=True)  # shape: (nb, 1)
    
    # Compute dot products: (nq, d) x (d, nb) = (nq, nb)
    dot_products = mx.matmul(xq, xb.T)
    
    # Combine: D = norms_q + norms_b.T - 2 * dot_products
    D = norms_q + mx.transpose(norms_b) - 2 * dot_products
    
    # Due to rounding, some distances may be slightly negative.
    return mx.maximum(D, 0.0)


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
    return mx.sqrt(mx.sum(x * x, axis=1))


def fvec_norms_L2sqr(x: mx.array) -> mx.array:
    """
    Compute the squared L2 norms for each vector (row) in the input matrix.
    
    Args:
        x: MLX array of shape (n, d).
    
    Returns:
        MLX array of shape (n,) with the squared L2 norm of each vector.
    """
    return mx.sum(x * x, axis=1)


def fvec_renorm_L2(x: mx.array) -> mx.array:
    """
    L2-renormalize each vector (row) in the input matrix.
    
    Args:
        x: MLX array of shape (n, d).
    
    Returns:
        MLX array of shape (n, d) where each row has been normalized to unit norm.
    """
    norms = mx.sqrt(mx.sum(x * x, axis=1, keepdims=True))
    # Avoid division by zero.
    norms = mx.where(norms > 0, norms, 1)
    return x / norms


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
    return mx.sum(diff * diff, axis=1)


###############################################################################
# End of File
###############################################################################
