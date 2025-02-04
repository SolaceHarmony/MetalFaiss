# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the LICENSE file.

import mlx.core as mx
import numpy as np
from typing import Optional, Tuple

from .metric_type import MetricType
from .distances import pairwise_L2sqr
from .heap import float_maxheap_array_t

# TODO: Review implementations from:
# ✓ faiss/utils/extra_distances.h (interface)
# ✓ faiss/utils/extra_distances.cpp (implementations)
# - faiss/utils/distances_fused/ (fused operations)
# - faiss/impl/simd/distances.h (SIMD primitives)
# - faiss/impl/platform_macros.h (platform detection)
# - faiss/IndexNeuralNetCodec.h (learned metrics)

def pairwise_extra_distances(
    xq: mx.array,
    xb: mx.array,
    metric_type: MetricType,
    metric_arg: float = 0.0
) -> mx.array:
    """Compute pairwise distances with extra metrics.
    
    Args:
        xq: Query vectors (nq x d)
        xb: Database vectors (nb x d)
        metric_type: Type of distance metric
        metric_arg: Optional metric argument
        
    Returns:
        Distance matrix (nq x nb)
    """
    if metric_type == MetricType.L2:
        return pairwise_L2sqr(xq, xb)
    elif metric_type == MetricType.INNER_PRODUCT:
        return -mx.matmul(xq, xb.T)
    elif metric_type == MetricType.L1:
        # Compute L1 distances efficiently using broadcasting
        return mx.sum(mx.abs(xq.reshape(len(xq), 1, -1) - xb), axis=2)
    elif metric_type == MetricType.LINF:
        # Compute L-inf distances using broadcasting
        return mx.max(mx.abs(xq.reshape(len(xq), 1, -1) - xb), axis=2)
    else:
        raise ValueError(f"Unsupported metric type: {metric_type}")

def knn_extra_metrics(
    x: mx.array,
    y: mx.array,
    metric_type: MetricType,
    k: int,
    metric_arg: float = 0.0
) -> Tuple[mx.array, mx.array]:
    """K-nearest neighbor search with extra metrics.
    
    Args:
        x: Query vectors
        y: Database vectors
        metric_type: Type of distance metric
        k: Number of neighbors
        metric_arg: Optional metric argument
        
    Returns:
        Tuple of (distances, indices) arrays
    """
    distances = pairwise_extra_distances(x, y, metric_type, metric_arg)
    
    # For INNER_PRODUCT we want largest values, for others smallest
    if metric_type == MetricType.INNER_PRODUCT:
        values, indices = mx.topk(distances, k, axis=1)
    else:
        values, indices = mx.topk(-distances, k, axis=1)
        values = -values
        
    return values, indices
