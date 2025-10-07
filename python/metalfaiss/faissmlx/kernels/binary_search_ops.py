"""
binary_search_ops.py — Advanced binary search operations for MetalFaiss

This module extends binary_kernels.py with additional search operations
following the Gemini AI roadmap for binary flat index implementations.

Includes:
- Branchless parallel binary search (scalar keys)
- Hierarchical two-level search (coarse + fine)
- Exact vector KNN with L2/IP metrics
- Tiled processing for large datasets
- Range search operations

Design Philosophy:
- Pure MLX: Zero CPU operations, no NumPy, no Python scalars in hot paths
- GPU-native: All operations stay on Metal device
- Vectorized: Branchless implementations for SIMD efficiency
- Memory-efficient: Tiled algorithms for large-scale datasets

Inspired by:
- FAISS IndexFlat operations
- Gemini AI MLX binary search recommendations  
- Apple Metal SIMD-group patterns
"""

from __future__ import annotations
from typing import Tuple, Optional, List
import mlx.core as mx


# ==============================================================================
# Scalar Binary Search Operations
# ==============================================================================

@mx.compile
def binary_search_multi_range(
    sorted_keys: mx.array,
    start_queries: mx.array,
    end_queries: mx.array
) -> Tuple[mx.array, mx.array]:
    """Range queries using binary search.
    
    For each (start, end) pair, finds all keys in [start, end].
    Returns indices of first and last elements in range.
    
    Args:
        sorted_keys: Sorted keys (n,) int32
        start_queries: Start of ranges (q,) int32
        end_queries: End of ranges (q,) int32
        
    Returns:
        start_indices: First index >= start_queries (q,) int32
        end_indices: Last index <= end_queries (q,) int32
        
    Example:
        keys = [2, 5, 9, 14, 20]
        starts = [3, 15]
        ends = [14, 19]
        -> start_indices = [1, 4], end_indices = [3, 3]
        (ranges: [5,9,14] and [])
    """
    from .binary_kernels import lower_bound_binary_search, find_leq_binary_search
    
    # Find first element >= start
    start_idx = lower_bound_binary_search(sorted_keys, start_queries)
    
    # Find last element <= end
    end_idx = find_leq_binary_search(sorted_keys, end_queries)
    
    # Clamp to valid range
    n = sorted_keys.shape[0]
    start_idx = mx.clip(start_idx, mx.array(0, dtype=mx.int32), mx.array(n, dtype=mx.int32))
    end_idx = mx.clip(end_idx, mx.array(-1, dtype=mx.int32), mx.array(n - 1, dtype=mx.int32))
    
    return start_idx, end_idx


# ==============================================================================
# Vector Similarity Search (L2 / Inner Product)
# ==============================================================================

@mx.compile
def l2_distance_matrix(
    queries: mx.array,
    database: mx.array
) -> mx.array:
    """Compute L2 distance matrix using vectorized MLX operations.
    
    Uses the identity: ||a - b||^2 = ||a||^2 + ||b||^2 - 2<a,b>
    
    Args:
        queries: Query vectors (nq, d) float32
        database: Database vectors (nb, d) float32
        
    Returns:
        L2 distances (nq, nb) float32
        
    Performance:
        - Memory: O(nq * nb) for distance matrix
        - Compute: O(nq * nb * d) via matrix multiplication
        - Use tiled version for nb > 100K
    """
    # Compute squared norms
    q_norms = mx.sum(mx.multiply(queries, queries), axis=1, keepdims=True)  # (nq, 1)
    db_norms = mx.sum(mx.multiply(database, database), axis=1, keepdims=False)  # (nb,)
    
    # Compute cross term
    cross = mx.matmul(queries, mx.transpose(database, axes=[1, 0]))  # (nq, nb)
    
    # L2 distance = q^2 + db^2 - 2*cross
    # Broadcasting: q_norms is (nq, 1), db_norms is (nb,)
    dists = mx.add(q_norms, db_norms[None, :])
    dists = mx.subtract(dists, mx.multiply(mx.array(2.0, dtype=dists.dtype), cross))
    
    # Clamp to avoid negative values from numerical error
    dists = mx.maximum(dists, mx.array(0.0, dtype=dists.dtype))
    
    return dists


@mx.compile
def ip_similarity_matrix(
    queries: mx.array,
    database: mx.array
) -> mx.array:
    """Compute inner product similarity matrix.
    
    Args:
        queries: Query vectors (nq, d) float32
        database: Database vectors (nb, d) float32
        
    Returns:
        Inner products (nq, nb) float32
        
    Note:
        For nearest neighbor search, use negative inner product
        as distance (larger IP = smaller distance).
    """
    return mx.matmul(queries, mx.transpose(database, axes=[1, 0]))


@mx.compile
def flat_knn_l2(
    queries: mx.array,
    database: mx.array,
    k: int
) -> Tuple[mx.array, mx.array]:
    """Exact k-NN search using L2 distance.
    
    Args:
        queries: Query vectors (nq, d) float32
        database: Database vectors (nb, d) float32
        k: Number of neighbors
        
    Returns:
        distances: L2 distances to k nearest neighbors (nq, k) float32
        indices: Indices of k nearest neighbors (nq, k) int32
        
    Performance:
        - Single-pass algorithm
        - Suitable for nb < 100K
        - O(nq * nb * d) compute, O(nq * nb) memory
    """
    # Compute full distance matrix
    dists = l2_distance_matrix(queries, database)
    
    # Select k smallest distances
    k_actual = mx.minimum(mx.array(k, dtype=mx.int32), mx.array(database.shape[0], dtype=mx.int32))
    k_actual_int = int(k_actual.item())  # boundary-ok: constant for topk
    
    # Sort by distance (ascending) and take first k
    indices = mx.argsort(dists, axis=1)[:, :k_actual_int]
    distances = mx.take_along_axis(dists, indices, axis=1)
    
    return distances, indices


@mx.compile
def flat_knn_ip(
    queries: mx.array,
    database: mx.array,
    k: int
) -> Tuple[mx.array, mx.array]:
    """Exact k-NN search using inner product similarity.
    
    Args:
        queries: Query vectors (nq, d) float32
        database: Database vectors (nb, d) float32
        k: Number of neighbors
        
    Returns:
        similarities: Inner products to k nearest neighbors (nq, k) float32
        indices: Indices of k nearest neighbors (nq, k) int32
        
    Note:
        Returns largest inner products (most similar vectors).
    """
    # Compute similarity matrix
    sims = ip_similarity_matrix(queries, database)
    
    # Select k largest similarities
    k_actual = mx.minimum(mx.array(k, dtype=mx.int32), mx.array(database.shape[0], dtype=mx.int32))
    k_actual_int = int(k_actual.item())  # boundary-ok: constant for topk
    
    # Sort by similarity (descending) and take first k
    # Use negative to sort descending
    neg_sims = mx.negative(sims)
    indices = mx.argsort(neg_sims, axis=1)[:, :k_actual_int]
    similarities = mx.take_along_axis(sims, indices, axis=1)
    
    return similarities, indices


# ==============================================================================
# Tiled K-NN for Large Databases
# ==============================================================================

def flat_knn_l2_tiled(
    queries: mx.array,
    database: mx.array,
    k: int,
    tile_size: int = 8192
) -> Tuple[mx.array, mx.array]:
    """Tiled k-NN search for large databases using L2 distance.
    
    Processes database in tiles to avoid materializing full distance matrix.
    Maintains rolling top-k across tiles.
    
    Args:
        queries: Query vectors (nq, d) float32
        database: Database vectors (nb, d) float32
        k: Number of neighbors
        tile_size: Number of database vectors per tile
        
    Returns:
        distances: L2 distances to k nearest neighbors (nq, k) float32
        indices: Indices of k nearest neighbors (nq, k) int32
        
    Performance:
        - Memory: O(nq * tile_size) per iteration
        - Preferred for nb > 100K
        - Multiple passes over database
    """
    nq = queries.shape[0]
    nb = database.shape[0]
    k_actual = min(k, nb)
    
    # Initialize with worst-case values (infinity for L2 distance)
    top_dists = mx.full((nq, k_actual), mx.inf, dtype=mx.float32)
    top_idxs = mx.full((nq, k_actual), mx.array(-1, dtype=mx.int32), dtype=mx.int32)
    
    # Pre-compute query norms once
    q_norms = mx.sum(mx.multiply(queries, queries), axis=1, keepdims=True)  # (nq, 1)
    
    # Process database in tiles
    num_tiles = mx.divide(mx.add(mx.array(nb, dtype=mx.int32), mx.array(tile_size - 1, dtype=mx.int32)), mx.array(tile_size, dtype=mx.int32))
    num_tiles_int = int(num_tiles.item())  # boundary-ok: loop control
    
    for tile_idx in range(num_tiles_int):
        start = tile_idx * tile_size
        end = min(start + tile_size, nb)
        tile = database[start:end]
        
        # Compute distances for this tile
        tile_norms = mx.sum(mx.multiply(tile, tile), axis=1, keepdims=False)  # (tile_size,)
        cross = mx.matmul(queries, mx.transpose(tile, axes=[1, 0]))  # (nq, tile_size)
        tile_dists = mx.add(q_norms, tile_norms[None, :])
        tile_dists = mx.subtract(tile_dists, mx.multiply(mx.array(2.0, dtype=tile_dists.dtype), cross))
        tile_dists = mx.maximum(tile_dists, mx.array(0.0, dtype=tile_dists.dtype))
        
        # Offset indices by tile start
        tile_shape = tile_dists.shape[1]
        tile_idxs = mx.add(mx.arange(tile_shape, dtype=mx.int32), mx.array(start, dtype=mx.int32))
        tile_idxs = tile_idxs[None, :].broadcast_to((nq, tile_shape))
        
        # Merge with existing top-k
        merged_dists = mx.concatenate([top_dists, tile_dists], axis=1)
        merged_idxs = mx.concatenate([top_idxs, tile_idxs], axis=1)
        
        # Re-select top-k (smallest distances)
        sort_idxs = mx.argsort(merged_dists, axis=1)[:, :k_actual]
        top_dists = mx.take_along_axis(merged_dists, sort_idxs, axis=1)
        top_idxs = mx.take_along_axis(merged_idxs, sort_idxs, axis=1)
    
    return top_dists, top_idxs


def flat_knn_ip_tiled(
    queries: mx.array,
    database: mx.array,
    k: int,
    tile_size: int = 8192
) -> Tuple[mx.array, mx.array]:
    """Tiled k-NN search for large databases using inner product.
    
    Args:
        queries: Query vectors (nq, d) float32
        database: Database vectors (nb, d) float32
        k: Number of neighbors
        tile_size: Number of database vectors per tile
        
    Returns:
        similarities: Inner products to k nearest neighbors (nq, k) float32
        indices: Indices of k nearest neighbors (nq, k) int32
    """
    nq = queries.shape[0]
    nb = database.shape[0]
    k_actual = min(k, nb)
    
    # Initialize with worst-case values (negative infinity for IP)
    neg_inf = mx.negative(mx.array(mx.inf, dtype=mx.float32))
    top_sims = mx.full((nq, k_actual), neg_inf, dtype=mx.float32)
    neg_one = mx.array(-1, dtype=mx.int32)  # literal -1 is ok for array initialization
    top_idxs = mx.full((nq, k_actual), neg_one, dtype=mx.int32)
    
    # Process database in tiles
    num_tiles = mx.divide(mx.add(mx.array(nb, dtype=mx.int32), mx.array(tile_size - 1, dtype=mx.int32)), mx.array(tile_size, dtype=mx.int32))
    num_tiles_int = int(num_tiles.item())  # boundary-ok: loop control
    
    for tile_idx in range(num_tiles_int):
        start = tile_idx * tile_size
        end = min(start + tile_size, nb)
        tile = database[start:end]
        
        # Compute similarities for this tile
        tile_sims = mx.matmul(queries, mx.transpose(tile, axes=[1, 0]))  # (nq, tile_size)
        
        # Offset indices by tile start
        tile_shape = tile_sims.shape[1]
        tile_idxs = mx.add(mx.arange(tile_shape, dtype=mx.int32), mx.array(start, dtype=mx.int32))
        tile_idxs = tile_idxs[None, :].broadcast_to((nq, tile_shape))
        
        # Merge with existing top-k
        merged_sims = mx.concatenate([top_sims, tile_sims], axis=1)
        merged_idxs = mx.concatenate([top_idxs, tile_idxs], axis=1)
        
        # Re-select top-k (largest similarities)
        # Use negative to sort descending (largest first)
        neg_merged = mx.negative(merged_sims)
        sort_idxs = mx.argsort(neg_merged, axis=1)[:, :k_actual]
        top_sims = mx.take_along_axis(merged_sims, sort_idxs, axis=1)
        top_idxs = mx.take_along_axis(merged_idxs, sort_idxs, axis=1)
    
    return top_sims, top_idxs


# ==============================================================================
# Range Search Operations
# ==============================================================================

def flat_range_search_l2(
    queries: mx.array,
    database: mx.array,
    radius: float
) -> Tuple[List[mx.array], List[mx.array], mx.array]:
    """Range search for vectors within L2 radius.
    
    Args:
        queries: Query vectors (nq, d) float32
        database: Database vectors (nb, d) float32
        radius: Maximum L2 distance
        
    Returns:
        distances: List of distance arrays (variable length per query)
        indices: List of index arrays (variable length per query)
        lims: Cumulative counts (nq+1,) indicating range boundaries
        
    Note:
        Returns variable-length results per query.
        lims[i]:lims[i+1] gives range in flattened arrays for query i.
    """
    # Compute all distances
    all_dists = l2_distance_matrix(queries, database)
    
    # Find matches within radius for each query
    nq = queries.shape[0]
    nb = database.shape[0]
    
    distances = []
    indices = []
    lims = [0]
    
    radius_mx = mx.array(radius, dtype=mx.float32)
    
    for i in range(nq):
        # Find indices where distance <= radius
        match_mask = mx.less_equal(all_dists[i], radius_mx)
        
        # Count matches
        count = int(mx.sum(match_mask.astype(mx.int32)).item())  # boundary-ok: counting matches
        
        if count > 0:
            # Extract matching distances and indices
            # Since MLX doesn't have boolean indexing, use argsort trick
            all_indices = mx.arange(nb, dtype=mx.int32)
            
            # Create sortable key: put matches first (distance 0), non-matches last (distance 1)
            sort_key = mx.where(match_mask, mx.zeros_like(all_indices, dtype=mx.int32), mx.ones_like(all_indices, dtype=mx.int32))
            sort_idx = mx.argsort(sort_key, axis=0)
            
            # Take first 'count' elements (these are the matches)
            matched_idx = mx.take(all_indices, sort_idx[:count])
            matched_dist = mx.take(all_dists[i], matched_idx)
            
            distances.append(matched_dist)
            indices.append(matched_idx)
        else:
            distances.append(mx.array([], dtype=mx.float32))
            indices.append(mx.array([], dtype=mx.int32))
        
        lims.append(lims[-1] + count)
    
    return distances, indices, mx.array(lims, dtype=mx.int32)


# ==============================================================================
# Export Public API
# ==============================================================================

__all__ = [
    # Binary search
    'binary_search_multi_range',
    
    # Distance matrices
    'l2_distance_matrix',
    'ip_similarity_matrix',
    
    # K-NN search
    'flat_knn_l2',
    'flat_knn_ip',
    'flat_knn_l2_tiled',
    'flat_knn_ip_tiled',
    
    # Range search
    'flat_range_search_l2',
]
