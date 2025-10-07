"""
binary_search_flat.py - Binary search operations for flat indices

Implements warp-style cooperative binary search and flat vector search
optimized for Apple Silicon Metal. All operations stay on GPU.

Based on advanced MLX patterns and Metal optimization techniques.
"""

import mlx.core as mx
from typing import Optional, Tuple


@mx.compile
def lower_bound_branchless(sorted_keys: mx.array, queries: mx.array) -> mx.array:
    """Find lower bound indices using branchless binary search.
    
    For each query, returns the index of the first element >= query.
    Uses binary lifting technique for completely branchless execution.
    
    Args:
        sorted_keys: Sorted array of keys, shape (N,)
        queries: Query values, shape (Q,)
        
    Returns:
        Indices array, shape (Q,). Value is N if all keys < query.
        
    Example:
        keys = mx.array([2, 5, 9, 14, 20], dtype=mx.int32)
        queries = mx.array([0, 3, 14, 100], dtype=mx.int32)
        idx = lower_bound_branchless(keys, queries)
        # Result: [0, 1, 3, 5] (indices of first element >= query)
    """
    n_elem = sorted_keys.shape[0]
    q_count = queries.shape[0]
    
    # Start at -1 (conceptually before first element)
    pos = mx.full((q_count,), mx.array(-1, dtype=mx.int32), dtype=mx.int32)
    
    # Compute largest power of 2 <= n
    n_bits = mx.array(32, dtype=mx.int32)
    leading_zeros = mx.zeros((1,), dtype=mx.int32)  # TODO: implement clz
    step_bits = mx.subtract(n_bits, leading_zeros)
    step_bits = mx.maximum(step_bits, mx.array(0, dtype=mx.int32))
    
    # Start with largest power of 2
    step = mx.array(1, dtype=mx.int32)
    max_steps = mx.array(32, dtype=mx.int32)  # Log2 bound
    
    # Binary lifting loop
    for i in range(32):  # Max 32 iterations for any practical size
        candidate = mx.add(pos, step)
        
        # Check if candidate is in valid range
        in_range = mx.less(candidate, mx.array(n_elem, dtype=mx.int32))
        
        # Fetch keys at candidate positions (use large value for out-of-range)
        inf_val = mx.array(2147483647, dtype=sorted_keys.dtype)  # Max int32
        broadcast_inf = mx.broadcast_to(inf_val, (q_count,))
        
        # Safe indexing: clamp to valid range
        safe_idx = mx.where(
            in_range,
            candidate,
            mx.zeros_like(candidate)
        )
        cand_keys = mx.take(sorted_keys, safe_idx, axis=0)
        cand_keys = mx.where(in_range, cand_keys, broadcast_inf)
        
        # Advance if candidate key < query (lower_bound semantics)
        should_advance = mx.less(cand_keys, queries)
        pos = mx.where(should_advance, candidate, pos)
        
        # Shift step right
        step = mx.right_shift(step, mx.array(1, dtype=mx.int32))
        
        # Early exit if step becomes 0
        is_zero = mx.equal(step, mx.array(0, dtype=mx.int32))
        if mx.all(is_zero):
            break
    
    # lower_bound is pos + 1
    return mx.add(pos, mx.array(1, dtype=mx.int32))


@mx.compile
def find_leq_index(sorted_keys: mx.array, queries: mx.array) -> mx.array:
    """Find index of largest key <= query.
    
    Args:
        sorted_keys: Sorted array, shape (N,)
        queries: Query values, shape (Q,)
        
    Returns:
        Indices, shape (Q,). Value is -1 if no key <= query.
    """
    lb = lower_bound_branchless(sorted_keys, queries)
    idx = mx.subtract(lb, mx.array(1, dtype=mx.int32))
    
    # Return -1 for invalid indices
    neg_one = mx.array(-1, dtype=mx.int32)
    broadcast_neg = mx.broadcast_to(neg_one, idx.shape)
    valid = mx.greater_equal(idx, mx.array(0, dtype=mx.int32))
    
    return mx.where(valid, idx, broadcast_neg)


@mx.compile  
def flat_knn_inner_product(
    database: mx.array,
    queries: mx.array,
    k: mx.array,
    tile_size: Optional[mx.array] = None
) -> Tuple[mx.array, mx.array]:
    """Flat KNN search using inner product similarity.
    
    Computes similarity scores using matrix multiplication and returns
    top-k results. All computation stays on GPU.
    
    Args:
        database: Database vectors, shape (N, d)
        queries: Query vectors, shape (Q, d)
        k: Number of neighbors to return (scalar)
        tile_size: Optional tile size for large databases
        
    Returns:
        Tuple of (indices, scores), each shape (Q, k)
        Indices are sorted by descending similarity score.
    """
    # Compute similarity: queries @ database.T
    # Shape: (Q, d) @ (d, N) = (Q, N)
    scores = mx.matmul(queries, mx.transpose(database))
    
    # Get top-k along database dimension
    # topk returns (values, indices) in descending order
    top_scores, top_indices = mx.topk(scores, k, axis=1)
    
    return top_indices, top_scores


@mx.compile
def flat_knn_l2_distance(
    database: mx.array,
    queries: mx.array,
    k: mx.array,
    tile_size: Optional[mx.array] = None
) -> Tuple[mx.array, mx.array]:
    """Flat KNN search using L2 distance.
    
    Uses the identity: ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a.b
    to compute distances efficiently.
    
    Args:
        database: Database vectors, shape (N, d)
        queries: Query vectors, shape (Q, d)  
        k: Number of neighbors to return (scalar)
        tile_size: Optional tile size for large databases
        
    Returns:
        Tuple of (indices, distances), each shape (Q, k)
        Indices are sorted by ascending distance.
    """
    # Compute query norms: ||q||^2
    # Shape: (Q, d) -> (Q, 1)
    q_norms = mx.sum(
        mx.multiply(queries, queries),
        axis=1,
        keepdims=True
    )
    
    # Compute database norms: ||d||^2
    # Shape: (N, d) -> (N,) -> (1, N)
    d_norms = mx.sum(
        mx.multiply(database, database),
        axis=1
    )
    d_norms = mx.reshape(d_norms, (mx.array(1, dtype=mx.int32), mx.array(-1, dtype=mx.int32)))
    
    # Compute cross term: 2 * queries @ database.T
    # Shape: (Q, d) @ (d, N) = (Q, N)
    cross = mx.matmul(queries, mx.transpose(database))
    cross = mx.multiply(cross, mx.array(2, dtype=cross.dtype))
    
    # Distance: ||q||^2 + ||d||^2 - 2*q.d
    # Broadcasting: (Q, 1) + (1, N) - (Q, N) = (Q, N)
    distances = mx.add(q_norms, d_norms)
    distances = mx.subtract(distances, cross)
    
    # For numerical stability, clamp to non-negative
    zero = mx.array(0, dtype=distances.dtype)
    distances = mx.maximum(distances, zero)
    
    # Get top-k with smallest distances
    # Since topk returns largest, negate distances
    neg_distances = mx.negative(distances)
    top_neg_dist, top_indices = mx.topk(neg_distances, k, axis=1)
    
    # Negate back to get actual distances
    top_distances = mx.negative(top_neg_dist)
    
    return top_indices, top_distances


def build_flat_index(
    embeddings: mx.array,
    metric: str = "ip"
) -> dict:
    """Build a flat index for exact nearest neighbor search.
    
    Args:
        embeddings: Embedding vectors, shape (N, d)
        metric: Distance metric, either "ip" (inner product) or "l2"
        
    Returns:
        Dictionary with index metadata
    """
    # Precompute norms for L2 distance
    norms = None
    if metric == "l2":
        norms = mx.sum(
            mx.multiply(embeddings, embeddings),
            axis=1
        )
    
    return {
        "embeddings": embeddings,
        "metric": metric,
        "norms": norms,
        "n_vectors": mx.array(embeddings.shape[0], dtype=mx.int32),
        "dimension": mx.array(embeddings.shape[1], dtype=mx.int32)
    }


@mx.compile
def search_flat_index(
    index: dict,
    queries: mx.array,
    k: mx.array
) -> Tuple[mx.array, mx.array]:
    """Search a flat index for k nearest neighbors.
    
    Args:
        index: Index dictionary from build_flat_index
        queries: Query vectors, shape (Q, d)
        k: Number of neighbors (scalar)
        
    Returns:
        Tuple of (indices, scores/distances), each shape (Q, k)
    """
    if index["metric"] == "ip":
        return flat_knn_inner_product(
            index["embeddings"],
            queries,
            k
        )
    else:  # L2
        return flat_knn_l2_distance(
            index["embeddings"],
            queries,
            k
        )


# TODO: Implement tiled search for very large databases
# TODO: Implement hierarchical (block-based) binary search
# TODO: Implement median-split binary tree traversal
# TODO: Add range search (find all within distance threshold)
# TODO: Add batch processing for memory-constrained scenarios
