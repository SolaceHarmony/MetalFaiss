# ordering.py
# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the LICENSE file.

import mlx.core as mx
import numpy as np
from typing import Tuple

def fvec_argsort(x: mx.array, axis: int = -1) -> mx.array:
    """
    Return the indices that would sort the array x along the specified axis.
    
    Args:
        x (mx.array): Input array.
        axis (int): Axis along which to sort (default is last axis).
        
    Returns:
        mx.array: Array of indices that would sort x.
    """
    # MLX provides a vectorized argsort that works similarly to NumPy's.
    return mx.argsort(x, axis=axis)

def fvec_argsort_parallel(x: mx.array, axis: int = -1) -> mx.array:
    """
    Parallel argsort of array x along the given axis.
    
    For MLX, since array operations are already vectorized and JIT‐compiled,
    this function simply calls mx.argsort.
    
    Args:
        x (mx.array): Input array.
        axis (int): Axis along which to sort.
        
    Returns:
        mx.array: Sorted indices.
    """
    # In MLX, parallelism is built into the array operations.
    return mx.argsort(x, axis=axis)

def bucket_sort(x: mx.array, num_buckets: int) -> Tuple[mx.array, mx.array]:
    """
    Perform a bucket sort on the 1D array x.
    
    This is a simple implementation that converts the MLX array to NumPy,
    performs a sort (using argsort), and converts the results back to MLX arrays.
    (In a production version you might implement a fully MLX‐native version.)
    
    Args:
        x (mx.array): 1D input array.
        num_buckets (int): Number of buckets (not used in this simple implementation,
                           but kept for interface compatibility).
    
    Returns:
        Tuple[mx.array, mx.array]: A tuple (sorted_values, sorted_indices).
    """
    # Convert to NumPy (MLX arrays are lazy; this forces evaluation)
    x_np = x.asnumpy()
    indices = np.argsort(x_np)
    sorted_values = x_np[indices]
    # Wrap results back into MLX arrays.
    return mx.array(sorted_values, dtype=x.dtype), mx.array(indices, dtype=mx.int64)

def matrix_bucket_sort_inplace(x: mx.array) -> mx.array:
    """
    Sort each row of a 2D array using MLX's built-in sort.
    
    Note: In MLX arrays are immutable in the sense that operations return new arrays.
    Therefore, this function returns a new array with each row sorted.
    
    Args:
        x (mx.array): A 2D array to sort along the last axis.
    
    Returns:
        mx.array: A new 2D array with each row sorted in ascending order.
    """
    # Use MLX's sort function along axis 1.
    sorted_x = mx.sort(x, axis=1)
    # Optionally, force immediate evaluation:
    mx.eval(sorted_x)
    return sorted_x

# Example usage and simple test:
if __name__ == "__main__":
    # Create a simple 1D array and sort it.
    a = mx.array([3.0, 1.0, 2.0], dtype=mx.float32)
    idx = fvec_argsort(a)
    sorted_a = a[idx]
    # Force evaluation (since MLX is lazy) so that we see immediate results.
    sorted_a = mx.eval(sorted_a)
    
    print("Original array:", a.asnumpy())
    print("Argsort indices:", idx.asnumpy())
    print("Sorted array:", sorted_a.asnumpy())