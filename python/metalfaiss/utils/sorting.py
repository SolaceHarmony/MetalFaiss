# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the LICENSE file.

import mlx.core as mx
import numpy as np
from typing import Tuple, Optional
import multiprocessing

bucket_sort_verbose = 0

def fvec_argsort(vals: mx.array) -> mx.array:
    """Indirect sort of floating-point array.
    
    Args:
        vals: Array to sort
        
    Returns:
        Permutation array such that vals[perm[i + 1]] >= vals[perm[i]]
    """
    return mx.argsort(vals)

def fvec_argsort_parallel(vals: mx.array) -> mx.array:
    """Parallel indirect sort of floating-point array.
    
    Args:
        vals: Array to sort
        
    Returns:
        Permutation array such that vals[perm[i + 1]] >= vals[perm[i]]
    """
    # MLX handles parallelization internally
    return fvec_argsort(vals)

def bucket_sort(
    vals: np.ndarray,
    nbucket: int,
    num_threads: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """Bucket sort values into buckets.
    
    Args:
        vals: Values to sort, max value nbucket-1
        nbucket: Number of buckets
        num_threads: Number of threads (0 for sequential)
        
    Returns:
        Tuple of:
            lims: Bucket limits array
            perm: Sorted indices within buckets
    """
    nval = len(vals)
    if num_threads == 0:
        num_threads = 1
        
    # Count elements per bucket
    lims = np.zeros(nbucket + 1, dtype=np.int64)
    np.add.at(lims[1:], vals, 1)
    np.cumsum(lims, out=lims)
    
    # Sort elements into buckets
    perm = np.zeros(nval, dtype=np.int64)
    pos = lims[:-1].copy()
    
    for i in range(nval):
        bucket = vals[i]
        perm[pos[bucket]] = i
        pos[bucket] += 1
        
    return lims, perm

def matrix_bucket_sort_inplace(
    vals: np.ndarray,
    nbucket: int,
    num_threads: int = 0
) -> np.ndarray:
    """In-place bucket sort for matrix rows.
    
    Args:
        vals: Matrix of values to sort, shape (nrow, ncol)
        nbucket: Number of buckets
        num_threads: Number of threads (0 for sequential)
        
    Returns:
        Bucket limits array
    """
    nrow, ncol = vals.shape
    if num_threads == 0:
        num_threads = 1
        
    # Count elements per bucket
    lims = np.zeros(nbucket + 1, dtype=np.int64)
    unique, counts = np.unique(vals.reshape(-1), return_counts=True)
    lims[1:][unique] = counts
    np.cumsum(lims, out=lims)
    
    # Sort in place
    pos = lims[:-1].copy()
    temp = vals.copy()
    
    for i in range(nrow):
        for j in range(ncol):
            bucket = temp[i,j]
            vals[pos[bucket] // ncol, pos[bucket] % ncol] = i
            pos[bucket] += 1
            
    return lims

class Int64HashTable:
    """Hashtable for int64 -> int64 mapping with external storage."""
    
    def __init__(self, log2_capacity: int):
        """Initialize hashtable.
        
        Args:
            log2_capacity: Log2 of hashtable capacity
        """
        self.capacity = 1 << log2_capacity
        self.mask = self.capacity - 1
        self.table = np.zeros(2 * self.capacity, dtype=np.int64)
        self.table[::2] = -1  # Initialize keys to -1
        
    def add(self, keys: np.ndarray, vals: np.ndarray) -> None:
        """Add key-value pairs to hashtable.
        
        Args:
            keys: Keys to add
            vals: Values to add
        """
        n = len(keys)
        for i in range(n):
            key = keys[i]
            val = vals[i]
            
            # Linear probing
            idx = (key & self.mask) * 2
            while True:
                if self.table[idx] == -1 or self.table[idx] == key:
                    self.table[idx] = key
                    self.table[idx + 1] = val
                    break
                idx = (idx + 2) & (2 * self.capacity - 1)
                if idx == (key & self.mask) * 2:
                    raise RuntimeError("Hashtable capacity exhausted")
                    
    def lookup(self, keys: np.ndarray) -> np.ndarray:
        """Look up values for keys.
        
        Args:
            keys: Keys to look up
            
        Returns:
            Values for keys (-1 if not found)
        """
        n = len(keys)
        vals = np.full(n, -1, dtype=np.int64)
        
        for i in range(n):
            key = keys[i]
            idx = (key & self.mask) * 2
            
            # Linear probing
            while True:
                if self.table[idx] == -1:
                    break
                if self.table[idx] == key:
                    vals[i] = self.table[idx + 1]
                    break
                idx = (idx + 2) & (2 * self.capacity - 1)
                if idx == (key & self.mask) * 2:
                    break
                    
        return vals
