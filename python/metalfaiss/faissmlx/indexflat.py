# indexflat.py
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the MIT license (see LICENSE file in root directory).

import mlx.core as mx
import numpy as np
from typing import List, Optional, Tuple

from ..metric_type import MetricType
from ..search_result import SearchResult  # Assume a SearchResult class exists in the project.
from .distances import pairwise_L2sqr  # A function to compute pairwise squared L2 distances.

class FlatIndex:
    """
    FlatIndex stores full vectors and performs exhaustive search.
    This index does not require training since it stores full precision vectors.
    """
    def __init__(self, d: int, metric_type: MetricType = MetricType.L2) -> None:
        self.d = d
        self.metric_type = metric_type
        self.ntotal = 0
        # xb will store the database vectors as an MLX array of shape (ntotal, d)
        self.xb: Optional[mx.array] = None
        # Optional: store IDs for the vectors. If not provided, sequential IDs are used.
        self.ids: Optional[mx.array] = None

    def train(self, xs: List[List[float]]) -> None:
        """
        For FlatIndex, training is a no-op.
        """
        pass

    def add(self, xs: List[List[float]], ids: Optional[List[int]] = None) -> None:
        """
        Add vectors to the index.
        
        Args:
            xs: A list of vectors (each a list of floats).
            ids: Optional list of integer IDs. If not provided, sequential IDs are used.
        """
        # Convert the list of vectors to an MLX array
        new_vectors = mx.array(xs, dtype=mx.float32)
        n_new = new_vectors.shape[0]
        if self.xb is None:
            self.xb = new_vectors
            if ids is not None:
                self.ids = mx.array(ids, dtype=mx.int64)
            else:
                self.ids = mx.arange(n_new, dtype=mx.int64)
        else:
            self.xb = mx.concatenate([self.xb, new_vectors], axis=0)
            if ids is not None:
                new_ids = mx.array(ids, dtype=mx.int64)
            else:
                new_ids = mx.arange(n_new, dtype=mx.int64) + self.ntotal
            self.ids = mx.concatenate([self.ids, new_ids], axis=0)
        self.ntotal = int(self.xb.shape[0])
        # Force immediate evaluation
        mx.eval(self.xb)
        mx.eval(self.ids)

    def search(self, xs: List[List[float]], k: int) -> SearchResult:
        """
        Perform a k-nearest neighbor search.
        
        Args:
            xs: Query vectors as a list of lists.
            k: Number of neighbors to return.
        
        Returns:
            A SearchResult object containing distances and corresponding indices.
        """
        if self.xb is None or self.ntotal == 0:
            raise ValueError("Index is empty; add vectors before searching.")
        xq = mx.array(xs, dtype=mx.float32)
        # Compute pairwise distances based on the metric type.
        if self.metric_type == MetricType.L2:
            distances = pairwise_L2sqr(xq, self.xb)
        elif self.metric_type == MetricType.INNER_PRODUCT:
            # For inner product, higher is better.
            distances = -mx.matmul(xq, self.xb.T)
        elif self.metric_type == MetricType.L1:
            distances = mx.sum(mx.abs(xq.reshape((len(xq), 1, -1)) - self.xb), axis=2)
        elif self.metric_type == MetricType.LINF:
            distances = mx.max(mx.abs(xq.reshape((len(xq), 1, -1)) - self.xb), axis=2)
        else:
            raise ValueError(f"Unsupported metric type: {self.metric_type}")
        
        # Use MLX's topk function to retrieve k neighbors.
        if self.metric_type == MetricType.INNER_PRODUCT:
            values, indices = mx.topk(distances, k, axis=1)
        else:
            neg_distances = -distances
            values, indices = mx.topk(neg_distances, k, axis=1)
            values = -values
        # Map indices to the stored IDs.
        selected_ids = mx.take(self.ids, indices, axis=0)
        return SearchResult(distances=values, labels=selected_ids)

    def reconstruct(self, key: int) -> List[float]:
        """
        Retrieve the stored vector corresponding to the given key.
        
        Args:
            key: The index (or ID) of the vector to reconstruct.
        
        Returns:
            The vector as a Python list.
        """
        if self.xb is None or key < 0 or key >= self.ntotal:
            raise ValueError("Invalid key for reconstruction.")
        return self.xb[key].asnumpy().tolist()

    def reset(self) -> None:
        """
        Reset the index by clearing stored vectors and IDs.
        """
        self.xb = None
        self.ids = None
        self.ntotal = 0

    def save_to_file(self, filename: str) -> None:
        """
        Save the index to files. For example, we save the vector bank and IDs.
        """
        mx.save(self.xb, filename + "_xb")
        mx.save(self.ids, filename + "_ids")

    def clone(self) -> "FlatIndex":
        """
        Create a deep copy of the index.
        """
        new_index = FlatIndex(self.d, self.metric_type)
        if self.xb is not None:
            new_index.xb = mx.copy(self.xb)
            new_index.ids = mx.copy(self.ids)
            new_index.ntotal = self.ntotal
        return new_index

if __name__ == "__main__":
    # Test FlatIndex functionality.
    import random

    # Set dimension and create random vectors.
    d = 4
    num_vectors = 10
    num_queries = 3
    data = [[random.random() for _ in range(d)] for _ in range(num_vectors)]
    queries = [[random.random() for _ in range(d)] for _ in range(num_queries)]
    
    # Create a FlatIndex with L2 metric.
    index = FlatIndex(d=d, metric_type=MetricType.L2)
    # Training is a no-op for FlatIndex.
    index.train(data)
    # Add the vectors to the index.
    index.add(data)
    print("Total vectors in index:", index.ntotal)
    
    # Perform a search for k nearest neighbors.
    k = 3
    result = index.search(queries, k=k)
    print("Search Results:")
    print("Distances:")
    print(result.distances)
    print("Labels (IDs):")
    print(result.labels)
    
    # Test reconstruction of a stored vector.
    key = 5
    try:
        reconstructed = index.reconstruct(key)
        print(f"Reconstructed vector at key {key}:")
        print(reconstructed)
    except ValueError as e:
        print(f"Error in reconstruction: {e}")
    
    # Reset the index.
    index.reset()
    print("Index reset. Total vectors:", index.ntotal)