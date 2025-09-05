"""
refine_flat_index.py
------------------
A refined index that wraps a base index (e.g. FlatIndex) and
reorders search results by recomputing exact distances.
"""

import mlx.core as mx
from typing import List, Optional, Tuple

from ..types.metric_type import MetricType
from ..utils.search_result import SearchResult
from ..faissmlx.distances import pairwise_L2sqr
from .base_index import BaseIndex
from .flat_index import FlatIndex

class RefineFlatIndex(BaseIndex):
    """
    RefineFlatIndex refines search results from an underlying base index.
    
    It performs an initial search using the base index (with an expanded candidate count),
    then reconstructs candidate vectors and recomputes exact distances to re-sort the top k results.
    
    Currently supports L2 (squared Euclidean) and INNER_PRODUCT metrics.
    """
    def __init__(self, base_index: BaseIndex, refine_factor: float = 1.5) -> None:
        """
        Initialize the refined flat index.
        
        Args:
            base_index: A BaseIndex instance (typically a FlatIndex) that stores full vectors.
            refine_factor: Multiplier for the number of candidates to refine (default is 1.5).
        """
        self.base_index = base_index
        self.d = base_index.d
        self.metric_type = base_index.metric_type
        self.refine_factor = refine_factor

    @property
    def ntotal(self) -> int:
        return self.base_index.ntotal

    def train(self, xs: List[List[float]]) -> None:
        # Delegate training to the base index.
        self.base_index.train(xs)

    def add(self, xs: List[List[float]], ids: Optional[List[int]] = None) -> None:
        # Delegate vector addition to the base index.
        self.base_index.add(xs, ids)

    def search(self, xs: List[List[float]], k: int) -> SearchResult:
        """
        Perform a refined search: first retrieve candidate neighbors,
        then recompute exact distances on candidates and re-sort.
        
        Args:
            xs: Query vectors as a list of lists.
            k: Number of desired neighbors.
        
        Returns:
            A SearchResult with refined distances and corresponding labels.
        """
        if self.base_index.ntotal == 0:
            raise ValueError("Base index is empty; add vectors before searching.")

        # Expand the candidate count according to the refine factor.
        candidate_count = int(k * self.refine_factor)
        candidate_result = self.base_index.search(xs, candidate_count)
        # candidate_result is assumed to have fields 'distances' and 'labels' (both MLX arrays)

        # Convert queries to an MLX array.
        xq = mx.array(xs, dtype=mx.float32)  # shape: (nq, d)

        # For a FlatIndex, we expect self.base_index.xb to be available.
        # We will gather the candidate vectors from the base index.
        candidate_ids = candidate_result.labels  # shape: (nq, candidate_count)
        candidate_vectors = mx.take(self.base_index.xb, candidate_ids, axis=0)  
        # candidate_vectors has shape (nq, candidate_count, d)

        if self.metric_type == MetricType.L2:
            # For L2, compute pairwise squared distances.
            # Reshape query vectors to (nq, 1, d) for broadcasting.
            refined_distances = pairwise_L2sqr(xq.reshape((xq.shape[0], 1, self.d)),
                                          candidate_vectors)
            # refined_distances: shape (nq, candidate_count)
            # For L2, lower is better.
            neg_refined = -refined_distances
            values, new_order = mx.topk(neg_refined, k, axis=1)
            refined_values = -values
            refined_ids = mx.take(candidate_ids, new_order, axis=1)
            return SearchResult(distances=refined_values, labels=refined_ids)

        elif self.metric_type == MetricType.INNER_PRODUCT:
            # For inner product, higher is better.
            refined_distances = mx.matmul(xq.reshape((xq.shape[0], 1, self.d)),
                                     candidate_vectors.transpose((0, 2, 1)))
            # refined_distances: shape (nq, 1, candidate_count); squeeze axis=1.
            refined_distances = refined_distances.squeeze(axis=1)
            values, new_order = mx.topk(refined_distances, k, axis=1)
            refined_ids = mx.take(candidate_ids, new_order, axis=1)
            return SearchResult(distances=values, labels=refined_ids)
        else:
            raise ValueError(f"Refined flat index currently supports only L2 and INNER_PRODUCT metrics; got {self.metric_type}.")

    def reconstruct(self, key: int) -> List[float]:
        """
        Reconstruct a vector from the base index.
        """
        return self.base_index.reconstruct(key)

    def reset(self) -> None:
        """
        Reset the index.
        """
        self.base_index.reset()


if __name__ == "__main__":
    # Simple test block for refined flat index
    import random

    # Create random data: 200 vectors of dimension 16
    d = 16
    n = 200
    k = 5  # desired number of neighbors
    data = [[random.uniform(0, 1) for _ in range(d)] for _ in range(n)]
    
    # Create 5 random query vectors.
    nq = 5
    queries = [[random.uniform(0, 1) for _ in range(d)] for _ in range(nq)]
    
    # Create a FlatIndex with L2 metric and add data.
    base_index = FlatIndex(d=d, metric_type=MetricType.L2)
    base_index.add(data)
    
    # Wrap it with refined flat index.
    refined_index = RefineFlatIndex(base_index, refine_factor=1.5)
    
    # Perform refined search.
    result = refined_index.search(queries, k)
    print("Refined search distances:")
    print(result.distances)
    print("Refined search labels (IDs):")
    print(result.labels)
    
    # Test reconstruction.
    print("Reconstruct vector with key 0:")
    print(refined_index.reconstruct(0))