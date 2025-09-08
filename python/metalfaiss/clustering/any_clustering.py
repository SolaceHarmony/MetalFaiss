import mlx.core as mx
from typing import List, Optional
from dataclasses import dataclass

from ..index_pointer import IndexPointer
from .base_clustering import BaseClustering
from .clustering_parameters import ClusteringParameters
from ..index.base_index import BaseIndex

@dataclass
class ClusteringParameters:
    max_iterations: int = 20
    tolerance: float = 1e-4
    spherical: bool = False
    seed: Optional[int] = None

class BaseClustering:
    def __init__(self, d: int, k: int):
        self.d = d  # dimensionality
        self.k = k  # number of clusters
        self.centroids_: Optional[mx.array] = None

    def train(self, x: List[List[float]]) -> None:
        raise NotImplementedError

    def centroids(self) -> mx.array:
        if self.centroids_ is None:
            raise ValueError("Clustering not trained yet")
        return self.centroids_

class AnyClustering(BaseClustering):
    """Clustering interface that matches the Swift Faiss implementation."""
    
    def __init__(self, index_pointer: Optional['IndexPointer'] = None, parameters: Optional[ClusteringParameters] = None):
        """Initialize clustering with optional index pointer and parameters.
        
        Args:
            index_pointer: Optional pointer to index (for Swift compatibility)
            parameters: Optional clustering parameters
        """
        super().__init__()
        self.parameters = parameters or ClusteringParameters()
        self._centroids = None
        
    @classmethod
    def new(cls, d: int, k: int, parameters: Optional[ClusteringParameters] = None) -> 'AnyClustering':
        """Create new clustering instance.
        
        Args:
            d: Dimension of vectors to cluster
            k: Number of clusters
            parameters: Optional clustering parameters
            
        Returns:
            New AnyClustering instance
        """
        instance = cls(parameters=parameters)
        instance.d = d
        instance.k = k
        return instance
    
    def train(self, xs: List[List[float]], index: Optional[BaseIndex] = None) -> None:
        """Train the clustering model.
        
        Args:
            xs: Training vectors
            index: Optional index for acceleration
            
        Raises:
            ValueError: If input dimensions don't match
        """
        if not xs:
            raise ValueError("Empty training data")
            
        x = mx.array(xs, dtype=mx.float32)
        if x.shape[1] != self.d:
            raise ValueError(f"Data dimension {x.shape[1]} does not match index dimension {self.d}")
            
        # Initialize centroids using k-means++
        with mx.stream():
            # Select first centroid randomly
            idx = [mx.random.randint(0, len(x), (1,))[0]]
            centroids = [x[idx[0]]]
            
            # Choose remaining centroids
            for _ in range(1, self.k):
                # Compute distances to existing centroids
                distances = mx.min(
                    mx.sum(mx.square(mx.subtract(x[:, None], mx.stack(centroids))), axis=-1),
                    axis=-1
                )
                # Select next centroid probabilistically
                probs = mx.divide(distances, mx.sum(distances))
                next_idx = mx.random.choice(len(x), p=probs)
                idx.append(next_idx)
                centroids.append(x[next_idx])
            
            self._centroids = mx.stack(centroids)
            
            # k-means iterations
            for _ in range(self.parameters.max_points):
                old_centroids = self._centroids
                
                # Compute assignments
                dists = mx.sum(mx.square(mx.subtract(x[:, None], self._centroids[None])), axis=2)
                labels = mx.argmin(dists, axis=1)
                
                # Update centroids (device-only; avoid Python scalar checks)
                new_centroids = []
                for j in range(self.k):
                    mask = mx.equal(labels, mx.array(j, dtype=labels.dtype))
                    count = mx.sum(mask)
                    # Safe mean: sum / max(count, 1)
                    sumj = mx.sum(mx.where(mask[:, None], x, mx.zeros_like(x)), axis=0)
                    denom = mx.maximum(count, mx.array(1, dtype=count.dtype))
                    meanj = mx.divide(sumj, denom)
                    use_mean = mx.greater(count, mx.array(0, dtype=count.dtype))
                    new_centroids.append(mx.where(use_mean, meanj, self._centroids[j]))
                
                self._centroids = mx.stack(new_centroids)
                
                # Optional early stop removed to avoid host pulls; run fixed iterations
                    
            mx.eval(self._centroids)

    def centroids(self) -> mx.array:
        """Get computed centroids.
        
        Returns:
            List of centroid vectors
        """
        if self._centroids is None:
            raise RuntimeError("Clustering not trained")
        return self._centroids

def kmeans_clustering(
    xs: List[List[float]],
    d: int,
    k: int,
    params: Optional[ClusteringParameters] = None
) -> List[List[float]]:
    """Simplified interface for k-means clustering."""
    clustering = AnyClustering.new(d=d, k=k, parameters=params)
    clustering.train(xs)
    return clustering.centroids()
