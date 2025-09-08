from abc import ABC, abstractmethod
from typing import List, Optional
import mlx.core as mx
from ..index.base_index import BaseIndex
from ..index_pointer import IndexPointer

class BaseClustering(ABC):
    """Base protocol for clustering algorithms."""
    
    def __init__(self, d: int, k: int):
        """Initialize clustering with dimension and number of clusters.
        
        Args:
            d: Vector dimension
            k: Number of clusters
        """
        self._d = d
        self._k = k
        self._niter = 25 
        self._nredo = 1
        self._verbose = False
        self._spherical = False
        self._int_centroids = False
        self._update_index = False
        self._frozen_centroids = False
        self._min_points_per_centroid = 39
        self._max_points_per_centroid = 256
        self._seed = 1234
        self._decode_block_size = 32
        self._centroids: Optional[mx.array] = None
        
    @property
    def k(self) -> int:
        """Number of clusters."""
        return self._k
        
    @property
    def d(self) -> int:
        """Vector dimension."""
        return self._d
        
    @property
    def niter(self) -> int:
        """Number of clustering iterations."""
        return self._niter
        
    @property
    def nredo(self) -> int:
        """Number of redo attempts."""
        return self._nredo
        
    @property
    def is_verbose(self) -> bool:
        """Verbose output flag."""
        return self._verbose
        
    @property
    def is_spherical(self) -> bool:
        """Spherical clustering flag."""
        return self._spherical
        
    @property
    def should_round_centroids_to_int(self) -> bool:
        """Integer centroid rounding flag."""
        return self._int_centroids
        
    @property
    def should_update_index(self) -> bool:
        """Index update flag."""
        return self._update_index
        
    @property
    def should_froze_centroids(self) -> bool:
        """Frozen centroids flag."""
        return self._frozen_centroids
        
    @property
    def min_points_per_centroid(self) -> int:
        """Minimum points per centroid."""
        return self._min_points_per_centroid
        
    @property
    def max_points_per_centroid(self) -> int:
        """Maximum points per centroid."""
        return self._max_points_per_centroid
        
    @property
    def seed(self) -> int:
        """Random seed."""
        return self._seed
        
    @property
    def decode_block_size(self) -> int:
        """Block size for decoding."""
        return self._decode_block_size

    @abstractmethod
    def train(self, xs: List[List[float]], index: Optional[BaseIndex] = None) -> None:
        """Train clustering algorithm.
        
        Args:
            xs: Training vectors
            index: Optional index for acceleration
            
        Raises:
            ValueError: If dimensions don't match
        """
        pass

    def centroids(self) -> mx.array:
        """Get computed centroids.
        
        Returns:
            List of centroid vectors
            
        Raises:
            RuntimeError: If clustering not trained
        """
        if self._centroids is None:
            raise RuntimeError("Clustering not trained")
        return self._centroids
