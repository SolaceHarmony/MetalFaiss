from dataclasses import dataclass
from typing import Any, Dict

class ClusteringParameters:
    """Parameters for configuring clustering algorithms.
    
    This class holds all the parameters that control the behavior of clustering algorithms.
    Each parameter can be configured independently to customize the clustering process.
    """
    
    def __init__(
        self,
        niter: int = 25,
        nredo: int = 1,
        is_verbose: bool = False,
        is_spherical: bool = False,
        should_round_centroids_to_int: bool = False,
        should_update_index: bool = False,
        should_froze_centroids: bool = False,
        min_points_per_centroid: int = 39,
        max_points_per_centroid: int = 256,
        seed: int = 1234,
        decode_block_size: int = 32
    ):
        """Initialize clustering parameters with default values.
        
        Args:
            niter: Number of clustering iterations
            nredo: Number of clustering retries
            is_verbose: Enable verbose output
            is_spherical: Use spherical clustering
            should_round_centroids_to_int: Round centroids to integers
            should_update_index: Update index during clustering
            should_froze_centroids: Keep centroids fixed
            min_points_per_centroid: Minimum points per centroid
            max_points_per_centroid: Maximum points per centroid
            seed: Random seed
            decode_block_size: Block size for decoding
        """
        # Initialize with default values matching Swift version
        self._niter = niter
        self._nredo = nredo
        self._is_verbose = is_verbose
        self._is_spherical = is_spherical
        self._should_round_centroids_to_int = should_round_centroids_to_int
        self._should_update_index = should_update_index
        self._should_froze_centroids = should_froze_centroids
        self._min_points_per_centroid = min_points_per_centroid
        self._max_points_per_centroid = max_points_per_centroid
        self._seed = seed
        self._decode_block_size = decode_block_size

    @classmethod
    def default(cls) -> 'ClusteringParameters':
        """Create instance with default parameters."""
        return cls()

    # Property accessors renamed to match Swift version
    @property
    def is_verbose(self) -> bool:
        return self._is_verbose

    @is_verbose.setter
    def is_verbose(self, value: bool) -> None:
        self._is_verbose = value

    @property
    def is_spherical(self) -> bool:
        return self._is_spherical

    @is_spherical.setter
    def is_spherical(self, value: bool) -> None:
        self._is_spherical = value

    @property
    def should_round_centroids_to_int(self) -> bool:
        return self._should_round_centroids_to_int

    @should_round_centroids_to_int.setter
    def should_round_centroids_to_int(self, value: bool) -> None:
        self._should_round_centroids_to_int = value

    @property
    def should_update_index(self) -> bool:
        return self._should_update_index

    @should_update_index.setter
    def should_update_index(self, value: bool) -> None:
        self._should_update_index = value

    @property
    def should_froze_centroids(self) -> bool:
        return self._should_froze_centroids

    @should_froze_centroids.setter
    def should_froze_centroids(self, value: bool) -> None:
        self._should_froze_centroids = value

    @property
    def min_points_per_centroid(self) -> int:
        return self._min_points_per_centroid

    @min_points_per_centroid.setter
    def min_points_per_centroid(self, value: int) -> None:
        self._min_points_per_centroid = value

    @property
    def max_points_per_centroid(self) -> int:
        return self._max_points_per_centroid

    @max_points_per_centroid.setter
    def max_points_per_centroid(self, value: int) -> None:
        self._max_points_per_centroid = value

    @property
    def seed(self) -> int:
        return self._seed

    @seed.setter
    def seed(self, value: int) -> None:
        self._seed = value

    @property
    def decode_block_size(self) -> int:
        return self._decode_block_size

    @decode_block_size.setter
    def decode_block_size(self, value: int) -> None:
        self._decode_block_size = value
        
    @property
    def niter(self) -> int:
        return self._niter
        
    @niter.setter 
    def niter(self, value: int) -> None:
        self._niter = value
        
    @property
    def nredo(self) -> int:
        return self._nredo
        
    @nredo.setter
    def nredo(self, value: int) -> None:
        self._nredo = value

    def to_dict(self) -> Dict[str, Any]:
        """Convert parameters to a dictionary representation.
        
        Returns:
            Dictionary containing all parameter values
        """
        return {
            'niter': self._niter,
            'nredo': self._nredo,
            'is_verbose': self._is_verbose,
            'is_spherical': self._is_spherical,
            'should_round_centroids_to_int': self._should_round_centroids_to_int,
            'should_update_index': self._should_update_index,
            'should_froze_centroids': self._should_froze_centroids,
            'min_points_per_centroid': self._min_points_per_centroid,
            'max_points_per_centroid': self._max_points_per_centroid,
            'seed': self._seed,
            'decode_block_size': self._decode_block_size
        }
