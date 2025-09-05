"""
opq.py - Optimized Product Quantization transform for MetalFaiss
"""

import numpy as np
import mlx.core as mx
from typing import Optional, Tuple
from .base_vector_transform import BaseVectorTransform
from .pca_matrix import PCAMatrixTransform

class OPQTransform(BaseVectorTransform):
    """Optimized Product Quantization transform.
    
    This transform learns a rotation matrix that minimizes quantization error
    when using product quantization. It alternates between:
    1. Learning sub-quantizers given rotation
    2. Learning rotation given sub-quantizers
    """
    
    def __init__(
        self,
        d_in: int,
        M: int,
        n_iter: int = 25,
        n_iter_pq: int = 25,
        random_rotation: bool = True,
        seed: Optional[int] = None
    ):
        """Initialize OPQ transform.
        
        Args:
            d_in: Input dimension
            M: Number of sub-quantizers
            n_iter: Number of OPQ iterations
            n_iter_pq: Number of k-means iterations for PQ
            random_rotation: Whether to apply random rotation initially
            seed: Random seed (default: None)
        """
        # OPQ preserves dimension: d_out == d_in
        super().__init__(d_in, d_in)
        if d_in % M != 0:
            raise ValueError(f"Input dimension {d_in} must be divisible by M={M}")
            
        self.M = M
        self.d_sub = d_in // M
        self.n_iter = n_iter
        self.n_iter_pq = n_iter_pq
        self.random_rotation = random_rotation
        self.seed = seed
        
        self.rotation_matrix = None
        self._is_trained = False
        
    def train(self, x: mx.array) -> None:
        """Train the transform.
        
        Args:
            x: Training vectors (n, d_in)
        """
        if x.shape[1] != self.d_in:
            raise ValueError(f"Training vectors dimension {x.shape[1]} != transform input dimension {self.d_in}")
            
        # Set random seed
        if self.seed is not None:
            np.random.seed(self.seed)
            
        # Initialize random rotation
        if self.random_rotation:
            R = np.random.randn(self.d_in, self.d_in).astype('float32')
            U, _, Vh = np.linalg.svd(R)
            R = U @ Vh  # Make orthogonal
        else:
            R = np.eye(self.d_in, dtype='float32')
            
        # Iterate OPQ
        for _ in range(self.n_iter):
            # Apply current rotation
            xr = mx.matmul(x, mx.array(R))
            
            # Learn sub-quantizers
            sub_centroids = []
            for m in range(self.M):
                # Extract sub-vectors
                start = m * self.d_sub
                end = (m + 1) * self.d_sub
                sub_x = xr[:, start:end]
                
                # K-means clustering
                centroids = self._kmeans(sub_x, k=256)
                sub_centroids.append(centroids)
                
            # Update rotation
            C = np.zeros((self.d_in, self.d_in), dtype='float32')
            for m in range(self.M):
                start = m * self.d_sub
                end = (m + 1) * self.d_sub
                
                # Get closest centroids
                sub_x = xr[:, start:end].numpy()
                sub_c = sub_centroids[m]
                dist = ((sub_x[:, None] - sub_c) ** 2).sum(axis=2)
                idx = dist.argmin(axis=1)
                
                # Update C matrix
                C[start:end, start:end] = sub_x.T @ sub_c[idx]
                
            # SVD to get new rotation
            U, _, Vh = np.linalg.svd(C)
            R = U @ Vh
            
        self.rotation_matrix = mx.array(R)
        self._is_trained = True
        
    def apply(self, x: mx.array) -> mx.array:
        """Apply transform to vectors.
        
        Args:
            x: Input vectors (n, d_in)
            
        Returns:
            Transformed vectors (n, d_in)
        """
        if not self.is_trained:
            raise RuntimeError("Transform must be trained before applying")
            
        if x.shape[1] != self.d_in:
            raise ValueError(f"Input vectors dimension {x.shape[1]} != transform input dimension {self.d_in}")
            
        return mx.matmul(x, self.rotation_matrix)
        
    def reverse_transform(self, x: mx.array) -> mx.array:
        """Apply inverse transform to vectors.
        
        Args:
            x: Input vectors (n, d_in)
            
        Returns:
            Inverse transformed vectors (n, d_in)
        """
        if not self.is_trained:
            raise RuntimeError("Transform must be trained before applying")
            
        if x.shape[1] != self.d_in:
            raise ValueError(f"Input vectors dimension {x.shape[1]} != transform input dimension {self.d_in}")
            
        return mx.matmul(x, mx.transpose(self.rotation_matrix))
        
    def _kmeans(self, x: mx.array, k: int) -> np.ndarray:
        """Run k-means clustering.
        
        Args:
            x: Input vectors (n, d)
            k: Number of clusters
            
        Returns:
            Centroids (k, d)
        """
        # Initialize centroids randomly
        n = len(x)
        idx = np.random.choice(n, k, replace=False)
        centroids = x[idx].numpy()
        
        # Iterate k-means
        for _ in range(self.n_iter_pq):
            # Assign points to clusters
            dist = ((x.numpy()[:, None] - centroids) ** 2).sum(axis=2)
            labels = dist.argmin(axis=1)
            
            # Update centroids
            new_centroids = []
            for i in range(k):
                mask = labels == i
                if mask.sum() > 0:
                    new_centroids.append(x[mask].mean(axis=0).numpy())
                else:
                    # Empty cluster - reinitialize randomly
                    idx = np.random.randint(n)
                    new_centroids.append(x[idx].numpy())
            centroids = np.stack(new_centroids)
            
        return centroids
        
    @property
    def is_trained(self) -> bool:
        """Check if transform is trained."""
        return self._is_trained
