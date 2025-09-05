"""
product_quantizer.py - Product Quantizer implementation using MLX

This is a port of FAISS's ProductQuantizer to MLX, maintaining similar interface
and functionality while leveraging MLX's GPU acceleration.
"""

import mlx.core as mx
import numpy as np
from typing import List, Optional, Tuple

class ProductQuantizer:
    """Product Quantizer implementation.
    
    PQ is trained using k-means, minimizing the L2 distance to centroids.
    Supports both L2 and Inner Product search, though quantization error
    is biased towards L2 distance.
    """
    
    def __init__(self, d: int, M: int, nbits: int = 8):
        """Initialize Product Quantizer.
        
        Args:
            d: Dimension of input vectors
            M: Number of subquantizers
            nbits: Number of bits per subquantizer index
        """
        if d % M != 0:
            raise ValueError(f"Dimension {d} not divisible by number of subquantizers {M}")
            
        self.d = d
        self.M = M  # Number of subquantizers
        self.nbits = nbits
        self.dsub = d // M  # Dimension of each subvector
        self.ksub = 1 << nbits  # Number of centroids for each subquantizer
        self.verbose = False
        
        # Centroid tables
        self.centroids = None  # Shape: (M, ksub, dsub)
        self.transposed_centroids = None  # Shape: (dsub, M, ksub)
        self.centroids_sq_lengths = None  # Shape: (M, ksub)
        
    def train(self, x: mx.array) -> None:
        """Train the product quantizer using k-means.
        
        Args:
            x: Training vectors (n, d)
        """
        n = len(x)
        if n == 0:
            raise ValueError("Empty training set")
            
        # Reshape input to (n, M, dsub)
        x_reshaped = x.reshape(n, self.M, self.dsub)
        
        # Initialize centroids
        self.centroids = []
        self.centroids_sq_lengths = []
        
        for m in range(self.M):
            # Select random initial centroids
            perm = mx.random.permutation(n)[:self.ksub]
            centroids = x_reshaped[:, m][perm]  # Shape: (ksub, dsub)
            
            # k-means iterations
            for _ in range(25):  # More iterations for better convergence
                # Compute distances to centroids
                diff = x_reshaped[:, m, None, :] - centroids[None, :, :]  # Shape: (n, ksub, dsub)
                distances = mx.sum(diff * diff, axis=2)  # Shape: (n, ksub)
                
                # Assign to nearest centroid
                assignments = mx.argmin(distances, axis=1)  # Shape: (n,)
                
                # Update centroids
                new_centroids = []
                for k in range(self.ksub):
                    cluster = x_reshaped[assignments == k, m]
                    if len(cluster) > 0:
                        new_centroids.append(mx.mean(cluster, axis=0))
                    else:
                        new_centroids.append(centroids[k])
                centroids = mx.stack(new_centroids)
                
            self.centroids.append(centroids)
            self.centroids_sq_lengths.append(mx.sum(centroids * centroids, axis=1))
            
        # Stack centroids into single array
        self.centroids = mx.stack(self.centroids)  # Shape: (M, ksub, dsub)
        self.centroids_sq_lengths = mx.stack(self.centroids_sq_lengths)  # Shape: (M, ksub)
        
        # Create transposed centroids for efficient distance computation
        self.sync_transposed_centroids()
        
    def sync_transposed_centroids(self) -> None:
        """Update transposed centroids to match current centroids."""
        if self.centroids is not None:
            self.transposed_centroids = mx.transpose(self.centroids, (2, 0, 1))
            
    def compute_codes(self, x: mx.array) -> mx.array:
        """Encode vectors using the product quantizer.
        
        Args:
            x: Input vectors (n, d)
            
        Returns:
            Encoded vectors (n, M) with values in range [0, ksub)
        """
        n = len(x)
        x_reshaped = x.reshape(n, self.M, self.dsub)  # Shape: (n, M, dsub)
        
        codes = []
        for m in range(self.M):
            # Compute distances to centroids
            diff = x_reshaped[:, m, None, :] - self.centroids[m]  # Shape: (n, ksub, dsub)
            distances = mx.sum(diff * diff, axis=2)  # Shape: (n, ksub)
            codes.append(mx.argmin(distances, axis=1))
            
        return mx.stack(codes, axis=1)  # Shape: (n, M)
        
    def decode(self, codes: mx.array) -> mx.array:
        """Decode vectors from their PQ codes.
        
        Args:
            codes: PQ codes (n, M) with values in range [0, ksub)
            
        Returns:
            Reconstructed vectors (n, d)
        """
        n = len(codes)
        decoded = []
        
        for m in range(self.M):
            decoded.append(self.centroids[m][codes[:, m]])
            
        return mx.concatenate(decoded, axis=1)  # Shape: (n, d)