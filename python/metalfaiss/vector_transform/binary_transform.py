"""
binary_transform.py - Binary vector transforms for MetalFaiss

Pure MLX implementation - zero CPU operations, no NumPy.
All random number generation and transforms execute on GPU.
"""

import mlx.core as mx
from typing import Optional, Tuple
from .base_vector_transform import BaseVectorTransform

class BaseBinaryTransform(BaseVectorTransform):
    """Base class for binary vector transforms."""
    
    def __init__(self, d_in: int, d_out: Optional[int] = None):
        """Initialize binary transform.
        
        Args:
            d_in: Input dimension (must be multiple of 8)
            d_out: Output dimension (default: same as input)
        """
        resolved_d_out = d_out if d_out is not None else d_in
        super().__init__(d_in, resolved_d_out)
        if d_in % 8 != 0:
            raise ValueError(f"Input dimension {d_in} must be multiple of 8")
        if self.d_out % 8 != 0:
            raise ValueError(f"Output dimension {self.d_out} must be multiple of 8")
            
    def apply(self, x: mx.array) -> mx.array:
        """Apply transform to binary vectors.
        
        Args:
            x: Input vectors (n, d_in)
            
        Returns:
            Transformed vectors (n, d_out)
        """
        raise NotImplementedError
        
    def reverse_transform(self, x: mx.array) -> mx.array:
        """Apply inverse transform to binary vectors.
        
        Args:
            x: Input vectors (n, d_out)
            
        Returns:
            Inverse transformed vectors (n, d_in)
        """
        raise NotImplementedError

class BinaryRotationTransform(BaseBinaryTransform):
    """Binary rotation transform.
    
    This transform applies a random binary rotation matrix to the input vectors.
    The rotation matrix is a random permutation matrix that preserves Hamming
    distances between vectors.
    """
    
    def __init__(self, d_in: int, seed: Optional[int] = None):
        """Initialize binary rotation transform.
        
        Args:
            d_in: Input dimension (must be multiple of 8)
            seed: Random seed (default: None)
        """
        super().__init__(d_in)
        self.seed = seed
        self.permutation = None
        self._is_trained = False
        
    def train(self, x: mx.array) -> None:
        """Train the transform by generating random permutation.
        
        Args:
            x: Training vectors (not used)
        """
        if x.shape[1] != self.d_in:
            raise ValueError(f"Training vectors dimension {x.shape[1]} != transform input dimension {self.d_in}")
            
        # Set random seed if provided
        if self.seed is not None:
            mx.random.seed(self.seed)
            
        # Generate random permutation using pure MLX
        # Strategy: generate random values and argsort to get permutation
        d_in_mx = mx.array(self.d_in, dtype=mx.int32)
        random_vals = mx.random.uniform(shape=(self.d_in,), dtype=mx.float32)
        self.permutation = mx.argsort(random_vals, axis=mx.array(0, dtype=mx.int32))
        self._is_trained = True
        
    def apply(self, x: mx.array) -> mx.array:
        """Apply binary rotation to vectors.
        
        Args:
            x: Input vectors (n, d_in)
            
        Returns:
            Rotated vectors (n, d_in)
        """
        if not self.is_trained:
            raise RuntimeError("Transform must be trained before applying")
            
        if x.shape[1] != self.d_in:
            raise ValueError(f"Input vectors dimension {x.shape[1]} != transform input dimension {self.d_in}")
            
        # Apply permutation
        return x[:, self.permutation]
        
    def reverse_transform(self, x: mx.array) -> mx.array:
        """Apply inverse binary rotation to vectors.
        
        Args:
            x: Input vectors (n, d_in)
            
        Returns:
            Inverse rotated vectors (n, d_in)
        """
        if not self.is_trained:
            raise RuntimeError("Transform must be trained before applying")
            
        if x.shape[1] != self.d_in:
            raise ValueError(f"Input vectors dimension {x.shape[1]} != transform input dimension {self.d_in}")
            
        # Build inverse permutation using pure MLX
        # inv_perm[permutation[i]] = i
        d_in_mx = mx.array(self.d_in, dtype=mx.int32)
        inv_perm = mx.zeros((self.d_in,), dtype=mx.int32)
        indices = mx.arange(self.d_in, dtype=mx.int32)
        
        # Scatter operation: inv_perm[self.permutation] = indices
        # MLX doesn't have direct scatter, so we use sorting trick
        # Create pairs (perm_value, original_index)
        # Sort by perm_value to get inverse mapping
        perm_indices = mx.stack([self.permutation, indices], axis=mx.array(0, dtype=mx.int32))
        sort_order = mx.argsort(perm_indices[0], axis=mx.array(0, dtype=mx.int32))
        inv_perm = perm_indices[1][sort_order]
        
        return x[:, inv_perm]
        
    @property
    def is_trained(self) -> bool:
        """Check if transform is trained."""
        return self._is_trained

class BinaryMatrixTransform(BaseBinaryTransform):
    """Binary matrix transform.
    
    This transform applies a binary matrix multiplication to the input vectors.
    The matrix is trained to minimize reconstruction error while preserving
    Hamming distances between vectors.
    """
    
    def __init__(
        self,
        d_in: int,
        d_out: Optional[int] = None,
        n_iter: int = 10,
        seed: Optional[int] = None
    ):
        """Initialize binary matrix transform.
        
        Args:
            d_in: Input dimension (must be multiple of 8)
            d_out: Output dimension (default: same as input)
            n_iter: Number of training iterations
            seed: Random seed (default: None)
        """
        super().__init__(d_in, d_out)
        self.n_iter = n_iter
        self.seed = seed
        self.matrix = None
        self._is_trained = False
        
    def train(self, x: mx.array) -> None:
        """Train the transform by learning binary matrix.
        
        Args:
            x: Training vectors (n, d_in)
        """
        if x.shape[1] != self.d_in:
            raise ValueError(f"Training vectors dimension {x.shape[1]} != transform input dimension {self.d_in}")
            
        # Set random seed if provided
        if self.seed is not None:
            mx.random.seed(self.seed)
            
        # Initialize random binary matrix using pure MLX
        # Generate uniform random [0, 1) and threshold at 0.5 to get binary {0, 1}
        random_matrix = mx.random.uniform(
            shape=(self.d_in, self.d_out),
            dtype=mx.float32
        )
        half = mx.array(0.5, dtype=mx.float32)
        self.matrix = mx.greater_equal(random_matrix, half).astype(mx.uint8)
        
        # Iterate to minimize reconstruction error
        n_iter_mx = mx.array(self.n_iter, dtype=mx.int32)
        for _ in range(self.n_iter):
            # Forward pass
            y = self.apply(x)
            
            # Backward pass - update matrix
            # Use float32 for gradient computation
            x_float = x.astype(mx.float32)
            y_float = y.astype(mx.float32)
            grad = mx.matmul(mx.transpose(x_float, axes=[1, 0]), y_float)
            grad_mean = mx.mean(grad)
            self.matrix = mx.greater(grad, grad_mean).astype(mx.uint8)
            
        self._is_trained = True
        
    def apply(self, x: mx.array) -> mx.array:
        """Apply binary matrix transform to vectors.
        
        Args:
            x: Input vectors (n, d_in)
            
        Returns:
            Transformed vectors (n, d_out)
        """
        if not self.is_trained:
            raise RuntimeError("Transform must be trained before applying")
            
        if x.shape[1] != self.d_in:
            raise ValueError(f"Input vectors dimension {x.shape[1]} != transform input dimension {self.d_in}")
            
        # Binary matrix multiplication
        # Convert to float for matmul, then threshold
        y = mx.matmul(x.astype(mx.float32), self.matrix.astype(mx.float32))
        y_mean = mx.mean(y)
        return mx.greater(y, y_mean).astype(mx.uint8)
        
    def reverse_transform(self, x: mx.array) -> mx.array:
        """Apply inverse binary matrix transform to vectors.
        
        Args:
            x: Input vectors (n, d_out)
            
        Returns:
            Inverse transformed vectors (n, d_in)
        """
        if not self.is_trained:
            raise RuntimeError("Transform must be trained before applying")
            
        if x.shape[1] != self.d_out:
            raise ValueError(f"Input vectors dimension {x.shape[1]} != transform output dimension {self.d_out}")
            
        # Binary matrix multiplication with transpose
        # Convert to float for matmul, then threshold
        y = mx.matmul(x.astype(mx.float32), mx.transpose(self.matrix.astype(mx.float32), axes=[1, 0]))
        y_mean = mx.mean(y)
        return mx.greater(y, y_mean).astype(mx.uint8)
        
    @property
    def is_trained(self) -> bool:
        """Check if transform is trained."""
        return self._is_trained
