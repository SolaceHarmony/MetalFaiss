from typing import List
import mlx.core as mx
from .base_linear_transform import BaseLinearTransform

class OPQMatrixTransform(BaseLinearTransform):
    """OPQ (Optimized Product Quantization) matrix transform."""
    
    def __init__(self, d: int, m: int, d2: int):
        """Initialize OPQ matrix transform.
        
        Args:
            d: Input dimension
            m: Number of subquantizers
            d2: Output dimension
        """
        super().__init__(d)  # Match Swift's constructor
        self._m = m
        self._d2 = d2
        self._niter = 25
        self._niter_pq = 25
        self.linear_transform = None
        
    @property 
    def niter(self) -> int:
        """Number of iterations for optimization."""
        return self._niter
        
    @niter.setter
    def niter(self, value: int) -> None:
        """Set number of iterations for optimization."""
        self._niter = value
        
    @property
    def niter_pq(self) -> int:
        """Number of iterations for PQ training."""
        return self._niter_pq
        
    @niter_pq.setter 
    def niter_pq(self, value: int) -> None:
        """Set number of iterations for PQ training."""
        self._niter_pq = value
        
    def train(self, vectors: List[List[float]]) -> None:
        with mx.stream():
            x = mx.array(vectors, dtype=mx.float32)
            # Initialize with random orthonormal matrix
            R = mx.random.normal(0, 1, (self.d_in, self.d_out))
            Q, _ = mx.linalg.qr(R)
            self.linear_transform = Q
            mx.eval(self.linear_transform)
            
            # Train for niter iterations
            for _ in range(self._niter):
                # Project and optimize subspaces
                projected = mx.matmul(x, self.linear_transform)
                reshaped = projected.reshape(-1, self._m, self.d_out // self._m)
                U, _, Vt = mx.linalg.svd(mx.matmul(x.T, projected))
                self.linear_transform = mx.matmul(U, Vt)
                mx.eval(self.linear_transform)
                
        self._is_trained = True
