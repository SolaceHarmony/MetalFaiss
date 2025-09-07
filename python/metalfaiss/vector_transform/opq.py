"""
opq.py - Optimized Product Quantization transform for MetalFaiss (MLX-only)
"""

import mlx.core as mx
from typing import Optional
from .base_vector_transform import BaseVectorTransform
from ..faissmlx.svd import topk_svd

class OPQTransform(BaseVectorTransform):
    """Optimized Product Quantization transform (MLX-only).
    
    Alternates between learning sub-quantizer centroids and updating an
    orthogonal rotation via Procrustes on the cross-covariance.
    """

    def __init__(
        self,
        d_in: int,
        M: int,
        n_iter: int = 25,
        n_iter_pq: int = 25,
        random_rotation: bool = True,
        seed: Optional[int] = None,
        key: Optional[object] = None,
    ):
        super().__init__(d_in, d_in)
        if d_in % M != 0:
            raise ValueError(f"Input dimension {d_in} must be divisible by M={M}")

        self.M = M
        self.d_sub = d_in // M
        self.n_iter = n_iter
        self.n_iter_pq = n_iter_pq
        self.random_rotation = random_rotation
        self.seed = seed
        self._key = key if key is not None else (mx.random.key(int(seed)) if seed is not None else None)

        self.rotation_matrix = None
        self._is_trained = False

    def train(self, x: mx.array) -> None:
        if x.shape[1] != self.d_in:
            raise ValueError(f"Training vectors dimension {x.shape[1]} != transform input dimension {self.d_in}")

        # Initialize rotation
        if self.random_rotation:
            if self._key is not None:
                kR, self._key = mx.random.split(self._key, num=2)
                R0 = mx.random.normal(shape=(self.d_in, self.d_in), key=kR).astype(mx.float32)
            else:
                R0 = mx.random.normal(shape=(self.d_in, self.d_in)).astype(mx.float32)
            # MLX SVD (CPU) is not allowed; use our top‑k SVD kernel
            U, _, Vt = topk_svd(R0, k=self.d_in, iters=3, use_kernel=True, use_compile=True)
            R = mx.matmul(U, Vt)
        else:
            R = mx.eye(self.d_in, dtype=mx.float32)

        # Alternate
        for _ in range(self.n_iter):
            xr = mx.matmul(x, R)

            # Subspace k-means
            sub_centroids = []
            for m in range(self.M):
                s, e = m * self.d_sub, (m + 1) * self.d_sub
                sub_x = xr[:, s:e]
                sub_centroids.append(self._kmeans_mlx(sub_x, k=256))

            # Build cross-covariance C
            C = mx.zeros((self.d_in, self.d_in), dtype=mx.float32)
            for m in range(self.M):
                s, e = m * self.d_sub, (m + 1) * self.d_sub
                sub_x = xr[:, s:e]
                sub_c = sub_centroids[m]
                nx = mx.sum(mx.square(sub_x), axis=1, keepdims=True)
                nc = mx.sum(mx.square(sub_c), axis=1, keepdims=True)
                d2 = mx.subtract(mx.add(nx, mx.transpose(nc)), mx.multiply(2, mx.matmul(sub_x, sub_c.T)))
                labels = mx.argmin(d2, axis=1)
                k = sub_c.shape[0]
                oh = (labels.reshape((-1, 1)) == mx.arange(k).reshape((1, -1))).astype(mx.float32)
                matched = mx.matmul(oh, sub_c)
                C_block = mx.matmul(sub_x.T, matched)
                C[s:e, s:e] = C_block

            U, _, Vt = topk_svd(C, k=self.d_in, iters=3, use_kernel=True, use_compile=True)
            R = mx.matmul(U, Vt)

        self.rotation_matrix = R
        self._is_trained = True

    def apply(self, x: mx.array) -> mx.array:
        if not self.is_trained:
            raise RuntimeError("Transform must be trained before applying")
        if x.shape[1] != self.d_in:
            raise ValueError(f"Input vectors dimension {x.shape[1]} != transform input dimension {self.d_in}")
        return mx.matmul(x, self.rotation_matrix)

    def reverse_transform(self, x: mx.array) -> mx.array:
        if not self.is_trained:
            raise RuntimeError("Transform must be trained before applying")
        if x.shape[1] != self.d_in:
            raise ValueError(f"Input vectors dimension {x.shape[1]} != transform input dimension {self.d_in}")
        return mx.matmul(x, mx.transpose(self.rotation_matrix))

    def _kmeans_mlx(self, x: mx.array, k: int, iters: int = 25) -> mx.array:
        """Simple MLX-only k-means returning centroids (k, d)."""
        n, d = x.shape
        if self._key is not None:
            kp, self._key = mx.random.split(self._key, num=2)
            perm = mx.random.permutation(n, key=kp)
        else:
            perm = mx.random.permutation(n)
        centroids = x[perm[:k]]
        for _ in range(self.n_iter_pq):
            nx = mx.sum(mx.square(x), axis=1, keepdims=True)
            nc = mx.sum(mx.square(centroids), axis=1, keepdims=True)
            d2 = mx.subtract(mx.add(nx, mx.transpose(nc)), mx.multiply(2, mx.matmul(x, centroids.T)))
            labels = mx.argmin(d2, axis=1)
            oh = (labels.reshape((-1, 1)) == mx.arange(k).reshape((1, -1))).astype(mx.float32)
            counts = mx.sum(oh, axis=0).reshape((k, 1))
            sums = mx.matmul(oh.T, x)
            counts = mx.maximum(counts, 1)
            centroids = mx.divide(sums, counts)
        return centroids
