from abc import ABC, abstractmethod
import mlx.core as mx
from typing import List
from .Errors import IndexError

class BaseVectorTransform(ABC):
    @property
    @abstractmethod
    def is_trained(self) -> bool:
        """Whether the transform is trained"""
        pass

    @property
    @abstractmethod
    def d_in(self) -> int:
        """Input dimension"""
        pass

    @property
    @abstractmethod
    def d_out(self) -> int:
        """Output dimension"""
        pass

    @abstractmethod
    def train(self, vectors: List[List[float]]) -> None:
        """Train the transform"""
        vectors = mx.array(vectors, dtype=mx.float32)
        pass

    @abstractmethod
    def apply(self, vectors: List[List[float]]) -> List[List[float]]:
        """Apply transform to vectors"""
        vectors = mx.array(vectors, dtype=mx.float32)
        pass

    @abstractmethod
    def reverse_transform(self, vectors: List[List[float]]) -> List[List[float]]:
        """Reverse transform vectors"""
        vectors = mx.array(vectors, dtype=mx.float32)
        pass

class BaseLinearTransform(BaseVectorTransform):
    def __init__(self, d_in, d_out):
        super().__init__(d_in, d_out)
        self.is_orthonormal = False
        self.have_bias = False

    def make_orthonormal(self):
        self.is_orthonormal = True

    def transform_transpose(self, xs):
        raise NotImplementedError

class CenteringTransform(BaseVectorTransform):
    def __init__(self, d):
        super().__init__(d, d)
        self.mean = None

    def train(self, xs):
        self.mean = mx.eval.mean(xs, axis=0)
        self.is_trained = True

    def apply(self, xs):
        if not self.is_trained:
            raise ValueError("Transform not trained")
        return xs - self.mean

    def reverse_transform(self, xs):
        if not self.is_trained:
            raise ValueError("Transform not trained")
        return xs + self.mean

class ITQMatrixTransform(BaseLinearTransform):
    def __init__(self, d):
        super().__init__(d, d)
        self.rotation_matrix = None

    def train(self, xs):
        self.rotation_matrix = mx.eval.random.randn(self.d_in, self.d_out)
        self.is_trained = True

    def apply(self, xs):
        if not self.is_trained:
            raise ValueError("Transform not trained")
        return mx.eval.dot(xs, self.rotation_matrix)

    def reverse_transform(self, xs):
        if not self.is_trained:
            raise ValueError("Transform not trained")
        return mx.eval.dot(xs, mx.eval.linalg.pinv(self.rotation_matrix))

class ITQTransform(BaseVectorTransform):
    def __init__(self, d_in, d_out, do_pca):
        super().__init__(d_in, d_out)
        self.do_pca = do_pca
        self.pca_matrix = None
        self.rotation_matrix = None

    def train(self, xs):
        if self.do_pca:
            self.pca_matrix = mx.eval.random.randn(self.d_in, self.d_out)
            xs = mx.eval.dot(xs, self.pca_matrix)
        self.rotation_matrix = mx.eval.random.randn(self.d_out, self.d_out)
        self.is_trained = True

    def apply(self, xs):
        if not self.is_trained:
            raise ValueError("Transform not trained")
        if self.do_pca:
            xs = mx.eval.dot(xs, self.pca_matrix)
        return mx.eval.dot(xs, self.rotation_matrix)

    def reverse_transform(self, xs):
        if not self.is_trained:
            raise ValueError("Transform not trained")
        xs = mx.eval.dot(xs, mx.eval.linalg.pinv(self.rotation_matrix))
        if self.do_pca:
            xs = mx.eval.dot(xs, mx.eval.linalg.pinv(self.pca_matrix))
        return xs

class NormalizationTransform(BaseVectorTransform):
    def __init__(self, d, norm):
        super().__init__(d, d)
        self.norm = norm

    def apply(self, xs):
        norms = mx.eval.linalg.norm(xs, axis=1, keepdims=True)
        return xs / norms * self.norm

    def reverse_transform(self, xs):
        norms = mx.eval.linalg.norm(xs, axis=1, keepdims=True)
        return xs / self.norm * norms

class OPQMatrixTransform(BaseLinearTransform):
    def __init__(self, d, m, d2):
        super().__init__(d, d2)
        self.m = m
        self.niter = 0
        self.niter_pq = 0
        self.opq_matrix = None

    def train(self, xs):
        self.opq_matrix = mx.eval.random.randn(self.d_in, self.d_out)
        self.is_trained = True

    def apply(self, xs):
        if not self.is_trained:
            raise ValueError("Transform not trained")
        return mx.eval.dot(xs, self.opq_matrix)

    def reverse_transform(self, xs):
        if not self.is_trained:
            raise ValueError("Transform not trained")
        return mx.eval.dot(xs, mx.eval.linalg.pinv(self.opq_matrix))

class PCAMatrixTransform(BaseLinearTransform):
    def __init__(self, d_in, d_out, eigen_power, random_rotation):
        super().__init__(d_in, d_out)
        self.eigen_power = eigen_power
        self.random_rotation = random_rotation
        self.pca_matrix = None

    def train(self, xs):
        self.pca_matrix = mx.eval.random.randn(self.d_in, self.d_out)
        self.is_trained = True

    def apply(self, xs):
        if not self.is_trained:
            raise ValueError("Transform not trained")
        return mx.eval.dot(xs, self.pca_matrix)

    def reverse_transform(self, xs):
        if not self.is_trained:
            raise ValueError("Transform not trained")
        return mx.eval.dot(xs, mx.eval.linalg.pinv(self.pca_matrix))

class RandomRotationMatrixTransform(BaseLinearTransform):
    def __init__(self, d_in, d_out):
        super().__init__(d_in, d_out)
        self.rotation_matrix = None

    def train(self, xs):
        self.rotation_matrix = mx.eval.random.randn(self.d_in, self.d_out)
        self.is_trained = True

    def apply(self, xs):
        if not self.is_trained:
            raise ValueError("Transform not trained")
        return mx.eval.dot(xs, self.rotation_matrix)

    def reverse_transform(self, xs):
        if not self.is_trained:
            raise ValueError("Transform not trained")
        return mx.eval.dot(xs, mx.eval.linalg.pinv(self.rotation_matrix))

class RemapDimensionsTransform(BaseVectorTransform):
    def __init__(self, d_in, d_out, uniform):
        super().__init__(d_in, d_out)
        self.uniform = uniform
        self.remap_matrix = None

    def train(self, xs):
        self.remap_matrix = mx.eval.random.randn(self.d_in, self.d_out)
        self.is_trained = True

    def apply(self, xs):
        if not self.is_trained:
            raise ValueError("Transform not trained")
        return mx.eval.dot(xs, self.remap_matrix)

    def reverse_transform(self, xs):
        if not self.is_trained:
            raise ValueError("Transform not trained")
        return mx.eval.dot(xs, mx.eval.linalg.pinv(self.remap_matrix))
