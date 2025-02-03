import numpy as np
import mlx
import metal

class BaseVectorTransform:
    def __init__(self, d_in, d_out):
        self.d_in = d_in
        self.d_out = d_out
        self.is_trained = False

    def train(self, xs):
        raise NotImplementedError

    def apply(self, xs):
        raise NotImplementedError

    def reverse_transform(self, xs):
        raise NotImplementedError

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
        self.mean = np.mean(xs, axis=0)
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
        self.rotation_matrix = np.random.randn(self.d_in, self.d_out)
        self.is_trained = True

    def apply(self, xs):
        if not self.is_trained:
            raise ValueError("Transform not trained")
        return np.dot(xs, self.rotation_matrix)

    def reverse_transform(self, xs):
        if not self.is_trained:
            raise ValueError("Transform not trained")
        return np.dot(xs, np.linalg.pinv(self.rotation_matrix))

class ITQTransform(BaseVectorTransform):
    def __init__(self, d_in, d_out, do_pca):
        super().__init__(d_in, d_out)
        self.do_pca = do_pca
        self.pca_matrix = None
        self.rotation_matrix = None

    def train(self, xs):
        if self.do_pca:
            self.pca_matrix = np.random.randn(self.d_in, self.d_out)
            xs = np.dot(xs, self.pca_matrix)
        self.rotation_matrix = np.random.randn(self.d_out, self.d_out)
        self.is_trained = True

    def apply(self, xs):
        if not self.is_trained:
            raise ValueError("Transform not trained")
        if self.do_pca:
            xs = np.dot(xs, self.pca_matrix)
        return np.dot(xs, self.rotation_matrix)

    def reverse_transform(self, xs):
        if not self.is_trained:
            raise ValueError("Transform not trained")
        xs = np.dot(xs, np.linalg.pinv(self.rotation_matrix))
        if self.do_pca:
            xs = np.dot(xs, np.linalg.pinv(self.pca_matrix))
        return xs

class NormalizationTransform(BaseVectorTransform):
    def __init__(self, d, norm):
        super().__init__(d, d)
        self.norm = norm

    def apply(self, xs):
        norms = np.linalg.norm(xs, axis=1, keepdims=True)
        return xs / norms * self.norm

    def reverse_transform(self, xs):
        norms = np.linalg.norm(xs, axis=1, keepdims=True)
        return xs / self.norm * norms

class OPQMatrixTransform(BaseLinearTransform):
    def __init__(self, d, m, d2):
        super().__init__(d, d2)
        self.m = m
        self.niter = 0
        self.niter_pq = 0
        self.opq_matrix = None

    def train(self, xs):
        self.opq_matrix = np.random.randn(self.d_in, self.d_out)
        self.is_trained = True

    def apply(self, xs):
        if not self.is_trained:
            raise ValueError("Transform not trained")
        return np.dot(xs, self.opq_matrix)

    def reverse_transform(self, xs):
        if not self.is_trained:
            raise ValueError("Transform not trained")
        return np.dot(xs, np.linalg.pinv(self.opq_matrix))

class PCAMatrixTransform(BaseLinearTransform):
    def __init__(self, d_in, d_out, eigen_power, random_rotation):
        super().__init__(d_in, d_out)
        self.eigen_power = eigen_power
        self.random_rotation = random_rotation
        self.pca_matrix = None

    def train(self, xs):
        self.pca_matrix = np.random.randn(self.d_in, self.d_out)
        self.is_trained = True

    def apply(self, xs):
        if not self.is_trained:
            raise ValueError("Transform not trained")
        return np.dot(xs, self.pca_matrix)

    def reverse_transform(self, xs):
        if not self.is_trained:
            raise ValueError("Transform not trained")
        return np.dot(xs, np.linalg.pinv(self.pca_matrix))

class RandomRotationMatrixTransform(BaseLinearTransform):
    def __init__(self, d_in, d_out):
        super().__init__(d_in, d_out)
        self.rotation_matrix = None

    def train(self, xs):
        self.rotation_matrix = np.random.randn(self.d_in, self.d_out)
        self.is_trained = True

    def apply(self, xs):
        if not self.is_trained:
            raise ValueError("Transform not trained")
        return np.dot(xs, self.rotation_matrix)

    def reverse_transform(self, xs):
        if not self.is_trained:
            raise ValueError("Transform not trained")
        return np.dot(xs, np.linalg.pinv(self.rotation_matrix))

class RemapDimensionsTransform(BaseVectorTransform):
    def __init__(self, d_in, d_out, uniform):
        super().__init__(d_in, d_out)
        self.uniform = uniform
        self.remap_matrix = None

    def train(self, xs):
        self.remap_matrix = np.random.randn(self.d_in, self.d_out)
        self.is_trained = True

    def apply(self, xs):
        if not self.is_trained:
            raise ValueError("Transform not trained")
        return np.dot(xs, self.remap_matrix)

    def reverse_transform(self, xs):
        if not self.is_trained:
            raise ValueError("Transform not trained")
        return np.dot(xs, np.linalg.pinv(self.remap_matrix))
