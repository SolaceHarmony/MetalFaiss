import mlx.core as mx
from typing import Optional
from .base_index import BaseIndex
from ..vector_transform.base_vector_transform import BaseVectorTransform

class PreTransformIndex(BaseIndex):
    """Index wrapper that applies a transform before delegating to a base index.

    Example: PreTransformIndex(PCAMatrixTransform(...), FlatIndex(...))
    """

    def __init__(self, transform: BaseVectorTransform, index: BaseIndex):
        self.transform = transform
        self.base_index = index
        self._d = self.transform.d_in
        self._is_trained = False
        
    @property
    def d(self):
        return self._d

    @property 
    def is_trained(self):
        return self._is_trained
        
    def train(self, x: mx.array) -> None:
        # Train both transform and base index
        self.transform.train(x)
        transformed = self.transform.apply(x)
        self.base_index.train(transformed)
        self._is_trained = True

    def add(self, x: mx.array, ids: Optional[list[int]] = None) -> None:
        transformed = self.transform.apply(x)
        self.base_index.add(transformed, ids)

    def search(self, x: mx.array, k: int):
        transformed = self.transform.apply(x)
        return self.base_index.search(transformed, k)

    def reset(self) -> None:
        self.base_index.reset()
        self._is_trained = False
