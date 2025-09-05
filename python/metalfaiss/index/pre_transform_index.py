import mlx.core as mx
import numpy as np
from typing import Optional, Union
from .index.FlatIndex import FlatIndex
from .index.IVFFlatIndex import IVFFlatIndex
from .index.BaseIndex import BaseIndex

class PreTransformIndex(BaseIndex):
    def __init__(self, base_index, transform):
        self.base_index = base_index
        self.transform = transform
        self._d = transform.d_in
        self._is_trained = False
        
    @property
    def d(self):
        return self._d

    @property 
    def is_trained(self):
        return self._is_trained
        
    def train(self, x):
        # Train both transform and base index
        self.transform.train(x)
        transformed = self.transform.apply(x)
        self.base_index.train(transformed)
        self._is_trained = True

    def add(self, x):
        transformed = self.transform.apply(x)
        self.base_index.add(transformed)

    def search(self, x, k):
        transformed = self.transform.apply(x)
        return self.base_index.search(transformed, k)

    def reset(self):
        self.base_index.reset()
        self._is_trained = False
