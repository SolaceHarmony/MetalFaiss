import mlx.core as mx
import numpy as np
from abc import ABC, abstractmethod

class BaseTransform(ABC):
    @abstractmethod
    def train(self, x):
        pass
    
    @abstractmethod
    def apply(self, x):
        pass
    
    @property
    def is_trained(self):
        return True

class CenteringTransform(BaseTransform):
    def __init__(self, d):
        self.d = d
        self.mean = None
        
    def train(self, x):
        self.mean = mx.mean(x, axis=0)
        
    def apply(self, x):
        return x - self.mean

# Additional transform classes following same pattern...
