import mlx.core as mx
import numpy as np
from .base_linear_transform import BaseLinearTransform

class RemapDimensionsTransform(BaseLinearTransform):
    def __init__(self, d_in, d_out, uniform=True):
        super().__init__(d_in, d_out)
        self.uniform = uniform
        self._is_trained = True  # No training needed
        
        # Create remap matrix
        if uniform:
            # Uniform selection of dimensions
            step = self.d_in / self.d_out
            selected_dims = mx.array([int(i * step) for i in range(self.d_out)])
        else:
            # Random selection of dimensions
            selected_dims = mx.random.choice(self.d_in, self.d_out, replace=False)
            
        # Create remap matrix
        remap = mx.zeros((self.d_in, self.d_out))
        for i, dim in enumerate(selected_dims):
            remap = remap.at[dim, i].set(1)
            
        self.linear_transform = remap
