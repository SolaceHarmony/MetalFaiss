"""
vector_transform - Vector transformation implementations for MetalFaiss

This package provides various vector transformations:
- Random rotation
- PCA matrix
- ITQ (Iterative Quantization)
- OPQ (Optimized Product Quantization)
- Binary transforms
- Simple transforms (normalization, centering, etc.)
"""

from .base_vector_transform import BaseVectorTransform
from .random_rotation import RandomRotationTransform
from .pca_matrix import PCAMatrixTransform
from .itq import ITQTransform
from .opq import OPQTransform
from .binary_transform import (
    BaseBinaryTransform,
    BinaryRotationTransform,
    BinaryMatrixTransform
)
from .simple_transforms import (
    RemapDimensionsTransform,
    NormalizationTransform,
    CenteringTransform
)

__all__ = [
    'BaseVectorTransform',
    'RandomRotationTransform',
    'PCAMatrixTransform',
    'ITQTransform',
    'OPQTransform',
    'BaseBinaryTransform',
    'BinaryRotationTransform',
    'BinaryMatrixTransform',
    'RemapDimensionsTransform',
    'NormalizationTransform',
    'CenteringTransform'
]