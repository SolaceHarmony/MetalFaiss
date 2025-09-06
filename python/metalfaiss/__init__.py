# MetalFaiss - A pure Python implementation of FAISS using MLX for Metal acceleration
# Copyright (c) 2024 Sydney Bach, The Solace Project
# Licensed under the Apache License, Version 2.0 (see LICENSE file)
#
# Original Swift implementation by Jan Krukowski used as reference for Python translation

"""
MetalFaiss: A Python implementation of FAISS using Apple's MLX framework.

This package provides vector similarity search and clustering algorithms
optimized for Apple Silicon using Metal Performance Shaders through MLX.
"""

__version__ = "0.1.0"
__author__ = "Sydney Bach"
__email__ = "sydney@solace.ofharmony.ai"

# Require MLX. No fallbacks.
try:
    import mlx.core as mx
    _HAS_MLX = True
except ImportError as e:
    raise RuntimeError(
        "MLX is required for MetalFaiss. Install with: pip install mlx"
    ) from e

# Core classes
from .metric_type import MetricType
from .search_result import SearchResult
from .search_range_result import SearchRangeResult

# Main index implementation
from .indexflat import FlatIndex

# Base classes (use MLX-aware package implementation)
from .vector_transform import BaseVectorTransform

# Utilities
# Demo utilities (helpers for examples)
from .demo_utils import load_data, encode_sentences, create_matrix, normalize_data

# Distance functions
from .distances import pairwise_L2sqr

# Clustering is MLX-only; import if present
try:
    from .clustering import BaseClustering
except ImportError:
    BaseClustering = None

# Export all functionality
__all__ = [
    # Core classes
    "FlatIndex", 
    "MetricType",
    "SearchResult",
    "SearchRangeResult",
    
    # Base classes
    "BaseVectorTransform",
    
    # Utilities
    "load_data",
    "encode_sentences", 
    "create_matrix",
    "normalize_data",
    
    # Distance functions
    "pairwise_L2sqr",
    
    # Version info
    "__version__",
    "__author__",
    "__email__"
]

# Add clustering if available
if BaseClustering is not None:
    __all__.append("BaseClustering")
