# MetalFaiss - A pure Python implementation of FAISS using MLX for Metal acceleration
# Copyright (c) 2024 Sydney Bach, The Solace Project
# Licensed under the Apache License, Version 2.0 (see LICENSE file)
#
# Original Swift implementation by Jan Krukowski used as reference for Python translation

# MetalFaiss - A pure Python implementation of FAISS using MLX for Metal acceleration
# Copyright (c) 2024 Sydney Bach, The Solace Project
# Licensed under the Apache License, Version 2.0 (see LICENSE file)
#
# Original Swift implementation by Jan Krukowski used as reference for Python translation

try:
    import mlx.core as mx
    _HAS_MLX = True
except ImportError:
    _HAS_MLX = False
    import numpy as np
    
    class MockMLX:
        @staticmethod
        def array(data, dtype=None):
            return np.array(data, dtype=dtype)
        
        @staticmethod
        def zeros(shape, dtype=None):
            return np.zeros(shape, dtype=dtype)
            
        @staticmethod
        def random_normal(shape, dtype=None):
            return np.random.randn(*shape).astype(dtype or np.float32)
            
        @staticmethod  
        def dot(a, b):
            return np.dot(a, b)
            
        float32 = np.float32
    
    mx = MockMLX()

import numpy as np

def load_data(file_path):
    """Load numerical data from a file."""
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append([float(x) for x in line.strip().split()])
    return mx.array(data, dtype=mx.float32)

def encode_sentences(sentences, embedding_model):
    """Encode sentences using an embedding model."""
    embeddings = []
    for sentence in sentences:
        embeddings.append(embedding_model.encode(sentence))
    return mx.array(embeddings, dtype=mx.float32)

def create_matrix(rows, columns):
    """Create a random matrix."""
    return mx.random_normal((rows, columns), dtype=mx.float32)

def normalize_data(data):
    """Normalize data to unit vectors."""
    data = mx.array(data, dtype=mx.float32)
    norms = np.linalg.norm(data, axis=1, keepdims=True)
    return data / norms

def compute_distances(data, query):
    """Compute distances between data points and query."""
    data = mx.array(data, dtype=mx.float32)
    query = mx.array(query, dtype=mx.float32)
    return np.linalg.norm(data - query, axis=1)

def random_projection(data, output_dim):
    """Apply random projection to reduce dimensionality."""
    data = mx.array(data, dtype=mx.float32)
    projection_matrix = mx.random_normal((data.shape[1], output_dim), dtype=mx.float32)
    return mx.dot(data, projection_matrix)
