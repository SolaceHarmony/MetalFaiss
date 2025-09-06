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

import mlx.core as mx

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
    return mx.random.normal(shape=(rows, columns)).astype(mx.float32)

def normalize_data(data):
    """Normalize data to unit vectors."""
    data = mx.array(data, dtype=mx.float32)
    norms = mx.sqrt(mx.sum(data * data, axis=1, keepdims=True))
    norms = mx.where(norms > 0, norms, 1)
    return data / norms

def compute_distances(data, query):
    """Compute distances between data points and query."""
    data = mx.array(data, dtype=mx.float32)
    query = mx.array(query, dtype=mx.float32)
    diff = data - query
    return mx.sqrt(mx.sum(diff * diff, axis=1))

def random_projection(data, output_dim):
    """Apply random projection to reduce dimensionality."""
    data = mx.array(data, dtype=mx.float32)
    projection_matrix = mx.random.normal(shape=(data.shape[1], output_dim)).astype(mx.float32)
    return mx.matmul(data, projection_matrix)
