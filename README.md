# MetalFaiss

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![MLX Compatible](https://img.shields.io/badge/MLX-compatible-green.svg)](https://github.com/ml-explore/mlx)
[![Apache 2.0 License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE.md)

A pure Python implementation of FAISS (Facebook AI Similarity Search) optimized for Apple Silicon using MLX for Metal acceleration.

## Features

- **Pure Python Implementation**: No C++ dependencies, easy to install and modify
- **Metal Acceleration**: Optimized for Apple Silicon using MLX framework
- **NumPy Fallback**: Works on all platforms with automatic fallback to NumPy
- **FAISS Compatible**: Similar API to original FAISS library
- **Lazy Evaluation**: Efficient computation graphs with MLX

## Installation

### Requirements

- Python 3.8+
- NumPy (required)
- MLX (optional, for Metal acceleration on Apple Silicon)

### Install from Source

```bash
git clone https://github.com/SolaceHarmony/MetalFaiss.git
cd MetalFaiss/python
pip install -e .
```

### Install MLX (Optional, for Apple Silicon)

```bash
pip install mlx
```

## Quick Start

```python
import metalfaiss

# Create vectors (embeddings)
embeddings = [
    [0.1, 0.2, 0.3],
    [0.4, 0.5, 0.6], 
    [0.7, 0.8, 0.9],
    [1.0, 1.1, 1.2],
    [1.3, 1.4, 1.5]
]

# Create index and add vectors
d = len(embeddings[0])  # dimension
index = metalfaiss.FlatIndex(d, metalfaiss.MetricType.L2)
index.add(embeddings)

# Search for similar vectors
query = [[0.1, 0.5, 0.9]]
k = 3  # number of nearest neighbors
result = index.search(query, k)

print(f"Distances: {result.distances}")
print(f"Labels: {result.labels}")
```

## Usage

### Command Line Examples

Run the included examples:

```bash
cd python
python example_usage.py
python advanced_examples.py
```

### Supported Distance Metrics

- **L2 (Euclidean)**: `MetricType.L2`
- **L1 (Manhattan)**: `MetricType.L1`
- **L∞ (Chebyshev)**: `MetricType.LINF`
- **Inner Product**: `MetricType.INNER_PRODUCT`

### Basic Operations

```python
# Create different index types
index_l2 = metalfaiss.FlatIndex(d, metalfaiss.MetricType.L2)
index_ip = metalfaiss.FlatIndex(d, metalfaiss.MetricType.INNER_PRODUCT)

# Add vectors to index
index.add(vectors)

# Search for k nearest neighbors
result = index.search(query_vectors, k=5)

# Range search (find all vectors within distance threshold)
range_result = index.range_search(query_vectors, radius=0.5)

# Reconstruct stored vectors
reconstructed = index.reconstruct(vector_id)
```

## Performance

MetalFaiss provides excellent performance characteristics:

- **Metal Acceleration**: Leverages Apple's Metal Performance Shaders via MLX
- **Lazy Evaluation**: Only computes what's needed when it's needed
- **Memory Efficient**: Optimized memory usage patterns
- **Parallel Processing**: Automatic parallelization on supported hardware

## MLX Integration

### Why MLX?

MLX (Machine Learning for Apple silicon) provides:
- **Metal Performance Shaders**: GPU acceleration on Apple Silicon
- **Lazy Evaluation**: Build computation graphs, execute efficiently
- **Unified Memory**: Efficient memory management between CPU/GPU
- **Apple Silicon Optimization**: Native performance on M1/M2/M3 chips

### Lazy Evaluation Benefits

```python
# Operations are recorded, not immediately executed
vectors = metalfaiss.create_matrix(1000, 128)
query = metalfaiss.normalize_data(vectors[:10])

# Computation happens only when result is needed
result = index.search(query, k=5)  # <- Evaluation occurs here
print(result.distances)  # <- Results available
```

## Examples

See the `python/` directory for complete examples:

- `example_usage.py`: Basic usage patterns
- `advanced_examples.py`: Complex scenarios and optimizations

## API Reference

### Core Classes

- `FlatIndex`: Flat (exhaustive search) index
- `MetricType`: Distance metric enumeration
- `SearchResult`: K-NN search results
- `SearchRangeResult`: Range search results

### Utilities

- `load_data()`: Load vectors from file
- `create_matrix()`: Create random matrices
- `normalize_data()`: Normalize vectors to unit length

## License

Licensed under the Apache License, Version 2.0. See [LICENSE.md](LICENSE.md) for details.

## Attribution

This implementation was created by Sydney Bach for The Solace Project, with the original Swift implementation by [Jan Krukowski](https://github.com/jkrukowski/SwiftFaiss) used as a reference for the Python translation.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## Related Projects

- [FAISS](https://github.com/facebookresearch/faiss): The original Facebook AI Similarity Search library
- [MLX](https://github.com/ml-explore/mlx): Apple's machine learning framework
- [SwiftFaiss](https://github.com/jkrukowski/SwiftFaiss): The original Swift implementation used as reference
