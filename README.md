# MetalFaiss

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![MLX Compatible](https://img.shields.io/badge/MLX-compatible-green.svg)](https://github.com/ml-explore/mlx)
[![Apache 2.0 License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE.md)

A pure Python implementation of FAISS (Facebook AI Similarity Search) optimized for Apple Silicon using MLX for Metal acceleration.

## Features

- **Pure Python Implementation**: No C++ dependencies, easy to install and modify
- **Metal Acceleration**: Optimized for Apple Silicon using MLX framework
- MLX-only: Requires MLX on Apple Silicon (Metal). No fallbacks.
- **FAISS Compatible**: Similar API to original FAISS library
- **Lazy Evaluation**: Efficient computation graphs with MLX

## Installation

### Requirements

- Python 3.8+
- MLX (required)

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

### Lazy Evaluation + Kernels

See docs/mlx/Kernel-Guide.md for working `mx.fast.metal_kernel` patterns (body‑only + header), grid/threadgroup sizing, and autoswitching strategies. See docs/mlx/Orthogonality.md for non‑square orthonormalization.

Attribution: Some kernel patterns and HPC techniques are adapted from the Ember ML project by Sydney Bach (The Solace Project). We’ve encoded those real‑world lessons here so others can build reliable MLX+Metal kernels.

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
=======
# Metal FAISS

[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![MLX](https://img.shields.io/badge/MLX-compatible-orange.svg)](https://ml-explore.github.io/mlx/build/html/index.html)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE.md)
[![Platform](https://img.shields.io/badge/platform-macOS%20%7C%20Linux-lightgrey.svg)]()

**Metal FAISS** is a pure Python implementation of [Facebook's FAISS](https://github.com/facebookresearch/faiss) (Facebook AI Similarity Search) library, optimized for Apple Silicon using MLX (Metal Learning Exchange). This implementation provides efficient similarity search and clustering of dense vectors with Metal Performance Shaders acceleration on supported hardware.

> 🚀 **Pure Python + MLX**: No C++ compilation required, leverages Apple's MLX framework for Metal acceleration

## ✨ Features

- **🔍 Vector Similarity Search**: Efficient k-NN search with multiple distance metrics (L2, Inner Product, L1, L∞)
- **🚀 Metal Acceleration**: Optimized for Apple Silicon using MLX and Metal Performance Shaders  
- **📊 Multiple Index Types**: FlatIndex, IVFIndex, and more advanced indexing structures
- **🔄 Vector Transforms**: PCA, normalization, centering, and other preprocessing transforms
- **🎯 Clustering**: K-means and other clustering algorithms
- **⚡ Lazy Evaluation**: Efficient computation graphs with MLX's lazy evaluation
- **🐍 Pure Python**: No C++ compilation, easy installation and deployment
  

## 🚀 Quick Start

```python
import mlx.core as mx
import metalfaiss

# Create some sample data (1000 vectors, 128 dimensions)
data = mx.random.normal(shape=(1000, 128)).astype(mx.float32)
query = mx.random.normal(shape=(5, 128)).astype(mx.float32)

# Create and populate index
index = metalfaiss.FlatIndex(d=128, metric_type=metalfaiss.MetricType.L2)
index.add(data)

# Search for 10 nearest neighbors
result = index.search(query, k=10)
print(f"Found {result.labels.shape[0]} nearest neighbors")
print(f"Distances: {result.distances.shape}, Labels: {result.labels.shape}")
```

## 📖 More Examples

Check out our example scripts:

- **[Basic Usage](python/example_usage.py)**: Simple similarity search with FlatIndex
- **[Advanced Examples](python/advanced_examples.py)**: IVF indexes, clustering, and transforms

## 📦 Installation

### Prerequisites

- **Python 3.8+**
- **MLX**: Apple's machine learning framework
  ```bash
  pip install mlx
  ```

### Install Metal FAISS

1. **Clone the repository:**
   ```bash
   git clone https://github.com/SolaceHarmony/MetalFaiss.git
   cd MetalFaiss
   ```

2. **Install the Python package:**
   ```bash
   cd python
   pip install -e .
   ```

3. **Verify installation:**
   ```python
   import metalfaiss
   print(f"Metal FAISS version: {metalfaiss.__version__}")
   ```

### Alternative: Direct Installation
```bash
pip install mlx numpy  # Dependencies
git clone https://github.com/SolaceHarmony/MetalFaiss.git
cd MetalFaiss/python && pip install -e .
```

## 🧪 Running Tests

```bash
cd python
python -m unittest discover metalfaiss.unittest -v
```

## 🏗️ Development

### Setting up Development Environment

```bash
git clone https://github.com/SolaceHarmony/MetalFaiss.git
cd MetalFaiss/python
pip install -e .  # Editable install
```

### Project Structure

```
MetalFaiss/
├── python/                 # Python Metal FAISS implementation
│   ├── metalfaiss/        # Main package
│   │   ├── __init__.py   # Package initialization
│   │   ├── indexflat.py  # Flat index implementation  
│   │   ├── metric_type.py # Distance metrics
│   │   └── ...
│   ├── example_usage.py   # Usage examples
│   └── setup.py          # Package setup
├── Sources/               # Swift implementation (legacy)
└── README.md             # This file
```

## 📚 API Documentation

### Core Classes

- **`FlatIndex`**: Exact similarity search using brute force
- **`IVFIndex`**: Inverted file index for faster approximate search  
- **`MetricType`**: Distance metrics (L2, InnerProduct, L1, Linf)
- **`VectorTransform`**: Data preprocessing (PCA, normalization, etc.)

### Example Usage Patterns

#### Basic Similarity Search
```python
import metalfaiss
import numpy as np  # or: import mlx.core as mx

# Create index
index = metalfaiss.FlatIndex(d=128, metric_type=metalfaiss.MetricType.L2)

# Add vectors
vectors = np.random.normal(size=(1000, 128)).astype(np.float32)
index.add(vectors)

# Search
query = np.random.normal(size=(1, 128)).astype(np.float32)
result = index.search(query, k=5)
print(f"Distances: {result.distances}")
print(f"Indices: {result.labels}")
```

#### Vector Preprocessing
```python
# Apply PCA transform (when available)
try:
    transform = metalfaiss.PCAMatrixTransform(d_in=128, d_out=64)
    transform.train(training_data)
    transformed_data = transform.apply(data)
except AttributeError:
    print("PCA transform not yet implemented")
```

## 📈 Performance

Metal FAISS is optimized for Apple Silicon but works on any platform:

- **🚀 Apple Silicon**: Full Metal acceleration via MLX
- **🖥️ Intel/AMD**: NumPy fallback, still efficient
- **☁️ Cloud/Linux**: Compatible with standard Python environments

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](.github/CONTRIBUTING.md) for details.

### Quick Contribution Steps

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Development Guidelines

- Follow PEP 8 style guidelines
- Add tests for new functionality
- Update documentation as needed
- Ensure MLX compatibility

## 👥 Contributors

Thanks to all contributors who have helped build Metal FAISS:

- **[Sydney Renee](https://github.com/sydneyrenee)** - Core Python implementation and MLX integration

*Want to contribute? Check out our [Contributing Guide](.github/CONTRIBUTING.md)!*

## 🔗 Useful Resources

- **[FAISS Documentation](https://faiss.ai/)** - Original FAISS library
- **[MLX Documentation](https://ml-explore.github.io/mlx/)** - Apple's MLX framework
- **[FAISS: The Missing Manual](https://www.pinecone.io/learn/series/faiss/)** - Comprehensive FAISS guide
- **[Implementation Status](IMPLEMENTATION_STATUS.md)** - Current feature completeness

## 📄 License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## 🙏 Acknowledgments

- **[Facebook Research](https://github.com/facebookresearch/faiss)** - Original FAISS library and research
- **[Apple MLX Team](https://github.com/ml-explore/mlx)** - MLX framework enabling Metal acceleration
- **[Jan Krukowski](https://github.com/jkrukowski/SwiftFaiss)** - Swift FAISS implementation that inspired this project
- **FAISS Community** - For the foundational algorithms and research

---

<div align="center">

**⭐ Star this repo if Metal FAISS helped you! ⭐**

[🐛 Report Bug](https://github.com/SolaceHarmony/MetalFaiss/issues) • [✨ Request Feature](https://github.com/SolaceHarmony/MetalFaiss/issues) • [💬 Discussions](https://github.com/SolaceHarmony/MetalFaiss/discussions)

Made with ❤️ by the Metal FAISS team

</div>

## 🗂️ Swift Implementation (Legacy)

> **Note**: This repository also contains a Swift implementation of FAISS in the `Sources/` directory. However, the primary focus is now on the Python + MLX implementation described above.

The Swift implementation is based on [SwiftFaiss](https://github.com/jkrukowski/SwiftFaiss) and provides:
- Native Swift bindings to FAISS
- iOS compatibility
- Command-line tools

For Swift usage, please refer to the original documentation or consider using the maintained [SwiftFaiss](https://github.com/jkrukowski/SwiftFaiss) project directly.
