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
- **🔧 NumPy Fallback**: Works on systems without Metal support

## 🚀 Quick Start

```python
import metalfaiss
import numpy as np  # MLX preferred, but NumPy works as fallback

# Create some sample data (1000 vectors, 128 dimensions)
data = np.random.normal(size=(1000, 128)).astype(np.float32)
query = np.random.normal(size=(5, 128)).astype(np.float32)

# Create and populate index
index = metalfaiss.FlatIndex(d=128, metric_type=metalfaiss.MetricType.L2)
index.add(data)

# Search for 10 nearest neighbors
result = index.search(query, k=10)
print(f"Found {len(result.labels)} nearest neighbors")
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
