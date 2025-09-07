# MetalFaiss

MetalFaiss is a Python library that re-implements core FAISS functionalities using MLX for array operations and Metal acceleration on Apple Silicon. It provides tools for efficient similarity search, clustering, and vector transformations while retaining a high-level, Pythonic API.

> **Note:** MetalFaiss is inspired by FAISS (Facebook AI Similarity Search) and reinterprets its core ideas using MLX on Apple Silicon. It aims to be fast and flexible while using lazy evaluation and MLX's automatic optimizations. There is no NumPy fallback.

## Features

- **Indexing and Similarity Search:** 
  - Flat indexes, IVF (Inverted File) indexes, and various quantizer-based indexes.
  - Supports metrics such as L2, inner product, L1, L∞, and more.
- **Vector Transformations:**
  - Implements PCA, ITQ, OPQ, normalization, centering, and random rotations.
- **Clustering:**
  - Clustering utilities inspired by FAISS clustering modules.
- **Lazy Evaluation & JIT:**
  - Leverages MLX’s lazy evaluation and composable function transformations (e.g., automatic differentiation and vectorization).
- **Extensible & Pythonic:**
  - Uses MLX arrays on Apple Silicon; no NumPy fallback.

## Installation

MetalFaiss is distributed as an editable package for development.

1. **Clone the repository:**

   ```bash
   git clone https://github.com/sydneyrenee/MetalFaiss.git
   cd MetalFaiss/python

2. **Install in editable mode:**
   ```bash
   pip install -e .
   ```
 **Troubleshooting** : If you encounter issues with the installation, ensure you have the latest version of pip and setuptools. You can upgrade them using:

   ```bash
   pip install --upgrade pip setuptools
   ```

## Usage Example:

```python
import mlx.core as mx
from metalfaiss.indexflat import FlatIndex
from metalfaiss.metric_type import MetricType

# Generate random data: 1000 vectors of dimension 64
data = mx.array(mx.rand(1000, 64, dtype='float32'))
query = mx.array(mx.rand(5, 64, dtype='float32'))

# Create a flat index with L2 metric
index = FlatIndex(d=64, metric_type=MetricType.L2)
index.train(data)
index.add(data)

# Perform search: Find the 5 nearest neighbors for each query vector
search_result = index.search(query, k=5)

print("Distances:")
print(search_result.distances)
print("Indices:")
print(search_result.labels)
```

## Running Tests:
```bash
python -m unittest discover metalfaiss.unittest -v
```

This runs the unit tests under `metalfaiss/unittest`.

# Contributing

If you want to contribute to MetalFaiss, please follow these steps:

1. Fork the repository on GitHub.
2. Create a new branch for your feature or bugfix.
3. Make your changes and commit them with clear messages.
4. Write tests for your changes.
5. Submit a pull request with a description of your changes.

Please ensure your code adheres to the project's coding standards and passes all tests before submitting a pull request.

# License
Licensed under the Apache License, Version 2.0. See the repository’s LICENSE.md for details.

# Acknowledgments

- MetalFaiss is inspired by FAISS (Facebook AI Similarity Search) and reinterprets its core ideas using MLX.

Some design patterns were inspired by prior Swift work on FAISS; this repository contains only the Python + MLX implementation.
