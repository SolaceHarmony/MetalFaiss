# Testing MetalFaiss

This document describes how to run tests for MetalFaiss to verify functionality and FAISS API compatibility.

## Test Structure

Tests are located in: `python/metalfaiss/unittest/`

The test suite uses Python's built-in `unittest` framework and includes:
- Unit tests for individual components
- Integration tests for complete workflows  
- FAISS API compatibility tests
- Benchmark tests for performance validation

## Running Tests

### Run All Tests

From the repository root:

```bash
cd python
python -m unittest discover metalfaiss.unittest -v
```

Or using PYTHONPATH:

```bash
PYTHONPATH=python python3 -m unittest discover -s python/metalfaiss/unittest -p "test_*.py" -v
```

### Run Specific Test Module

Run a single test file:

```bash
cd /path/to/MetalFaiss
PYTHONPATH=python python3 -m metalfaiss.unittest.test_faiss_api_compatibility
```

### Run FAISS API Compatibility Tests

To specifically test FAISS API compatibility:

```bash
PYTHONPATH=python python3 -m metalfaiss.unittest.test_faiss_api_compatibility -v
```

Expected output:
```
test_flat_index_add ... ok
test_flat_index_creation ... ok
test_flat_index_search ... ok
test_train_method ... ok
...
----------------------------------------------------------------------
Ran 25 tests in 0.036s

OK
```

### Run Specific Test Classes

```bash
# Run only core API tests
PYTHONPATH=python python3 -m unittest metalfaiss.unittest.test_faiss_api_compatibility.TestFaissApiCore -v

# Run only factory tests  
PYTHONPATH=python python3 -m unittest metalfaiss.unittest.test_faiss_api_compatibility.TestFaissApiFactory -v

# Run only compatibility layer tests
PYTHONPATH=python python3 -m unittest metalfaiss.unittest.test_faiss_api_compatibility.TestFaissCompatLayer -v
```

## Test Categories

### 1. FAISS API Compatibility Tests (`test_faiss_api_compatibility.py`)

**Purpose:** Verify that MetalFaiss provides FAISS-compatible APIs

**Test Classes:**
- `TestFaissApiCore` - Core index operations (add, search, train, properties)
- `TestFaissApiFactory` - Index factory string grammar compatibility
- `TestFaissCompatLayer` - FAISS compatibility layer (faiss_compat module)
- `TestIVFCompatibility` - IVF index FAISS compatibility
- `TestHNSWCompatibility` - HNSW index FAISS compatibility
- `TestIDMapCompatibility` - ID mapping wrapper compatibility
- `TestMetricTypeCompatibility` - Metric type compatibility
- `TestSearchResultCompatibility` - Search result format compatibility

**Total Tests:** 25

**Run Command:**
```bash
PYTHONPATH=python python3 -m metalfaiss.unittest.test_faiss_api_compatibility
```

### 2. Component Tests

Individual component tests are available in the `unittest/` directory:

- `test_index_factory.py` - Index factory system tests
- `test_distances.py` - Distance computation tests
- `test_hnsw.py` - HNSW index tests
- `test_id_map.py` - ID mapping tests
- `test_product_quantizer.py` - PQ tests
- `test_transforms.py` - Vector transform tests
- And many more...

### 3. Benchmark Tests

Performance benchmarks can be run separately:

```bash
# GEMM kernel benchmarks
METALFAISS_USE_GEMM_KERNEL=1 PYTHONPATH=python python3 -m metalfaiss.unittest.test_kernel_autotune_bench

# IVF performance
METALFAISS_USE_IVF_TOPK=1 PYTHONPATH=python python3 -m metalfaiss.unittest.test_ivf_benchmarks

# PyTorch vs MLX comparison
METALFAISS_USE_GEMM_KERNEL=1 PYTHONPATH=python python3 -m metalfaiss.unittest.test_torch_vs_mlx_bench
```

## Test Coverage

### What's Tested ✅

1. **Core Index Operations**
   - Index creation (Flat, IVF, HNSW, PQ)
   - Add vectors
   - Search (K-NN)
   - Training (where required)
   - Properties (ntotal, d, is_trained, nprobe, etc.)

2. **Index Factory**
   - String grammar parsing
   - Supported index types
   - Reverse factory (index → string)

3. **FAISS Compatibility Layer**
   - IndexFlatL2, IndexFlatIP
   - IndexIVFFlat, IndexIVFPQ
   - IndexHNSWFlat
   - IDMap, IDMap2
   - normalize_L2 function

4. **Metric Types**
   - L2, INNER_PRODUCT, L1, LINF
   - Metric type usage in indexes

5. **Search Results**
   - SearchResult object properties
   - Distances and labels/indices
   - FAISS-style unpacking

### Known Limitations ⚠️

Some advanced features are stubbed or partially implemented:
- IndexIVFScalarQuantizer (stub)
- IndexShards/IndexReplicas (stub)
- Binary indexes (partial)

These limitations are documented in tests and will raise clear NotImplementedError exceptions.

## Continuous Integration

### Running Tests in CI

For CI/CD pipelines, run the full test suite:

```bash
#!/bin/bash
set -e

# Install dependencies
pip install mlx

# Run all tests
cd python
python -m unittest discover metalfaiss.unittest -v

# Run FAISS compatibility tests specifically
PYTHONPATH=python python3 -m metalfaiss.unittest.test_faiss_api_compatibility -v
```

### Exit Codes

- `0` - All tests passed
- Non-zero - Test failures or errors

## Writing New Tests

When adding new functionality, add corresponding tests:

1. Create test file in `python/metalfaiss/unittest/`
2. Name it `test_<feature>.py`
3. Use `unittest.TestCase` as base class
4. Import modules using relative imports from parent package

Example:

```python
import unittest
import mlx.core as mx
from metalfaiss import FlatIndex, MetricType

class TestNewFeature(unittest.TestCase):
    def setUp(self):
        self.d = 3
        
    def test_feature_basic(self):
        """Test basic feature functionality."""
        index = FlatIndex(self.d, MetricType.L2)
        self.assertEqual(index.d, self.d)
        
    def test_feature_edge_case(self):
        """Test edge cases."""
        # Test code here
        pass

if __name__ == '__main__':
    unittest.main()
```

## Debugging Tests

### Run with verbose output:

```bash
PYTHONPATH=python python3 -m metalfaiss.unittest.test_faiss_api_compatibility -v
```

### Run specific test method:

```bash
PYTHONPATH=python python3 -m unittest metalfaiss.unittest.test_faiss_api_compatibility.TestFaissApiCore.test_flat_index_creation -v
```

### Add debug prints:

Temporarily add print statements in test methods for debugging.

## Test Requirements

- Python 3.8+
- MLX (Apple Silicon)
- NumPy (optional, for some tests)

Install test dependencies:

```bash
pip install mlx
```

## Summary

MetalFaiss includes a comprehensive test suite with **25+ FAISS API compatibility tests** that verify drop-in replacement capability for common FAISS use cases on Apple Silicon.

**Quick Test Commands:**

```bash
# All tests
cd python && python -m unittest discover metalfaiss.unittest -v

# FAISS compatibility only
PYTHONPATH=python python3 -m metalfaiss.unittest.test_faiss_api_compatibility

# Specific test class
PYTHONPATH=python python3 -m unittest metalfaiss.unittest.test_faiss_api_compatibility.TestFaissApiCore -v
```

For more information, see:
- `FAISS_API_COMPATIBILITY.md` - Detailed API compatibility analysis
- `README.md` - General project documentation
- `python/metalfaiss/unittest/` - Test source code
