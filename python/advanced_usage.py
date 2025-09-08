#!/usr/bin/env python
"""
Advanced MetalFaiss Examples

This demonstrates more sophisticated usage patterns including:
- Working with larger datasets
- Different vector dimensions
- Batch operations
- Performance measurement
"""

import sys
import os
import time
import numpy as np
import mlx.core as mx

# Add the package path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'metalfaiss'))

import metalfaiss

def large_dataset_example():
    """Example with a larger synthetic dataset."""
    print("=== Large Dataset Example ===")
    
    # Generate a larger synthetic dataset
    n_vectors = 10000
    d = 128
    
    print(f"Generating {n_vectors} random {d}-dimensional vectors...")
    np.random.seed(42)
    vectors = np.random.normal(0, 1, (n_vectors, d)).astype(np.float32)
    
    # Normalize vectors for better distribution
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / norms
    
    # Create index and measure build time
    start_time = time.time()
    index = metalfaiss.FlatIndex(d, metalfaiss.MetricType.L2)
    index.add(mx.array(vectors))
    build_time = time.time() - start_time
    
    print(f"✓ Built index with {index.ntotal} vectors in {build_time:.3f} seconds")
    
    # Generate query vectors
    n_queries = 100
    queries = np.random.normal(0, 1, (n_queries, d)).astype(np.float32)
    queries = queries / np.linalg.norm(queries, axis=1, keepdims=True)
    
    # Measure search performance
    k = 10
    start_time = time.time()
    results = index.search(mx.array(queries), k)
    search_time = time.time() - start_time
    
    print(f"✓ Searched {n_queries} queries (k={k}) in {search_time:.3f} seconds")
    print(f"  Average search time: {search_time/n_queries*1000:.2f} ms per query")
    
    # Analyze results
    distances = results.distances
    labels = results.labels
    
    print(f"✓ Distance statistics:")
    print(f"  Min distance: {float(mx.min(distances).item()):.4f}")  # boundary-ok
    print(f"  Max distance: {float(mx.max(distances).item()):.4f}")  # boundary-ok
    print(f"  Mean distance: {float(mx.mean(distances).item()):.4f}")  # boundary-ok
    print()

def metric_comparison_example():
    """Compare different distance metrics on the same data."""
    print("=== Distance Metric Comparison ===")
    
    # Create test vectors with known relationships
    vectors = [
        [1.0, 0.0, 0.0],    # Unit vector in X direction
        [0.0, 1.0, 0.0],    # Unit vector in Y direction
        [0.0, 0.0, 1.0],    # Unit vector in Z direction
        [0.7, 0.7, 0.0],    # 45° in XY plane
        [0.6, 0.8, 0.0],    # Different angle in XY plane
        [-1.0, 0.0, 0.0],   # Opposite to first vector
    ]
    
    query = [[0.8, 0.6, 0.0]]  # Close to vector 4
    
    metrics = [
        (metalfaiss.MetricType.L2, "L2 (Euclidean)"),
        (metalfaiss.MetricType.L1, "L1 (Manhattan)"), 
        (metalfaiss.MetricType.INNER_PRODUCT, "Inner Product"),
    ]
    
    print(f"Query vector: {query[0]}")
    print(f"Database vectors:")
    for i, v in enumerate(vectors):
        print(f"  {i}: {v}")
    print()
    
    for metric_type, metric_name in metrics:
        index = metalfaiss.FlatIndex(3, metric_type)
        index.add(mx.array(vectors))
        result = index.search(mx.array(query), k=3)
        
        print(f"{metric_name} - Top 3 matches:")
        for i, (dist, label) in enumerate(zip(result.distances[0], result.labels[0])):
            vector = vectors[label]
            print(f"  {i+1}. Vector {label} {vector} -> distance: {dist:.4f}")
        print()

def batch_operations_example():
    """Demonstrate batch operations and incremental building."""
    print("=== Batch Operations Example ===")
    
    d = 64
    index = metalfaiss.FlatIndex(d, metalfaiss.MetricType.L2)
    
    # Add vectors in batches
    batch_sizes = [1000, 2000, 1500]
    total_added = 0
    
    for i, batch_size in enumerate(batch_sizes):
        print(f"Adding batch {i+1}: {batch_size} vectors")
        
        # Generate batch
        batch = np.random.normal(0, 1, (batch_size, d)).astype(np.float32)
        
        # Add to index
        start_time = time.time()
        index.add(mx.array(batch))
        add_time = time.time() - start_time
        total_added += batch_size
        
        print(f"  ✓ Added {batch_size} vectors in {add_time:.3f}s")
        print(f"  ✓ Index now contains {index.ntotal} vectors")
    
    print(f"Final index size: {index.ntotal} vectors")
    
    # Test search on the complete index
    query = mx.array(np.random.normal(0, 1, d).astype(np.float32)).reshape(1, -1)
    result = index.search(query, k=5)
    
    print(f"Sample search result distances: {result.distances[0]}")
    print()

def reconstruction_example():
    """Demonstrate vector reconstruction capabilities."""
    print("=== Vector Reconstruction Example ===")
    
    # Use meaningful vectors for easy verification
    original_vectors = [
        [1.0, 0.0, 0.0],  # X-axis
        [0.0, 1.0, 0.0],  # Y-axis  
        [0.0, 0.0, 1.0],  # Z-axis
        [1.0, 1.0, 0.0],  # XY diagonal
        [1.0, 1.0, 1.0],  # Space diagonal
    ]
    
    index = metalfaiss.FlatIndex(3, metalfaiss.MetricType.L2)
    index.add(original_vectors)
    
    print("Original vectors:")
    for i, v in enumerate(original_vectors):
        print(f"  {i}: {v}")
    print()
    
    print("Reconstructed vectors:")
    for i in range(len(original_vectors)):
        reconstructed = index.reconstruct(i)
        original = mx.array(original_vectors[i], dtype="float32")
        # Check accuracy with MLX
        err = float(mx.sqrt(mx.sum(mx.square(mx.subtract(reconstructed, original)))).item())  # boundary-ok
        print(f"  {i}: {reconstructed} (error: {err:.6f})")
    
    print()

if __name__ == "__main__":
    try:
        large_dataset_example()
        metric_comparison_example() 
        batch_operations_example()
        reconstruction_example()
        
        print("✅ All advanced examples completed successfully!")
        
    except Exception as e:
        print(f"❌ Error in advanced examples: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
