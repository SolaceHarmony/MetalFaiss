# (Removed shebang - examples are importable modules, not direct OS scripts)
"""
Example usage of MetalFaiss - Basic vector search demo

This example shows how to:
1. Create a FlatIndex
2. Add vectors to the index
3. Perform similarity search
4. Use different distance metrics
"""

import sys
import os

# Add the package path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'metalfaiss'))

import metalfaiss

def basic_example():
    """Basic usage example with FlatIndex."""
    print("=== MetalFaiss Basic Example ===")
    print(f"MetalFaiss version: {metalfaiss.__version__}")
    print()
    
    # Create a FlatIndex for 3-dimensional vectors using L2 distance
    d = 3
    index = metalfaiss.FlatIndex(d, metalfaiss.MetricType.L2)
    print(f"✓ Created FlatIndex with dimension {d}")
    
    # Add some example vectors
    vectors = [
        [1.0, 0.0, 0.0],  # Point on X-axis
        [0.0, 1.0, 0.0],  # Point on Y-axis  
        [0.0, 0.0, 1.0],  # Point on Z-axis
        [1.0, 1.0, 0.0],  # Point in XY-plane
        [1.0, 1.0, 1.0],  # Point in all positive octant
    ]
    
    index.add(vectors)
    print(f"✓ Added {len(vectors)} vectors to index")
    print(f"✓ Index now contains {index.ntotal} vectors")
    print()
    
    # Perform a search
    query = [[0.9, 0.1, 0.1]]  # Close to first vector
    k = 3  # Find 3 nearest neighbors
    
    result = index.search(query, k)
    print(f"Search results for query {query[0]}:")
    print(f"  Found {k} nearest neighbors:")
    
    for i, (dist, label) in enumerate(zip(result.distances[0], result.labels[0])):
        original_vector = vectors[label]
        print(f"    {i+1}. Vector {label}: {original_vector} (distance: {dist:.4f})")
    print()
    
    # Test different metrics
    print("=== Testing Different Metrics ===")
    
    # Inner product (cosine similarity)
    index_ip = metalfaiss.FlatIndex(d, metalfaiss.MetricType.INNER_PRODUCT)
    index_ip.add(vectors)
    result_ip = index_ip.search(query, k)
    
    print(f"Inner product search results:")
    for i, (dist, label) in enumerate(zip(result_ip.distances[0], result_ip.labels[0])):
        print(f"    {i+1}. Vector {label}: {vectors[label]} (inner product: {-dist:.4f})")
    print()
    
    # Test reconstruction (if supported)
    try:
        reconstructed = index.reconstruct(0)
        print(f"✓ Reconstructed vector 0: {reconstructed}")
    except NotImplementedError:
        print("ℹ  Reconstruction not supported by this index type")
    print()

def distance_metrics_example():
    """Example showing different distance metrics."""
    print("=== Distance Metrics Comparison ===")
    
    # Create vectors that are easy to understand
    vectors = [
        [1.0, 0.0],  # East
        [0.0, 1.0],  # North
        [-1.0, 0.0], # West
        [0.0, -1.0], # South
    ]
    
    query = [[0.5, 0.5]]  # Northeast direction
    
    print(f"Database vectors: {vectors}")
    print(f"Query vector: {query[0]}")
    print()
    
    metrics = [
        (metalfaiss.MetricType.L2, "L2 (Euclidean)"),
        (metalfaiss.MetricType.L1, "L1 (Manhattan)"),
        (metalfaiss.MetricType.INNER_PRODUCT, "Inner Product"),
    ]
    
    for metric_type, metric_name in metrics:
        index = metalfaiss.FlatIndex(2, metric_type)
        index.add(vectors)
        result = index.search(query, k=2)
        
        print(f"{metric_name} results:")
        for i, (dist, label) in enumerate(zip(result.distances[0], result.labels[0])):
            vector = vectors[label]
            print(f"  {i+1}. {vector} -> distance: {dist:.4f}")
        print()

if __name__ == "__main__":
    try:
        basic_example()
        distance_metrics_example()
        print("✓ All examples completed successfully!")
        
    except Exception as e:
        print(f"✗ Error running examples: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
