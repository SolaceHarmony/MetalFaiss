import argparse
import mlx.core as mx
import numpy as np
from .index.FlatIndex import FlatIndex
from .index.IVFFlatIndex import IVFFlatIndex
from .PQIndex import PQIndex
from .Utils import load_data, encode_sentences

def main():
    parser = argparse.ArgumentParser(description='MetalFaiss CLI')
    parser.add_argument('command', choices=['flat', 'ivfflat', 'pq', 'clustering', 'pretransform'])
    parser.add_argument('--dim', type=int, default=64, help='Vector dimension')
    parser.add_argument('--nlist', type=int, default=100, help='Number of clusters for IVF')
    parser.add_argument('--data', type=str, required=True, help='Path to data file')
    parser.add_argument('--query', type=str, required=True, help='Path to query file')
    parser.add_argument('--k', type=int, default=5, help='Number of nearest neighbors')
    
    args = parser.parse_args()
    
    data = load_data(args.data)
    query = load_data(args.query)
    
    if args.command == 'flat':
        index = FlatIndex(args.dim)
        index.add(data)
    elif args.command == 'ivfflat':
        index = IVFFlatIndex(args.dim, args.nlist)
        index.train(data)
        index.add(data)
    elif args.command == 'pq':
        index = PQIndex(args.dim, m=8)
        index.train(data)
        index.add(data)
    elif args.command == 'clustering':
        from .clustering.any_clustering import AnyClustering
        clustering = AnyClustering.new(d=args.dim, k=args.k)
        clustering.train(data)
        centroids = clustering.centroids()
        print(f"Computed {len(centroids)} centroids")
    elif args.command == 'pretransform':
        base_index = FlatIndex(args.dim)
        matrix = mx.random.normal(shape=(args.dim, args.dim)).astype(mx.float32)
        index = PreTransformIndex(base_index, matrix)
        index.add(data)
        
    indices, distances = index.search(query, args.k)
    for idx, dist in zip(indices, distances):
        print(f"Index: {idx}, Distance: {dist}")

if __name__ == '__main__':
    main()
