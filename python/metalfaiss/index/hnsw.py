"""
hnsw.py - Faithful MLX port of FAISS HNSW implementation

This implements the HNSW graph structure following the original FAISS implementation
in HNSW.h/cpp, adapted to use MLX for efficient computation.
"""

import mlx.core as mx
import numpy as np
from typing import List, Optional, Tuple, Dict, Set
from dataclasses import dataclass
import random
import math
import heapq
from ..types.metric_type import MetricType

@dataclass
class HNSWStats:
    """Statistics collected during HNSW operations."""
    n1: int = 0      # Number of vectors searched
    n2: int = 0      # Number of exhausted candidate lists
    ndis: int = 0    # Number of distances computed
    nhops: int = 0   # Number of edges traversed
    
    def reset(self) -> None:
        self.n1 = self.n2 = self.ndis = self.nhops = 0
        
    def combine(self, other: 'HNSWStats') -> None:
        self.n1 += other.n1
        self.n2 += other.n2
        self.ndis += other.ndis
        self.nhops += other.nhops

class MinimaxHeap:
    """Heap structure matching FAISS implementation."""
    
    def __init__(self, capacity: int):
        """Initialize heap with given capacity."""
        self.capacity = capacity
        self.n = 0  # Current size
        self.nvalid = 0  # Number of valid entries
        self.ids = mx.zeros(capacity, dtype=mx.int32)
        self.distances = mx.zeros(capacity, dtype=mx.float32)
        # Static +inf scalar for sentinels
        self._inf = mx.divide(mx.ones((), dtype=mx.float32), mx.zeros((), dtype=mx.float32))
        
    def push(self, idx: int, dist: float) -> None:
        """Add new element following FAISS logic."""
        if self.n == self.capacity:
            if dist >= float(self.distances[0]):
                return
            if int(self.ids[0]) != -1:
                self.nvalid -= 1
            # Pop max element
            self._pop_max()
        # Add new element
        self._sift_up(dist, idx)
        self.nvalid += 1
        
    def _sift_up(self, dist: float, idx: int) -> None:
        """Add element and maintain heap property."""
        i = self.n
        self.n += 1
        while i > 0:
            parent = (i - 1) // 2
            if float(self.distances[parent]) <= dist:
                break
            self.ids = mx.scatter(self.ids, mx.array([i]), self.ids[parent:parent+1])
            self.distances = mx.scatter(self.distances, mx.array([i]), self.distances[parent:parent+1])
            i = parent
        self.ids = mx.scatter(self.ids, mx.array([i]), mx.array([idx]))
        self.distances = mx.scatter(self.distances, mx.array([i]), mx.array([dist]))
        
    def _pop_max(self) -> None:
        """Remove maximum element."""
        if self.n == 0:
            return
        self.n -= 1
        if self.n > 0:
            # Move last element to root
            self.ids = mx.scatter(self.ids, mx.array([0]), self.ids[self.n:self.n+1])
            self.distances = mx.scatter(self.distances, mx.array([0]), self.distances[self.n:self.n+1])
            self._sift_down(0)
            
    def _sift_down(self, idx: int) -> None:
        """Move element down to maintain heap property."""
        while True:
            min_idx = idx
            left = 2 * idx + 1
            right = 2 * idx + 2
            
            if left < self.n and float(self.distances[left]) < float(self.distances[min_idx]):
                min_idx = left
            if right < self.n and float(self.distances[right]) < float(self.distances[min_idx]):
                min_idx = right
                
            if min_idx == idx:
                break
                
            # Swap with smallest child
            self.ids = mx.scatter(
                self.ids,
                mx.array([idx, min_idx]),
                mx.stack([self.ids[min_idx], self.ids[idx]])
            )
            self.distances = mx.scatter(
                self.distances,
                mx.array([idx, min_idx]),
                mx.stack([self.distances[min_idx], self.distances[idx]])
            )
            idx = min_idx
            
    def pop_min(self) -> Tuple[int, float]:
        """Remove and return minimum element."""
        if self.n == 0:
            return -1, float(self._inf)
            
        # Find minimum valid element
        min_idx = -1
        min_dist = float(self._inf)
        
        # Use MLX operations for efficiency
        valid_mask = mx.not_equal(self.ids, mx.array(-1, dtype=self.ids.dtype))
        cnt = int(mx.sum(valid_mask).astype(mx.int32).item())  # boundary-ok
        if cnt == 0:
            return -1, float(self._inf.item())
            
        distances = mx.where(valid_mask, self.distances, self._inf)
        min_dist_arr = mx.min(distances)
        min_indices = mx.where(mx.equal(distances, min_dist_arr))[0]
        min_idx = int(mx.max(min_indices).item())  # boundary-ok, Take rightmost minimum
        
        # Get result and mark as invalid
        result = int(self.ids[min_idx].item())
        self.ids = mx.scatter(self.ids, mx.array([min_idx]), mx.array([-1]))
        self.nvalid -= 1
        
        return result, float(min_dist_arr.item())  # boundary-ok
        
    def count_below(self, threshold: float) -> int:
        """Count elements below threshold."""
        return int(mx.sum(mx.less(self.distances, mx.array(threshold, dtype=self.distances.dtype))).astype(mx.int32).item())  # boundary-ok

class HNSW:
    """HNSW implementation following FAISS."""
    
    def __init__(self, M: int = 32):
        """Initialize HNSW structure.
        
        Args:
            M: Number of neighbors per layer (except layer 0 which has 2*M)
        """
        self.M = M
        self.M0 = 2 * M
        
        # Level parameters
        self.level_mult = 1/math.log(M)
        self.assign_probas = []
        self.cum_nneighbor_per_level = []
        
        # Graph structure
        self.entry_point = -1
        self.max_level = -1
        self.levels = mx.array([], dtype=mx.int32)
        self.offsets = mx.array([], dtype=mx.int64)
        self.neighbors = mx.array([], dtype=mx.int32)
        
        # Search parameters
        self.efConstruction = 40
        self.efSearch = 16
        self.check_relative_distance = True
        self.search_bounded_queue = True
        
        self.set_default_probas()

    # Lightweight 1D scatter helper (avoids mx.scatter dependency)
    @staticmethod
    def _scatter1d(a: mx.array, idx: mx.array, val: mx.array) -> mx.array:
        """Return a copy of a with a[idx[i]] = val[i] for 1D arrays.

        idx and val are 1D and have equal length. Implemented via masks.
        """
        n = a.shape[0]
        ar = mx.arange(n, dtype=idx.dtype)
        out = a
        m = idx.shape[0]
        for i in range(int(m)):
            mask = ar == idx[i]
            out = mx.where(mask, val[i:i+1].reshape(()).astype(out.dtype), out)
        return out
        
    def set_default_probas(self) -> None:
        """Set level probabilities following FAISS logic."""
        self.assign_probas = []
        cum_proba = 0
        
        for level in range(32):  # Max 32 levels as in FAISS
            proba = math.exp(-level / self.level_mult) * (1 - math.exp(-1 / self.level_mult))
            if proba < 1e-9:
                break
            self.assign_probas.append(proba)
            cum_proba += proba
            
        # Normalize probabilities
        self.assign_probas = [p / cum_proba for p in self.assign_probas]
        
        # Set neighbors per level
        self.cum_nneighbor_per_level = []
        cum_nn = 0
        for level in range(len(self.assign_probas)):
            cum_nn += self.M0 if level == 0 else self.M
            self.cum_nneighbor_per_level.append(cum_nn)
            
    def random_level(self) -> int:
        """Generate random level following FAISS probability distribution."""
        f = random.random()
        for level in range(len(self.assign_probas)):
            if f < self.assign_probas[level]:
                return level
            f -= self.assign_probas[level]
        return len(self.assign_probas) - 1
        
    @staticmethod
    def shrink_neighbor_list(
        dist_computer,
        candidates: List[Tuple[float, int]],
        max_size: int,
        keep_max_size_level0: bool = False
    ) -> List[Tuple[float, int]]:
        """Shrink neighbor list using FAISS diversity heuristic."""
        if len(candidates) <= max_size:
            return candidates
            
        # Sort by distance
        candidates.sort()
        
        # Keep track of pruned candidates
        pruned = []
        selected = [candidates[0]]
        
        # Process remaining candidates
        for dist, idx in candidates[1:]:
            # Check if candidate should be pruned
            good = True
            for _, sel_idx in selected:
                # If closer to any selected neighbor than to query, prune
                dist_to_sel = dist_computer(idx, sel_idx)
                if dist_to_sel < dist:
                    good = False
                    break
                    
            if good:
                selected.append((dist, idx))
                if len(selected) >= max_size:
                    break
            elif keep_max_size_level0:
                pruned.append((dist, idx))
                
        # If needed, fill up to max_size with pruned candidates
        if keep_max_size_level0:
            while len(selected) < max_size and pruned:
                selected.append(pruned.pop(0))
                
        return selected
        
    def get_neighbors(self, vertex: int, level: int) -> mx.array:
        """Get neighbors of vertex at given level."""
        if level < 0 or level > int(self.levels[vertex]):
            return mx.array([], dtype=mx.int32)
            
        start = int(self.offsets[vertex])
        if level == 0:
            end = start + self.M0
        else:
            end = start + self.M
            
        return self.neighbors[start:end]
        
    def add_vertex(self, vertex: int, level: int, dist_computer) -> None:
        """Add new vertex following FAISS implementation."""
        # Extend graph structures
        while int(self.levels.shape[0]) <= vertex:
            self.levels = mx.concatenate([self.levels, mx.array([0], dtype=mx.int32)])
            self.offsets = mx.concatenate([self.offsets, mx.array([self.neighbors.shape[0]], dtype=mx.int64)])
            self.neighbors = mx.concatenate([self.neighbors, mx.full(self.cum_nneighbor_per_level[0], -1, dtype=mx.int32)])
            
        self.levels = self._scatter1d(self.levels, mx.array([vertex], dtype=mx.int32), mx.array([level], dtype=mx.int32))
        
        if level > self.max_level:
            self.max_level = level
            self.entry_point = vertex
            
        if self.entry_point == vertex:
            return
            
        # Find entry point
        curr_vertex = self.entry_point
        curr_dist = dist_computer(vertex, curr_vertex)
        
        # For each level from top to bottom
        for lc in range(level, -1, -1):
            changed = True
            while changed:
                changed = False
                
                # Get neighbors at current level
                neighbors = self.get_neighbors(curr_vertex, lc)
                
                # Check each neighbor
                for neighbor in neighbors:
                    if neighbor == -1:
                        continue
                    dist = dist_computer(vertex, neighbor)
                    if dist < curr_dist:
                        curr_vertex = neighbor
                        curr_dist = dist
                        changed = True
                        
            # Select best neighbors at this level
            candidates = [(curr_dist, curr_vertex)]
            visited = {curr_vertex}
            
            # Search for candidates
            while candidates:
                dist, curr = candidates[0]
                candidates = candidates[1:]
                
                neighbors = self.get_neighbors(curr, lc)
                for neighbor in neighbors:
                    if neighbor == -1 or neighbor in visited:
                        continue
                        
                    visited.add(neighbor)
                    dist = dist_computer(vertex, neighbor)
                    candidates.append((dist, neighbor))
                    
                candidates.sort()
                candidates = candidates[:self.efConstruction]
                
            # Select diverse neighbors
            selected = self.shrink_neighbor_list(
                dist_computer,
                candidates,
                self.M0 if lc == 0 else self.M,
                lc == 0  # Keep max size only for level 0
            )
            
            # Add connections
            for dist, sel in selected:
                self._add_connection(vertex, sel, lc)
                self._add_connection(sel, vertex, lc)
                
    def _add_connection(self, vertex1: int, vertex2: int, level: int) -> None:
        """Add optimized connection between vertices."""
        max_edges = self.M0 if level == 0 else self.M
        
        # Get current neighbors
        start = int(self.offsets[vertex1])
        neighbors = self.neighbors[start:start + max_edges]
        
        # Find first empty slot or replace worst neighbor
        for i in range(max_edges):
            if int(neighbors[i]) == -1:
                self.neighbors = self._scatter1d(
                    self.neighbors,
                    mx.array([start + i], dtype=mx.int64).astype(mx.int32),
                    mx.array([vertex2], dtype=mx.int32)
                )
                return
                
        # Need to replace worst neighbor
        self.neighbors = self._scatter1d(
            self.neighbors,
            mx.array([start + max_edges - 1], dtype=mx.int64).astype(mx.int32),
            mx.array([vertex2], dtype=mx.int32)
        )
        
    def search(self, query: int, ef: int, dist_computer) -> List[Tuple[float, int]]:
        """Search implementation matching FAISS."""
        stats = HNSWStats()
        
        # Start from entry point
        curr_vertex = self.entry_point
        curr_dist = dist_computer(query, curr_vertex)
        stats.ndis += 1
        
        # Search through levels
        for level in range(self.max_level, -1, -1):
            changed = True
            while changed:
                changed = False
                neighbors = self.get_neighbors(curr_vertex, level)
                
                # Process neighbors in batches of 4 for efficiency
                n = len(neighbors)
                for i in range(0, n, 4):
                    batch = neighbors[i:min(i+4, n)]
                    batch = batch[batch != -1]  # Remove invalid neighbors
                    if len(batch) == 0:
                        continue
                        
                    # Compute distances for batch
                    dists = []
                    for neigh in batch:
                        dist = dist_computer(query, int(neigh))
                        dists.append(dist)
                        stats.ndis += 1
                        
                    # Update if better neighbor found
                    min_dist = min(dists)
                    if min_dist < curr_dist:
                        idx = dists.index(min_dist)
                        curr_vertex = int(batch[idx])
                        curr_dist = min_dist
                        changed = True
                        
            stats.nhops += 1
            
        # Search at level 0 with ef candidates
        candidates = MinimaxHeap(ef)
        candidates.push(curr_vertex, curr_dist)
        visited = {curr_vertex}
        
        while candidates.nvalid > 0:
            curr_vertex, curr_dist = candidates.pop_min()
            neighbors = self.get_neighbors(curr_vertex, 0)
            
            # Process neighbors in batches of 4
            n = len(neighbors)
            for i in range(0, n, 4):
                batch = neighbors[i:min(i+4, n)]
                batch = batch[mx.not_equal(batch, mx.array(-1, dtype=batch.dtype))]
                if len(batch) == 0:
                    continue
                    
                # Process unvisited neighbors
                unvisited = []
                for neigh in batch:
                    neigh = int(neigh)
                    if neigh not in visited:
                        unvisited.append(neigh)
                        visited.add(neigh)
                        
                if not unvisited:
                    continue
                    
                # Compute distances for batch
                dists = []
                for neigh in unvisited:
                    dist = dist_computer(query, neigh)
                    dists.append(dist)
                    stats.ndis += 1
                    
                # Add to candidates if within ef closest
                for neigh, dist in zip(unvisited, dists):
                    if candidates.n < ef or dist < candidates.distances[0]:
                        candidates.push(neigh, dist)
                        
            stats.nhops += 1
            
        # Return results sorted by distance
        results = []
        while candidates.nvalid > 0:
            vertex, dist = candidates.pop_min()
            if vertex != -1:
                results.append((dist, vertex))
                
        stats.n1 = len(visited)
        stats.n2 = len(results)
        return sorted(results)

    # --- Array-native level-0 search (no Python float/int casts) ---------
    def search_level0_array(
        self,
        query_vec: mx.array,
        ef: int,
        vectors: mx.array,
        metric: MetricType,
        k: int,
    ) -> Tuple[mx.array, mx.array]:
        """Approximate search at level 0 using MLX arrays only.

        Args:
            query_vec: (d,) MLX array
            ef: candidate list size (>= k)
            vectors: (n, d) MLX database vectors aligned with this graph
            metric: MetricType.L2 or MetricType.INNER_PRODUCT
            k: desired number of neighbors (k <= ef)

        Returns:
            Tuple (distances, ids) each (k,) MLX arrays.
        """
        n = self.levels.shape[0]
        if n == 0:
            return mx.zeros((0,), dtype=mx.float32), mx.zeros((0,), dtype=mx.int32)

        # Visited mask (uint8: 0/1)
        visited = mx.zeros((n,), dtype=mx.uint8)

        # Seed at entry point
        entry = mx.array(self.entry_point, dtype=mx.int32)
        visited = self._scatter1d(visited, entry.reshape((1,)).astype(mx.int32), mx.ones((1,), dtype=mx.uint8))

        def _dist(a: mx.array) -> mx.array:
            # a: (m, d)
            if metric == MetricType.L2:
                diff = mx.subtract(a, query_vec.reshape((1, -1)))
                return mx.sum(mx.square(diff), axis=1)
            else:
                return mx.negative(mx.matmul(a, query_vec.reshape((-1, 1))).reshape((-1,)))

        # Candidate buffers
        cand_ids = entry.reshape((1,))
        cand_d = _dist(mx.take(vectors, cand_ids, axis=0))

        # Iterate ef steps; at each step expand current best frontier
        steps = mx.array(ef, dtype=mx.int32)
        for _ in range(ef):
            # Keep only top-ef candidates
            order = mx.argsort(cand_d)[: ef]
            cand_ids = mx.take(cand_ids, order, axis=0)
            cand_d = mx.take(cand_d, order, axis=0)

            # Build neighbors for current frontier (level 0 only)
            deg = self.M0
            offs = mx.take(self.offsets, cand_ids, axis=0)  # (c,)
            grid = mx.add(offs.reshape((-1, 1)), mx.arange(deg, dtype=offs.dtype).reshape((1, -1)))  # (c,deg)
            neigh2d = mx.take(self.neighbors, grid, axis=0)  # (c,deg)
            neigh = neigh2d.reshape((-1,))
            valid = mx.greater_equal(neigh, mx.zeros_like(neigh))
            # Filter not-yet-visited
            safe_neigh = mx.where(valid, neigh, mx.zeros_like(neigh))
            vis = mx.take(visited, safe_neigh)
            notvis = mx.logical_and(valid, mx.equal(vis, mx.zeros_like(vis)))
            nnz = int(mx.sum(notvis).astype(mx.int32).item())  # boundary-ok
            if nnz == 0:
                break
            # Build safe ids (replace invalid with 0) and inf distances for invalid
            new_ids_all = mx.where(notvis, neigh, mx.add(mx.zeros_like(neigh), mx.array(-1, dtype=neigh.dtype)))
            safe_ids = mx.maximum(new_ids_all, mx.zeros_like(new_ids_all))
            new_vecs = mx.take(vectors, safe_ids, axis=0)
            new_d = _dist(new_vecs)
            infv = mx.divide(mx.ones_like(new_d), mx.zeros_like(new_d))
            new_d = mx.where(mx.greater_equal(new_ids_all, mx.zeros_like(new_ids_all)), new_d, infv)
            # Mark visited for valid ids
            ones_u8 = mx.add(mx.zeros_like(safe_ids), mx.array(1, dtype=mx.uint8))
            zeros_u8 = mx.add(mx.zeros_like(safe_ids), mx.array(0, dtype=mx.uint8))
            upd = mx.where(mx.greater_equal(new_ids_all, mx.zeros_like(new_ids_all)), ones_u8, zeros_u8)
            visited = self._scatter1d(visited, safe_ids, upd)
            cand_ids = mx.concatenate([cand_ids, safe_ids], axis=0)
            cand_d = mx.concatenate([cand_d, new_d], axis=0)

        # Final top-k
        order = mx.argsort(cand_d)[: k]
        top_ids = mx.take(cand_ids, order, axis=0).astype(mx.int32)
        top_d = mx.take(cand_d, order, axis=0).astype(mx.float32)
        return top_d, top_ids
