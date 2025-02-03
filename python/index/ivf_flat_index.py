from typing import List, Optional
import mlx.core as mx
from .base_index import BaseIndex
from .flat_index import FlatIndex
from .search_result import SearchResult
from ..metric_type import MetricType
from ..errors import IndexError

class IVFFlatIndex(BaseIndex):
    def __init__(self, 
                 quantizer: FlatIndex, 
                 d: int, 
                 nlist: int, 
                 metric_type: MetricType = MetricType.L2):
        super().__init__(d)
        self._quantizer = quantizer
        self._metric_type = metric_type
        self._nlist = nlist
        self._nprobe = 1
        self._invlists = [[] for _ in range(nlist)]
        self._is_trained = False

    @staticmethod 
    def from_(index_pointer) -> Optional['IVFFlatIndex']:
        """Swift compatibility: Create from index pointer"""
        return index_pointer if isinstance(index_pointer, IVFFlatIndex) else None

    @property
    def index_pointer(self):
        """Swift compatibility: Get index pointer"""
        return self

    @property
    def nprobe(self) -> int:
        return self._nprobe

    @nprobe.setter
    def nprobe(self, value: int) -> None:
        if value < 1:
            raise ValueError("nprobe must be positive")
        self._nprobe = value

    @property 
    def nlist(self) -> int:
        return self._nlist

    @property
    def quantizer(self) -> FlatIndex:
        return self._quantizer

    def train(self, xs: List[List[float]]) -> None:
        """Train the index.
        
        Args:
            xs: Training vectors
        """
        if not xs:
            raise ValueError("Empty training data")
            
        x = mx.array(xs, dtype=mx.float32)
        if x.shape[1] != self.d:
            raise ValueError(f"Data dimension {x.shape[1]} does not match index dimension {self.d}")
            
        # Train quantizer
        self._quantizer.train(xs)
        self._centroids = mx.array(self._quantizer.xb(), dtype=mx.float32)
        self._is_trained = True

    def add(self, xs: List[List[float]], ids: Optional[List[int]] = None) -> None:
        """Add vectors to the index.
        
        Args:
            xs: Vectors to add
            ids: Optional vector IDs
        """
        if not self.is_trained:
            raise RuntimeError("Index must be trained before adding vectors")
            
        x = mx.array(xs, dtype=mx.float32)
        if x.shape[1] != self.d:
            raise ValueError(f"Data dimension {x.shape[1]} does not match index dimension {self.d}")
            
        # Assign vectors to lists
        assignments = self._quantizer.search(xs, 1)
        for i, label in enumerate(assignments.labels):
            self._lists[label[0]].append((ids[i] if ids else i, x[i]))
            
        self._ntotal += len(x)

    def search(self, xs: List[List[float]], k: int) -> SearchResult:
        """Search for nearest neighbors.
        
        Args:
            xs: Query vectors
            k: Number of nearest neighbors
            
        Returns:
            SearchResult containing distances and labels
        """
        if not self.is_trained:
            raise RuntimeError("Index must be trained before searching")
            
        x = mx.array(xs, dtype=mx.float32)
        if x.shape[1] != self.d:
            raise ValueError(f"Query dimension {x.shape[1]} does not match index dimension {self.d}")
            
        # Find nearest lists using quantizer
        coarse_dists, coarse_labels = self._quantizer.search(xs, self.nprobe)
        
        # Search within selected lists
        distances = []
        labels = []
        for query, probe_labels in zip(x, coarse_labels):
            probe_vectors = []
            probe_ids = []
            
            # Gather vectors from selected lists
            for list_id in probe_labels:
                for vid, vec in self._lists[list_id]:
                    probe_vectors.append(vec)
                    probe_ids.append(vid)
                    
            if not probe_vectors:
                distances.append([float('inf')] * k)
                labels.append([0] * k)
                continue
                
            probe_vectors = mx.stack(probe_vectors)
            if self.metric_type == MetricType.L2:
                dists = mx.sum((probe_vectors - query) ** 2, axis=1)
            else:
                dists = -mx.sum(probe_vectors * query, axis=1)
                
            idx = mx.argsort(dists)[:k]
            distances.append(dists[idx].tolist())
            labels.append([probe_ids[i] for i in idx.tolist()])
            
        return SearchResult(distances=distances, labels=labels)

    def reset(self) -> None:
        """Remove all vectors from the index."""
        super().reset()
        self._invlists = [[] for _ in range(self._nlist)]
        self._quantizer.reset()
