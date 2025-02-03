from __future__ import annotations  # Added for forward refs
from typing import List, Optional, Tuple
import numpy as np

class RangeSearchResult:
    """Container for range search results that minimizes memory copies."""
    
    def __init__(self, nq: int, alloc_lims: bool = True):
        """Initialize range search result.
        
        Args:
            nq: Number of queries
            alloc_lims: Whether to allocate lims array
        """
        self.nq = nq
        self.lims = np.zeros(nq + 1, dtype=np.int64) if alloc_lims else None
        self.labels: Optional[np.ndarray] = None
        self.distances: Optional[np.ndarray] = None
        self.buffer_size: int = 0
        
    def do_allocation(self):
        """Allocate result arrays based on lims."""
        if self.lims is None:
            raise RuntimeError("lims must be set before allocation")
        total = int(self.lims[-1])
        self.buffer_size = total
        self.labels = np.empty(total, dtype=np.int64)
        self.distances = np.empty(total, dtype=np.float32)


class BufferList:
    """List of temporary buffers for storing results."""
    
    def __init__(self, buffer_size: int):
        """Initialize buffer list.
        
        Args:
            buffer_size: Size of each buffer in entries
        """
        self.buffer_size = buffer_size
        self.buffers: List[Tuple[np.ndarray, np.ndarray]] = []  # Added type hint
        self.wp: int = 0
        
    def append_buffer(self):
        """Create a new buffer."""
        self.buffers.append((
            np.empty(self.buffer_size, dtype=np.int64),
            np.empty(self.buffer_size, dtype=np.float32)
        ))
        self.wp = 0
        
    def add(self, id: int, dis: float):
        """Add one result, appending buffer if needed.
        
        Args:
            id: Vector ID 
            dis: Distance value
        """
        if not self.buffers or self.wp >= self.buffer_size:
            self.append_buffer()
            
        buf = self.buffers[-1]
        buf[0][self.wp] = id
        buf[1][self.wp] = dis
        self.wp += 1
        
    def copy_range(self, ofs: int, n: int, dest_ids: np.ndarray, dest_dis: np.ndarray):
        """Copy range of results to destination arrays.
        
        Args:
            ofs: Start offset
            n: Number of elements
            dest_ids: Destination IDs array
            dest_dis: Destination distances array 
        """
        start_buf = ofs // self.buffer_size
        in_buf_ofs = ofs % self.buffer_size
        
        copied = 0
        while copied < n:
            buf = self.buffers[start_buf]
            to_copy = min(self.buffer_size - in_buf_ofs, n - copied)
            
            dest_ids[copied:copied + to_copy] = buf[0][in_buf_ofs:in_buf_ofs + to_copy]
            dest_dis[copied:copied + to_copy] = buf[1][in_buf_ofs:in_buf_ofs + to_copy]
            
            copied += to_copy
            start_buf += 1
            in_buf_ofs = 0


class RangeQueryResult:
    """Result structure for a single range query."""
    
    def __init__(self, qno: int, pres: 'RangeSearchPartialResult'):
        """Initialize query result.
        
        Args:
            qno: Query number
            pres: Parent partial result
        """
        self.qno = qno
        self.nres = 0
        self.pres = pres
        
    def add(self, dis: float, id: int):
        """Add a new result.
        
        Args:
            dis: Distance value
            id: Vector ID
        """
        self.pres.add(id, dis)
        self.nres += 1


class RangeSearchPartialResult(BufferList):
    """Partial results for range search with per-query splitting."""
    
    def __init__(self, res: RangeSearchResult):
        """Initialize partial result.
        
        Args:
            res: Final result container
        """
        super().__init__(1024)  # Default buffer size
        self.res = res
        self.queries: List[RangeQueryResult] = []
        
    def new_result(self, qno: int) -> RangeQueryResult:
        """Begin a new query result.
        
        Args:
            qno: Query number
            
        Returns:
            New query result
        """
        qres = RangeQueryResult(qno, self)
        self.queries.append(qres)
        return qres
        
    def finalize(self):
        """Merge partial results into final result."""
        self.res.lims[0] = 0
        for i, qres in enumerate(self.queries):
            self.res.lims[i + 1] = self.res.lims[i] + qres.nres
            
    def copy_result(self, incremental: bool = False):
        """Copy results to final container.
        
        Args:
            incremental: Whether copying incrementally
        """
        if not incremental:
            self.finalize()
            self.res.do_allocation()
            
        for i, qres in enumerate(self.queries):
            start = self.res.lims[i]
            n = qres.nres
            self.copy_range(
                int(start), 
                n,
                self.res.labels[start:start + n],
                self.res.distances[start:start + n]
            )
