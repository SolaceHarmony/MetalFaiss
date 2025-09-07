"""
simple_transforms.py - Simple vector transforms

This implements several basic vector transforms from FAISS:
- RemapDimensionsTransform: Remap or subset dimensions
- NormalizationTransform: Per-vector normalization
- CenteringTransform: Subtract mean from vectors

These transforms provide common preprocessing operations that can be
combined with more complex transforms.

Original: faiss/VectorTransform.h
"""

import mlx.core as mx
from typing import Optional, List, Union
from .base_vector_transform import BaseVectorTransform
from ..errors import InvalidArgumentError

class RemapDimensionsTransform(BaseVectorTransform):
    """Remap dimensions for input vectors.
    
    This transform can:
    1. Reorder dimensions using a custom mapping
    2. Select a subset of dimensions
    3. Insert zeros for missing dimensions
    4. Distribute dimensions uniformly
    
    The mapping is specified as indices into the input dimensions,
    with -1 indicating dimensions that should be set to 0.
    """
    
    def __init__(
        self,
        d_in: int,
        d_out_or_map: Union[int, List[int]],
        dimension_map: Optional[List[int]] = None,
        uniform: bool = True
    ):
        """Initialize dimension remapping.
        
        Args:
            d_in: Input dimension
            d_out: Output dimension
            dimension_map: Optional explicit mapping from output to input dims.
                Each value should be in range [-1, d_in) where -1 means output 0.
                If None, creates uniform or prefix mapping based on uniform arg.
            uniform: If True and dimension_map=None, distribute dimensions
                uniformly. If False, take first d_out dimensions.
                
        Raises:
            InvalidArgumentError: If dimension_map contains invalid indices
        """
        # Support legacy usage: RemapDimensionsTransform(d_in, indices)
        if isinstance(d_out_or_map, list):
            d_out = len(d_out_or_map)
            dimension_map = d_out_or_map
        else:
            d_out = d_out_or_map
        super().__init__(d_in, d_out)
        
        if dimension_map is not None:
            # Validate mapping
            if any(i >= d_in for i in dimension_map if i >= 0):
                raise InvalidArgumentError(
                    f"Dimension map contains index >= input dimension {d_in}"
                )
            self.map = dimension_map
        else:
            self.map = self._create_uniform_map() if uniform else self._create_prefix_map()
            
        self._is_trained = True
        
    def _create_uniform_map(self) -> List[int]:
        """Create uniform dimension mapping.
        
        Returns:
            List mapping output dims to input dims, distributing uniformly
        """
        if self.d_out <= self.d_in:
            # Subsample input dimensions
            step = self.d_in / self.d_out
            return [int(i * step) for i in range(self.d_out)]
        else:
            # Repeat input dimensions with padding
            repeats = self.d_out // self.d_in
            remainder = self.d_out % self.d_in
            mapping = []
            for _ in range(repeats):
                mapping.extend(range(self.d_in))
            mapping.extend(range(remainder))
            return mapping
            
    def _create_prefix_map(self) -> List[int]:
        """Create prefix dimension mapping.
        
        Returns:
            List mapping output dims to input dims, taking prefix
        """
        if self.d_out <= self.d_in:
            # Take first d_out dimensions
            return list(range(self.d_out))
        else:
            # Take all input dims then pad with -1
            return list(range(self.d_in)) + [-1] * (self.d_out - self.d_in)
            
    def apply_noalloc(self, x: mx.array, xt: mx.array) -> None:
        """Apply dimension remapping.
        
        Args:
            x: Input vectors (n, d_in)
            xt: Output buffer (n, d_out)
        """
        # Handle each output dimension
        for i, src_dim in enumerate(self.map):
            if src_dim >= 0:
                # Copy from input dimension
                xt[:, i] = x[:, src_dim]
            else:
                # Set to 0
                xt[:, i] = 0
                
    def reverse_transform(self, xs: List[List[float]]) -> List[List[float]]:
        """Reverse transform (only for permutation mappings).
        
        Args:
            xs: Input vectors
            
        Returns:
            Reverse transformed vectors
            
        Raises:
            ValueError: If mapping is not a permutation
        """
        # Check if mapping is a permutation
        used_dims = set(i for i in self.map if i >= 0)
        if len(used_dims) != self.d_in or -1 in self.map:
            raise ValueError(
                "Reverse transform only supported for permutation mappings"
            )
            
        # Create reverse mapping
        reverse_map = [-1] * self.d_in
        for i, src_dim in enumerate(self.map):
            if src_dim >= 0:
                reverse_map[src_dim] = i
                
        # Apply reverse mapping
        x = mx.array(xs)
        xt = mx.zeros((len(x), self.d_in))
        for i, src_dim in enumerate(reverse_map):
            xt[:, i] = x[:, src_dim]
            
        return xt.tolist()
        
    def check_identical(self, other: BaseVectorTransform) -> None:
        """Check if transforms are identical.
        
        Args:
            other: Transform to compare with. Must be RemapDimensionsTransform.
            
        Raises:
            ValueError: If transforms are not identical or if other is not
                a RemapDimensionsTransform
        """
        super().check_identical(other)
        
        if not isinstance(other, RemapDimensionsTransform):
            raise ValueError("Not a dimension remapping transform")
            
        if self.map != other.map:
            raise ValueError("Dimension mappings do not match")

class NormalizationTransform(BaseVectorTransform):
    """Per-vector normalization transform.
    
    This normalizes each vector to have unit norm according to the
    specified norm type (e.g. L2 norm). The transform is not reversible
    since the original magnitudes are lost.
    """
    
    def __init__(self, d: int, norm: float = 2.0):
        """Initialize normalization transform.
        
        Args:
            d: Input/output dimension
            norm: Type of norm to use (e.g. 2.0 for L2 norm)
        """
        super().__init__(d, d)
        self.norm = norm
        self._is_trained = True
        
    def apply_noalloc(self, x: mx.array, xt: mx.array) -> None:
        """Apply normalization.
        
        Args:
            x: Input vectors (n, d)
            xt: Output buffer (n, d)
        """
        # Compute norms
        p = mx.array(self.norm, dtype=x.dtype)
        absx = mx.abs(x)
        norms = mx.sum(mx.power(absx, p), axis=1)
        invp = mx.divide(mx.array(1.0, dtype=x.dtype), p)
        norms = mx.power(norms, invp)
        norms = mx.maximum(norms, mx.array(1e-10, dtype=x.dtype))  # Avoid division by 0
        # Normalize
        xt[:] = mx.divide(x, norms.reshape(-1, 1))
        
    def reverse_transform(self, xs: List[List[float]]) -> List[List[float]]:
        """Reverse transform (identity since norm is lost).
        
        Args:
            xs: Input vectors
            
        Returns:
            Input vectors unchanged
        """
        return xs
        
    def check_identical(self, other: BaseVectorTransform) -> None:
        """Check if transforms are identical.
        
        Args:
            other: Transform to compare with. Must be NormalizationTransform.
            
        Raises:
            ValueError: If transforms are not identical or if other is not
                a NormalizationTransform
        """
        super().check_identical(other)
        
        if not isinstance(other, NormalizationTransform):
            raise ValueError("Not a normalization transform")
            
        if self.norm != other.norm:
            raise ValueError("Norm types do not match")

class CenteringTransform(BaseVectorTransform):
    """Subtract mean from vectors.
    
    This centers the data by subtracting the mean of each dimension.
    The mean is computed during training from representative vectors.
    """
    
    def __init__(self, d: int):
        """Initialize centering transform.
        
        Args:
            d: Input/output dimension
        """
        super().__init__(d, d)
        self.mean: Optional[mx.array] = None
        self._is_trained = False
        
    def train(self, xs: List[List[float]]) -> None:
        """Train transform by computing mean.
        
        Args:
            xs: Training vectors
        """
        x = mx.array(xs)
        self.mean = mx.mean(x, axis=0)
        self._is_trained = True
        
    def apply_noalloc(self, x: mx.array, xt: mx.array) -> None:
        """Apply centering by subtracting mean.
        
        Args:
            x: Input vectors (n, d)
            xt: Output buffer (n, d)
            
        Raises:
            RuntimeError: If transform not trained
        """
        if not self.is_trained or self.mean is None:
            raise RuntimeError("Transform must be trained before applying")
            
        xt[:] = x - self.mean
        
    def reverse_transform(self, xs: List[List[float]]) -> List[List[float]]:
        """Reverse transform by adding mean back.
        
        Args:
            xs: Input vectors
            
        Returns:
            Reverse transformed vectors
            
        Raises:
            RuntimeError: If transform not trained
        """
        if not self.is_trained or self.mean is None:
            raise RuntimeError("Transform must be trained before reversing")
            
        x = mx.array(xs)
        return (x + self.mean).tolist()
        
    def check_identical(self, other: BaseVectorTransform) -> None:
        """Check if transforms are identical.
        
        Args:
            other: Transform to compare with. Must be CenteringTransform.
            
        Raises:
            ValueError: If transforms are not identical or if other is not
                a CenteringTransform
        """
        super().check_identical(other)
        
        if not isinstance(other, CenteringTransform):
            raise ValueError("Not a centering transform")
            
        if not mx.all(self.mean == other.mean):
            raise ValueError("Means do not match")
