# MetalFaiss - A pure Python implementation of FAISS using MLX for Metal acceleration
# Copyright (c) 2024 Sydney Bach, The Solace Project
# Licensed under the Apache License, Version 2.0 (see LICENSE file)

from __future__ import annotations

"""
Index Factory System for MetalFaiss

Provides string-based index creation similar to faiss.index_factory().
Enables easy experimentation and user-friendly API for creating complex indexes.

Examples:
    index = index_factory(128, "Flat")
    index = index_factory(128, "IVF100,Flat") 
    index = index_factory(128, "IVF100,PQ8")
    index = index_factory(128, "HNSW32")
    index = index_factory(128, "PQ8")
"""

import re
from typing import Optional, Union, Any, Dict
import mlx.core as mx

from .metric_type import MetricType
from .errors import InvalidArgumentError

# Modern index modules
from .index.base_index import BaseIndex
from .index.flat_index import FlatIndex
from .index.ivf_flat_index import IVFFlatIndex
from .index.ivf_index import IVFIndex
from .index.ivf_pq_index import IVFPQIndex
from .index.scalar_quantizer_index import ScalarQuantizerIndex
from .index.product_quantizer_index import ProductQuantizerIndex
from .index.hnsw_index import HNSWFlatIndex
from .index.lsh_index import LSHIndex
from .index.id_map import IDMap
from .index.pre_transform_index import PreTransformIndex

# Vector transforms (names normalized here)
from .vector_transform.pca_matrix import PCAMatrixTransform
from .vector_transform.opq import OPQTransform
from .vector_transform.itq import ITQTransform
from .vector_transform.random_rotation import RandomRotationTransform


class IndexFactory:
    """
    Factory class for creating indexes from string descriptions.
    
    Supports the following index types:
    - Flat: Brute-force exact search
    - IVF{nlist},Flat: Inverted file with flat quantizer
    - IVF{nlist},PQ{m}: IVF with Product Quantization
    - IVF{nlist},SQ{bits}: IVF with Scalar Quantization
    - PQ{m}: Product Quantization
    - SQ{bits}: Scalar Quantization  
    - HNSW{M}: Hierarchical NSW graph
    - LSH: Locality Sensitive Hashing
    - IDMap,{base}: ID mapping wrapper
    - PCA{dim},{base}: PCA preprocessing
    - OPQ{dim},{base}: Optimized Product Quantization preprocessing
    """
    
    # Scalar quantizer type mappings
    SQ_TYPES = {
        'SQ4': 4,
        'SQ6': 6, 
        'SQ8': 8,
        'SQ16': 16,
        'SQfp16': 'fp16',
        'SQbf16': 'bf16',
    }
    
    @classmethod
    def create_index(
        cls, d: int, description: str, metric: Union[MetricType, str] = MetricType.L2
    ) -> BaseIndex:
        """Create an index from a factory string (subset of Faiss grammar).

        Supports:
        - Prefix vector transforms (PCA, PCAR, PCAW, PCAWR, OPQ, ITQ, RR, Pad, L2norm)
        - Base indexes: Flat, IVF{nlist}[,Flat], HNSW{M}, PQ{m}, SQ{bits|fp16|bf16}
        - Suffix wrappers: IDMap, IDMap2, RFlat, Refine(Flat)
        """
        if isinstance(metric, str):
            metric = MetricType.from_string(metric)

        description = description.strip()

        # Tokenize by top-level commas (no parentheses use in our supported subset)
        tokens = [t.strip() for t in description.split(',') if t.strip()]
        if not tokens:
            raise InvalidArgumentError("Empty factory description")

        # Handle suffix wrappers at the end (IDMap/IDMap2/RFlat/Refine(...))
        suffixes: list[str] = []
        while tokens:
            last = tokens[-1]
            if last in ("IDMap", "IDMap2", "RFlat") or last.startswith("Refine("):
                suffixes.append(tokens.pop())
            else:
                break
        suffixes.reverse()  # apply in textual order

        # Handle optional IDMap/IDMap2 prefix wrapper (limited support: IDMap,Flat)
        prefix_wrapper: Optional[str] = None
        if tokens and tokens[0] in ("IDMap", "IDMap2"):
            prefix_wrapper = tokens.pop(0)

        # Handle prefix transforms
        transforms: list[Any] = []
        while tokens:
            head = tokens[0]
            if cls._is_transform_token(head):
                ops = cls._create_transform(d, head)
                if isinstance(ops, list):
                    transforms.extend(ops)
                else:
                    transforms.append(ops)
                tokens.pop(0)
            else:
                break

        if not tokens:
            # Allow shorthand 'RFlat' that implies Refine(Flat)
            if suffixes and (suffixes[0] == 'RFlat' or (suffixes[0].startswith('Refine(') and suffixes[0].endswith(')'))):
                base = FlatIndex(d, metric_type=metric)
            else:
                raise InvalidArgumentError("Missing base index in factory string")
        else:
            # Parse base index (support IVF with optional 'Flat')
            if tokens[0].startswith('IVF'):
                nlist = cls._extract_number(tokens[0], 'IVF', default=100)
                quant = FlatIndex(d, metric_type=metric)
                base = IVFFlatIndex(quantizer=quant, d=d, nlist=nlist)
                # Optional explicit Flat token is ignored here
                if len(tokens) >= 2 and tokens[1] == 'Flat':
                    tokens = tokens[2:]
                else:
                    tokens = tokens[1:]
            else:
                # Single-token base
                base = cls._parse_simple_index(d, tokens[0], metric)
                tokens = tokens[1:]

        if tokens:
            raise InvalidArgumentError(f"Unexpected tokens after base index: {tokens}")

        # Apply prefix wrapper if present (only IDMap/IDMap2 with Flat base supported for now)
        if prefix_wrapper is not None:
            if prefix_wrapper == 'IDMap':
                base = IDMap(base)
            elif prefix_wrapper == 'IDMap2':
                try:
                    from .index.id_map import IDMap2 as _IDMap2
                    base = _IDMap2(base)
                except Exception as e:
                    raise InvalidArgumentError(f"IDMap2 unsupported: {e}")

        # Apply suffix wrappers
        for s in suffixes:
            if s == 'IDMap':
                base = IDMap(base)
            elif s == 'IDMap2':
                try:
                    from .index.id_map import IDMap2 as _IDMap2
                    base = _IDMap2(base)
                except Exception as e:
                    raise InvalidArgumentError(f"IDMap2 unsupported: {e}")
            elif s == 'RFlat':
                from .index.refine_flat_index import RefineFlatIndex
                base = RefineFlatIndex(base)
            elif s.startswith('Refine(') and s.endswith(')'):
                inner = s[len('Refine('):-1].strip()
                if inner != 'Flat':
                    raise InvalidArgumentError(f"Refine() only supports Flat base for now, got {inner}")
                from .index.refine_flat_index import RefineFlatIndex
                base = RefineFlatIndex(base)
            elif s == 'IDMap' or s == 'IDMap2':
                # Prefix forms IDMap,Flat handled separately; suffix forms wrap base
                continue
            else:
                raise InvalidArgumentError(f"Unknown suffix token: {s}")

        # Apply transforms as outer wrappers (first transform is outermost)
        for t in reversed(transforms):
            base = PreTransformIndex(transform=t, index=base)

        return base
    
    @classmethod
    def _parse_composite_index(
        cls, 
        d: int, 
        description: str, 
        metric: MetricType
    ) -> BaseIndex:
        """Parse composite index descriptions like IVF100,PQ8."""
        parts = [part.strip() for part in description.split(',')]
        
        if len(parts) != 2:
            raise InvalidArgumentError(f"Invalid composite description: {description}")
        
        first_part, second_part = parts
        
        # Handle IVF + quantizer combinations
        if first_part.startswith('IVF'):
            nlist = cls._extract_number(first_part, 'IVF', default=100)
            
            if second_part == 'Flat':
                quant = FlatIndex(d, metric_type=metric)
                return IVFFlatIndex(quantizer=quant, d=d, nlist=nlist)
            elif second_part.startswith('PQ'):
                m = cls._extract_number(second_part, 'PQ', default=8)
                return IVFPQIndex(d=d, nlist=nlist, M=m, nbits=8, metric_type=metric)
            elif second_part.startswith('SQ'):
                # Not yet implemented in this codebase
                sq_type = cls._parse_sq_type(second_part)
                raise NotImplementedError(f"IVF+SQ ({sq_type}) is not implemented yet")
        
        # Handle preprocessing + index combinations
        elif first_part.startswith(('PCA', 'OPQ', 'ITQ', 'RR')):
            transform = cls._create_transform(d, first_part)
            base_index = cls._parse_simple_index(d, second_part, metric)
            return PreTransformIndex(transform=transform, index=base_index)
        
        # Handle IDMap/IDMap2 wrappers
        elif first_part in ('IDMap', 'IDMap2'):
            base_index = cls._parse_simple_index(d, second_part, metric)
            if first_part == 'IDMap2':
                try:
                    from .index.id_map import IDMap2 as _IDMap2
                    return _IDMap2(base_index)
                except Exception as e:
                    raise InvalidArgumentError(f"IDMap2 unsupported: {e}")
            return IDMap(base_index)
        
        raise InvalidArgumentError(f"Unsupported composite description: {description}")
    
    @classmethod
    def _parse_simple_index(
        cls, 
        d: int, 
        description: str, 
        metric: MetricType
    ) -> BaseIndex:
        """Parse simple index descriptions."""
        
        if description == 'Flat':
            return FlatIndex(d, metric_type=metric)
        if description == 'RFlat':
            from .index.refine_flat_index import RefineFlatIndex
            base = FlatIndex(d, metric_type=metric)
            return RefineFlatIndex(base)
        
        elif description.startswith('IVF'):
            nlist = cls._extract_number(description, 'IVF', default=100)
            quant = FlatIndex(d, metric_type=metric)
            return IVFFlatIndex(quantizer=quant, d=d, nlist=nlist)
        
        elif description.startswith('PQ'):
            m = cls._extract_number(description, 'PQ', default=8)
            return ProductQuantizerIndex(d, M=m, metric_type=metric)
        
        elif description.startswith('SQ'):
            qtype = cls._parse_sq_type(description)
            return ScalarQuantizerIndex(d, qtype=qtype)
        
        elif description.startswith('HNSW'):
            m = cls._extract_number(description, 'HNSW', default=16)
            return HNSWFlatIndex(d, M=m, metric_type=metric)
        elif description in ('IDMap', 'IDMap2'):
            # Single-token wrapper without base isn't valid here
            raise InvalidArgumentError("IDMap requires a base index, use 'IDMap,Flat' or suffix form")
        
        elif description == 'LSH':
            return LSHIndex(d, metric_type=metric)
        
        raise InvalidArgumentError(f"Unsupported index description: {description}")
    
    @classmethod
    def _is_transform_token(cls, token: str) -> bool:
        t = token.upper()
        return (
            t.startswith('PCA') or t.startswith('PCAR') or t.startswith('PCAW') or t.startswith('PCAWR') or
            t.startswith('OPQ') or t.startswith('ITQ') or t.startswith('RR') or
            t.startswith('PAD') or t == 'L2NORM'
        )

    @classmethod
    def _create_transform(cls, d: int, token: str):
        """Create vector transform from a transform token."""
        t = token.strip()
        tu = t.upper()
        # PCA family: PCA{dim}, PCAR{dim}, PCAW{dim}, PCAWR{dim}
        if tu.startswith('PCA'):
            # detect flags W (whitening) and R (random rotation)
            whitening = 'W' in tu[:5]  # PCAW or PCAWR
            random = 'R' in tu[:5] and not tu.startswith('PCAW')  # PCAR* or PCAWR*
            # When PCAWR, both apply
            if tu.startswith('PCAWR'):
                whitening = True; random = True
            # Extract digits at the end
            dim = cls._extract_number(''.join([c for c in t if c.isalnum()]), 'PCAWR' if tu.startswith('PCAWR') else ('PCAW' if tu.startswith('PCAW') else ('PCAR' if tu.startswith('PCAR') else 'PCA')), default=d//2)
            eigen_power = -1.0 if whitening else 0.0
            return [PCAMatrixTransform(d_in=d, d_out=dim, random_rotation=random, eigen_power=eigen_power)]
        if tu.startswith('OPQ'):
            # OPQm or OPQm_outdim with optional _outdim
            body = t[3:]
            M = None; outdim = d
            if '_' in body:
                m_s, out_s = body.split('_', 1)
                M = int(m_s) if m_s else None
                outdim = int(out_s)
            else:
                M = int(body) if body else None
            if M is None:
                M = 16
            ops = [OPQTransform(d_in=d, M=M)]
            if outdim != d:
                ops.append(PCAMatrixTransform(d_in=d, d_out=outdim))
            return ops
        if tu.startswith('ITQ'):
            num = t[3:]
            dim = int(num) if num else d
            return [ITQTransform(d_in=d, d_out=dim)]
        if tu.startswith('RR'):
            num = t[2:]
            dout = int(num) if num else d
            return [RandomRotationTransform(d_in=d, d_out=dout)]
        if tu.startswith('PAD'):
            dout = cls._extract_number(t, 'Pad', default=d)
            from .vector_transform.simple_transforms import RemapDimensionsTransform
            return [RemapDimensionsTransform(d_in=d, d_out_or_map=dout)]
        if tu == 'L2NORM':
            from .vector_transform.simple_transforms import NormalizationTransform
            return [NormalizationTransform(d=d)]
        else:
            raise InvalidArgumentError(f"Unknown transform: {description}")
    
    @classmethod
    def _extract_number(cls, text: str, prefix: str, default: int) -> int:
        """Extract number from string like 'IVF100' -> 100."""
        if text == prefix:
            return default
        
        number_part = text[len(prefix):]
        if not number_part:
            return default
            
        try:
            return int(number_part)
        except ValueError:
            raise InvalidArgumentError(f"Invalid number in '{text}'")
    
    @classmethod
    def _parse_sq_type(cls, description: str) -> str:
        """Parse scalar quantizer type from description into internal qtype string.

        Maps Faiss-style keys like 'SQ8' or 'SQfp16' to our ScalarQuantizerIndex qtype
        strings (e.g., 'QT_8bit', 'QT_fp16').
        """
        if description.lower() in ("sqfp16",):
            return "QT_fp16"
        if description.lower() in ("sqbf16",):
            return "QT_bf16"
        m = re.match(r"SQ(\d+)$", description)
        if m:
            bits = int(m.group(1))
            return f"QT_{bits}bit"
        raise InvalidArgumentError(f"Invalid scalar quantizer: {description}")


def index_factory(
    d: int, 
    description: str, 
    metric: Union[MetricType, str] = MetricType.L2
) -> BaseIndex:
    """
    Create an index from string description.
    
    This is the main entry point for string-based index creation,
    similar to faiss.index_factory().
    
    Args:
        d: Vector dimension
        description: String description of index type
        metric: Distance metric to use (default: L2)
        
    Returns:
        Configured index instance
        
    Examples:
        >>> index = index_factory(128, "Flat")
        >>> index = index_factory(128, "IVF100,PQ8") 
        >>> index = index_factory(128, "HNSW32")
        >>> index = index_factory(64, "PCA32,IVF50,Flat")
    """
    return IndexFactory.create_index(d, description, metric)


def reverse_factory(index: BaseIndex) -> str:
    """Return a factory string for a supported index instance.

    Covers: Flat, IVFFlat (Flat quantizer), HNSW, PQ, SQ, IDMap/IDMap2,
    and PreTransformIndex with PCA/OPQ/ITQ/RR + Flat. Raises InvalidArgumentError
    for unsupported types/combinations.
    """
    # Delayed imports to avoid cycles
    from .index.flat_index import FlatIndex as _Flat
    from .index.ivf_flat_index import IVFFlatIndex as _IVFFlat
    from .index.hnsw_index import HNSWIndex as _HNSW
    from .index.product_quantizer_index import ProductQuantizerIndex as _PQI
    from .index.scalar_quantizer_index import ScalarQuantizerIndex as _SQI
    from .index.id_map import IDMap as _IDMap
    try:
        from .index.id_map import IDMap2 as _IDMap2
    except Exception:
        _IDMap2 = None
    from .index.pre_transform_index import PreTransformIndex as _PTI
    from .vector_transform.pca_matrix import PCAMatrixTransform as _PCA
    from .vector_transform.opq import OPQTransform as _OPQ
    from .vector_transform.itq import ITQTransform as _ITQ
    from .vector_transform.random_rotation import RandomRotationTransform as _RR

    if isinstance(index, _Flat):
        return "Flat"
    if isinstance(index, _IVFFlat):
        # Only support Flat quantizer flavor in reverse for now
        return f"IVF{index.nlist},Flat"
    if isinstance(index, _HNSW):
        M = getattr(index.hnsw, 'M', 32)
        return f"HNSW{M}"
    if isinstance(index, _PQI):
        M = getattr(index.pq, 'M', None)
        if M is None:
            raise InvalidArgumentError("PQ index missing subquantizer count")
        return f"PQ{M}"
    if isinstance(index, _SQI):
        qtype = getattr(index, '_qtype', '')
        if qtype == 'QT_fp16':
            return 'SQfp16'
        if qtype == 'QT_bf16':
            return 'SQbf16'
        if qtype.startswith('QT_') and qtype.endswith('bit'):
            bits = qtype[len('QT_'):-len('bit')]
            return f"SQ{bits}"
        raise InvalidArgumentError(f"Unsupported SQ qtype: {qtype}")
    if _IDMap2 is not None and isinstance(index, _IDMap2):
        # Limit to Flat base for reverse
        return "IDMap2,Flat"
    if isinstance(index, _IDMap):
        return "IDMap,Flat"
    if isinstance(index, _PTI):
        # Unwrap transform chain: outermost -> innermost
        transforms = []
        base = index
        while isinstance(base, _PTI):
            transforms.append(base.transform)
            base = base.base_index
        base_key = reverse_factory(base)

        # Convert transform chain to tokens
        tokens: list[str] = []
        i = 0
        while i < len(transforms):
            t = transforms[i]
            # Compress OPQ + PCA(d_out) into OPQm_outdim
            if isinstance(t, _OPQ) and i + 1 < len(transforms) and isinstance(transforms[i + 1], _PCA):
                pca = transforms[i + 1]
                tokens.append(f"OPQ{t.M}_{pca.d_out}")
                i += 2
                continue
            if isinstance(t, _PCA):
                # Determine PCA variant
                # We used eigen_power = -1.0 for whitening; random_rotation flag for PCAR
                flags = "PCA"
                has_w = getattr(t, 'eigen_power', 0.0) != 0.0
                has_r = getattr(t, 'random_rotation', False)
                if has_w and has_r:
                    flags = "PCAWR"
                elif has_w:
                    flags = "PCAW"
                elif has_r:
                    flags = "PCAR"
                tokens.append(f"{flags}{t.d_out}")
            elif isinstance(t, _OPQ):
                tokens.append(f"OPQ{t.M}")
            elif isinstance(t, _ITQ):
                tokens.append(f"ITQ{t.d_out if t.d_out is not None else ''}")
            elif isinstance(t, _RR):
                tokens.append(f"RR{t.d_out}")
            else:
                # Other simple transforms supported
                from .vector_transform.simple_transforms import RemapDimensionsTransform as _Pad
                from .vector_transform.simple_transforms import NormalizationTransform as _Norm
                if isinstance(t, _Pad):
                    tokens.append(f"Pad{t.d_out}")
                elif isinstance(t, _Norm):
                    tokens.append("L2norm")
                else:
                    name = type(t).__name__
                    raise InvalidArgumentError(f"Unsupported transform in PreTransformIndex: {name}")
            i += 1
        return ",".join(tokens + [base_key])
    # Refine wrapper (RFlat)
    try:
        from .index.refine_flat_index import RefineFlatIndex as _RFlat
        if isinstance(index, _RFlat):
            inner = reverse_factory(index.base_index)
            return f"{inner},RFlat"
    except Exception:
        pass
    raise InvalidArgumentError(f"reverse_factory: unsupported index type {type(index).__name__}")


def get_supported_descriptions() -> Dict[str, str]:
    """
    Get dictionary of supported index descriptions and their explanations.
    
    Returns:
        Dictionary mapping description patterns to explanations
    """
    return {
        "Flat": "Brute-force exact search (IndexFlat)",
        "IVF{nlist}": "Inverted file with {nlist} centroids, flat quantizer", 
        "IVF{nlist},Flat": "Same as IVF{nlist}",
        "IVF{nlist},PQ{m}": "IVF with Product Quantization ({m} subquantizers)",
        "IVF{nlist},SQ{bits}": "IVF with Scalar Quantization ({bits} bits per component)",
        "PQ{m}": "Product Quantization with {m} subquantizers",
        "SQ{bits}": "Scalar Quantization with {bits} bits per component", 
        "HNSW{M}": "Hierarchical NSW graph with parameter M",
        "LSH": "Locality Sensitive Hashing",
        "IDMap,{base}": "ID mapping wrapper around {base} index",
        "IDMap2,{base}": "Two-way ID mapping wrapper around {base} index",
        "PCA{dim},{base}": "PCA preprocessing to {dim} dimensions + {base} index",
        "OPQ{dim},{base}": "Optimized PQ preprocessing + {base} index",
        # Additional patterns
        "SQfp16": "Scalar quantization with fp16 precision",
        "SQbf16": "Scalar quantization with bf16 precision",
    }


# Convenience function for binary indexes (future extension)
def binary_index_factory(
    d: int,
    description: str
):
    """
    Create binary index from string description.
    
    Note: This is a placeholder for future binary index factory support.
    """
    raise NotImplementedError("Binary index factory not yet implemented")
