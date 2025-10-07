"""Backwards-compatible FlatIndex shim exporting the core MLX implementation."""

from .index.flat_index import FlatIndex  # re-export

__all__ = ["FlatIndex"]
