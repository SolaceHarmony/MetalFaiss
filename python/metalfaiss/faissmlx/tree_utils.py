"""
tree_utils.py — Thin wrappers around mlx.utils tree utilities

Why:
- Work uniformly over nested structures (dict/list/tuples of mx.array)
- Pack small param dicts into stable MLX buffers for kernels
- Force evaluation across nested results without host scalar pulls

Functions:
- flatten_tree(tree) -> (leaves, spec)
- unflatten_tree(spec, leaves)
- map_tree(fn, tree)
- reduce_tree(fn, tree, init)
- tree_eval(tree)
- tree_cast(tree, dtype)
- tree_to_device(tree)  (future-proof; currently default device)
- pack_params_uint32(params: dict) -> (buf, keys)
- pack_params_f32(params: dict) -> (buf, keys)
"""
from __future__ import annotations
from typing import Any, Callable, Iterable, List, Tuple
import mlx.core as mx
import mlx.utils as mxu


def flatten_tree(tree: Any) -> Tuple[List[Any], Any]:
    leaves, spec = mxu.tree_flatten(tree)
    return list(leaves), spec


def unflatten_tree(spec: Any, leaves: Iterable[Any]) -> Any:
    return mxu.tree_unflatten(spec, list(leaves))


def map_tree(fn: Callable[[Any], Any], tree: Any) -> Any:
    return mxu.tree_map(fn, tree)


def reduce_tree(fn: Callable[[Any, Any], Any], tree: Any, init: Any) -> Any:
    return mxu.tree_reduce(fn, tree, init)


def tree_eval(tree: Any) -> None:
    # Collect MLX arrays from a nested structure and force evaluation
    arrs: List[mx.array] = []
    def _collect(x):
        if isinstance(x, mx.array):
            arrs.append(x)
        return x
    mxu.tree_map(_collect, tree)
    if arrs:
        mx.eval(*arrs)


def tree_cast(tree: Any, dtype: mx.Dtype) -> Any:
    def _cast(x):
        if isinstance(x, mx.array):
            return x.astype(dtype)
        return x
    return mxu.tree_map(_cast, tree)


def pack_params_uint32(params: dict) -> Tuple[mx.array, List[str]]:
    """Pack int params into a stable uint32 buffer sorted by key.

    Returns the buffer and the ordered list of keys for reproducible unpacking.
    """
    keys = sorted(params.keys())
    vals = [int(params[k]) for k in keys]
    buf = mx.array(vals, dtype=mx.uint32)
    return buf, keys


def pack_params_f32(params: dict) -> Tuple[mx.array, List[str]]:
    keys = sorted(params.keys())
    vals = [float(params[k]) for k in keys]
    buf = mx.array(vals, dtype=mx.float32)
    return buf, keys

