from __future__ import annotations

"""
Quick comparator: MetalFaiss (MLX) vs FAISS CPU for exact Flat L2/IP.

Usage:
  PYTHONPATH=. METALFAISS_ALLOW_CPU=1 \
  python3 tools/compare_vs_faiss.py --d 128 --nb 10000 --nq 200 --k 10 --metric l2 --seed 123

Requires `faiss-cpu` to be installed in the Python env (pip install faiss-cpu).
"""

import argparse
import numpy as np
import mlx.core as mx

from metalfaiss.index.flat_index import FlatIndex
from metalfaiss.types.metric_type import MetricType


def make_dataset(d: int, nb: int, nq: int, seed: int, metric: str):
    rng = np.random.default_rng(seed)
    xb = rng.standard_normal((nb, d), dtype=np.float32)
    xq = rng.standard_normal((nq, d), dtype=np.float32)
    if metric == "ip":
        xb /= np.linalg.norm(xb, axis=1, keepdims=True) + 1e-12
        xq /= np.linalg.norm(xq, axis=1, keepdims=True) + 1e-12
    return xb, xq


def recall_at_k(Ia: np.ndarray, Ib: np.ndarray, k: int) -> float:
    nq = Ia.shape[0]
    tot = 0.0
    for i in range(nq):
        tot += len(set(Ia[i, :k]).intersection(set(Ib[i, :k]))) / float(k)
    return float(tot / nq)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--nb", type=int, default=10000)
    ap.add_argument("--nq", type=int, default=200)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--metric", choices=["l2", "ip"], default="l2")
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    xb, xq = make_dataset(args.d, args.nb, args.nq, args.seed, args.metric)

    # MetalFaiss
    mt = MetricType.L2 if args.metric == "l2" else MetricType.INNER_PRODUCT
    mf = FlatIndex(d=args.d, metric_type=mt)
    mf.train(xb.tolist())
    mf.add(xb.tolist())
    Dm, Im = mf.search(xq.tolist(), args.k)
    Im = np.array(Im, dtype=np.int64)
    Dm = np.array(Dm, dtype=np.float32)

    # FAISS CPU
    try:
        import faiss
    except Exception as e:
        raise SystemExit("faiss-cpu not installed; pip install faiss-cpu") from e
    if args.metric == "l2":
        idx = faiss.IndexFlatL2(args.d)
    else:
        idx = faiss.IndexFlatIP(args.d)
    idx.add(xb)
    Df, If = idx.search(xq, args.k)

    rec = recall_at_k(If, Im, args.k)
    print(f"recall@{args.k}: {rec:.4f}")
    md = np.mean(np.abs(Df - Dm))
    Mx = np.max(np.abs(Df - Dm))
    print(f"mean |Δdist|: {md:.6g} | max |Δdist|: {Mx:.6g}")


if __name__ == "__main__":
    main()

