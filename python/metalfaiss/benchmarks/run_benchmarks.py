"""
Run core MetalFaiss benchmarks and write CSV + PNG charts in docs/benchmarks/.

Benchmarks
- QR projection: MLX dot vs kernel simple vs kernel simd
- IVF search: Baseline MLX (emulated) vs fused concat vs batched (same X)
- Orthogonality: orthonormal_columns vs orthogonalize_blocked vs orthogonal initializer

Outputs
- docs/benchmarks/qr.csv, ivf.csv, orthogonality.csv
- docs/benchmarks/qr.png, ivf.png, orthogonality.png (if matplotlib available)
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

import mlx.core as mx


# Utilities
def median_time(fn, warmup: int = 1, repeats: int = 5):
    for _ in range(warmup):
        out = fn()
        if isinstance(out, tuple):
            mx.eval(*out)
        else:
            mx.eval(out)
    ts: List[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        if isinstance(out, tuple):
            mx.eval(*out)
        else:
            mx.eval(out)
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[len(ts)//2]


def ensure_dir(path: str):
    import os
    os.makedirs(path, exist_ok=True)


# QR projection benchmark
def bench_qr(m: int = 4096, k: int = 128) -> Tuple[List[str], List[float]]:
    from metalfaiss.faissmlx.kernels.qr_kernels import project_coeffs

    mx.random.seed(0)
    Q = mx.random.normal(shape=(m, k)).astype(mx.float32)
    v = mx.random.normal(shape=(m,)).astype(mx.float32)

    labels: List[str] = ["MLX dot", "Kernel simple", "Kernel simd"]

    # MLX
    t_mlx = median_time(lambda: mx.matmul(mx.transpose(Q), v))

    # Simple kernel
    os.environ["METALFAISS_QR_DOT"] = "simple"
    t_simple = median_time(lambda: project_coeffs(Q, v))

    # SIMD kernel
    os.environ["METALFAISS_QR_DOT"] = "simd"
    t_simd = median_time(lambda: project_coeffs(Q, v))

    os.environ.pop("METALFAISS_QR_DOT", None)
    return labels, [t_mlx, t_simple, t_simd]


# IVF benchmark (emulated baseline vs fused kernels)
def _simple_quantizer(xb: mx.array, nlist: int):
    # very small k-means (few iters)
    N, d = int(xb.shape[0]), int(xb.shape[1])
    k = nlist
    idx = mx.random.randint(0, N, (k,), dtype=mx.int32)
    C = xb[idx, :]
    for _ in range(3):
        d2 = mx.sum((xb[:, None, :] - C[None, :, :])**2, axis=2)
        I = mx.argmin(d2, axis=1)
        for j in range(k):
            mask = (I == j)
            cnt = int(mx.sum(mask).item())
            if cnt > 0:
                C[j] = mx.sum(xb * mask[:, None], axis=0) / cnt
    return C


def bench_ivf(d: int = 64, N: int = 32768, nlist: int = 128, QN: int = 16, nprobe: int = 8, k: int = 10) -> Tuple[List[str], List[float]]:
    from metalfaiss.faissmlx.kernels.ivf_kernels import ivf_list_topk_l2, ivf_list_topk_l2_batch
    mx.random.seed(0)
    xb = mx.random.normal(shape=(N, d)).astype(mx.float32)
    C = _simple_quantizer(xb, nlist)
    # Build inverted lists
    d2 = mx.sum((xb[:, None, :] - C[None, :, :])**2, axis=2)
    I = mx.argmin(d2, axis=1)
    invlists = [[] for _ in range(nlist)]
    for i in range(N):
        invlists[int(I[i].item())].append((i, xb[i]))
    xq = mx.random.normal(shape=(QN, d)).astype(mx.float32)

    def probe_lists(qv):
        d2q = mx.sum((qv[None, :] - C)**2, axis=1)
        idx = mx.argsort(d2q)[:nprobe]
        return [int(t.item()) for t in idx]

    # Baseline (MLX argsort):
    def run_baseline():
        for qi in range(QN):
            qv = xq[qi]
            labs = probe_lists(qv)
            vecs = []
            for lid in labs:
                vecs.extend([vec for (_, vec) in invlists[lid]])
            if not vecs:
                continue
            X = mx.stack(vecs)
            dists = mx.sum((X - qv)**2, axis=1)
            _ = mx.argsort(dists)[:k]
    t_mlx = median_time(run_baseline)

    # Fused concat
    def run_fused():
        for qi in range(QN):
            qv = xq[qi]
            labs = probe_lists(qv)
            vecs, ids = [], []
            for lid in labs:
                for vid, vec in invlists[lid]:
                    vecs.append(vec); ids.append(vid)
            if not vecs:
                continue
            X = mx.stack(vecs); Iarr = mx.array(ids, dtype=mx.int32)
            ivf_list_topk_l2(qv, X, Iarr, k)
    t_fused = median_time(run_fused)

    # Batched (same X) synthetic
    def run_batched_sameX():
        qv = xq[0]
        labs = probe_lists(qv)
        vecs, ids = [], []
        for lid in labs:
            for vid, vec in invlists[lid]:
                vecs.append(vec); ids.append(vid)
        if not vecs:
            return None
        X = mx.stack(vecs); Iarr = mx.array(ids, dtype=mx.int32)
        ivf_list_topk_l2_batch(xq, X, Iarr, k)
    t_batch = median_time(run_batched_sameX)

    labels = ["Baseline MLX", "Fused concat", "Batched sameX"]
    return labels, [t_mlx, t_fused, t_batch]


# Orthogonality bench
def bench_orthogonality(m: int = 1024, n: int = 256) -> Tuple[List[str], List[float]]:
    from metalfaiss.faissmlx.orthogonality import orthonormal_columns, orthogonalize_blocked, orthogonal
    mx.random.seed(0)
    X = mx.random.normal(shape=(m, n)).astype(mx.float32)
    labels = ["orthonormal_columns", "orthogonalize_blocked", "orthogonal_init"]
    t_cols = median_time(lambda: orthonormal_columns(X))
    t_blk = median_time(lambda: orthogonalize_blocked(X, B=32))
    t_init = median_time(lambda: orthogonal((m, n)))
    return labels, [t_cols, t_blk, t_init]


def write_csv(path: str, labels: List[str], values: List[float]):
    with open(path, "w") as f:
        f.write("label,value\n")
        for l, v in zip(labels, values):
            f.write(f"{l},{v}\n")


def save_bar_png(path: str, title: str, labels: List[str], values: List[float]):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    plt.figure(figsize=(6,4))
    idx = range(len(labels))
    plt.bar(idx, values, color=["#4e79a7", "#f28e2b", "#e15759"][:len(labels)])
    plt.xticks(list(idx), labels, rotation=15, ha="right")
    plt.ylabel("seconds (median)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def main():
    out_dir = "docs/benchmarks"
    ensure_dir(out_dir)

    # QR
    qr_labels, qr_values = bench_qr()
    write_csv(f"{out_dir}/qr.csv", qr_labels, qr_values)
    save_bar_png(f"{out_dir}/qr.png", "QR Projection (c = Q^T v)", qr_labels, qr_values)

    # IVF
    ivf_labels, ivf_values = bench_ivf()
    write_csv(f"{out_dir}/ivf.csv", ivf_labels, ivf_values)
    save_bar_png(f"{out_dir}/ivf.png", "IVF Search (d=64,N=32k,nprobe=8,Q=16,k=10)", ivf_labels, ivf_values)

    # Orthogonality
    ortho_labels, ortho_values = bench_orthogonality()
    write_csv(f"{out_dir}/orthogonality.csv", ortho_labels, ortho_values)
    save_bar_png(f"{out_dir}/orthogonality.png", "Orthogonality (m=1024,n=256)", ortho_labels, ortho_values)

    # Emit provenance metadata
    try:
        import platform, json, subprocess
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    except Exception:
        platform, commit = None, ""
    # Try to capture MLX + Metal device details
    mlx_version = getattr(mx, "__version__", "")
    device_name = ""
    try:
        import mlx.core.metal as metal
        info = metal.device_info()
        device_name = str(info.get("device_name", ""))
    except Exception:
        pass
    meta: Dict[str, Any] = {
        "timestamp": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
        "git_commit": commit,
        "python": {
            "version": __import__("sys").version.split(" ")[0],
            "platform": __import__("platform").platform(),
            "machine": __import__("platform").machine(),
        },
        "mlx": {"version": mlx_version, "device": device_name},
        "benches": {
            "qr": dict(labels=qr_labels, values_seconds=qr_values, shape={"m": 4096, "k": 128}),
            "ivf": dict(labels=ivf_labels, values_seconds=ivf_values, params={"d": 64, "N": 32768, "nlist": 128, "Q": 16, "nprobe": 8, "k": 10}),
            "orthogonality": dict(labels=ortho_labels, values_seconds=ortho_values, shape={"m": 1024, "n": 256}),
        },
    }
    try:
        with open(f"{out_dir}/bench_meta.json", "w") as f:
            json.dump(meta, f, indent=2)
    except Exception:
        pass

    print(f"Wrote CSV and PNGs to {out_dir}")


if __name__ == "__main__":
    main()
