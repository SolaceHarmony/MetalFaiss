#!/usr/bin/env python3
"""
Validate benchmark docs against CSVs and suggest updated tables.

Reads qr.csv, ivf.csv, orthogonality.csv and prints a markdown table snippet
you can paste into Results.md. Also checks which label is fastest.
"""
from pathlib import Path
import pandas as pd


def ms(x: float) -> float:
    return x * 1000.0


def load_csv(name: str, base: Path) -> pd.DataFrame:
    df = pd.read_csv(base / f"{name}.csv")
    return df


def table_qr(df: pd.DataFrame) -> str:
    # Sort by value ascending (fastest first)
    d = df.copy()
    d["ms"] = d["value"].apply(ms)
    d = d.sort_values("value").reset_index(drop=True)
    fastest = d.loc[0, "label"]
    baseline = d.loc[0, "value"]
    lines = ["| Method | Time (ms) | Speedup vs fastest | Status |",
             "|--------|-----------|--------------------|---------|"]
    for i, r in d.iterrows():
        speed = baseline / r["value"]
        status = "Fastest" if i == 0 else ("Close second" if i == 1 else "Reference")
        label = r["label"].replace("MLX dot", "MetalFAISS MLX dot").replace("Kernel", "MetalFAISS Kernel")
        lines.append(f"| {('**' + label + '**') if i==0 else label} | {r['ms']:.3f} | {speed:.2f}x | {status} |")
    return "\n".join(lines), fastest


def table_ivf(df: pd.DataFrame) -> str:
    d = df.copy(); d["ms"] = d["value"].apply(ms)
    lines = ["| Method | Time (ms) | Status |",
             "|--------|-----------|---------|"]
    order = ["Batched sameX", "Fused concat", "Baseline MLX"]
    d = d.set_index("label")
    for label in order:
        r = d.loc[label]
        pretty = label.replace("Batched sameX", "MetalFAISS Batched sameX").replace("Fused concat", "MetalFAISS Fused concat").replace("Baseline MLX", "MetalFAISS Baseline MLX")
        status = "Specialized" if "Batched" in label else ("Production" if "Fused" in label else "Reference")
        lines.append(f"| {pretty} | {r['ms']:.2f} | {status} |")
    return "\n".join(lines)


def table_orth(df: pd.DataFrame) -> str:
    d = df.copy(); d["ms"] = d["value"].apply(ms)
    d = d.set_index("label")
    rows = [
        ("orthogonalize_blocked", "MetalFAISS Orthogonalize blocked", "Production MGS"),
        ("orthonormal_columns", "MetalFAISS Orthonormal columns", "Standard QR"),
        ("orthogonal_init", "MetalFAISS Orthogonal init", "Initialization"),
    ]
    lines = ["| Method | Time (ms) | Use Case | Status |",
             "|--------|-----------|----------|---------|"]
    for key, pretty, use in rows:
        r = d.loc[key]
        status = "Recommended" if key == "orthogonalize_blocked" else ("Stable" if key == "orthonormal_columns" else "Heavy")
        lines.append(f"| {pretty} | {r['ms']:.2f} | {use} | {status} |")
    return "\n".join(lines)


def main():
    base = Path(__file__).parent
    qr = load_csv("qr", base)
    ivf = load_csv("ivf", base)
    ortho = load_csv("orthogonality", base)

    print("\n# Suggested QR table")
    t, fastest = table_qr(qr)
    print(t)
    print(f"\nFastest method per CSV: {fastest}")

    print("\n# Suggested IVF table")
    print(table_ivf(ivf))

    print("\n# Suggested Orthogonality table")
    print(table_orth(ortho))

    meta = base / "bench_meta.json"
    if meta.exists():
        print("\nProvenance: see docs/benchmarks/bench_meta.json for device + commit info.")

if __name__ == "__main__":
    main()

