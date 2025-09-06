# MetalFAISS Benchmarks

This directory contains comprehensive benchmark results and competitive analysis for MetalFAISS performance.

## Directory Contents

### Results & Analysis
- **[Results.md](Results.md)** - Main benchmark results with enhanced visualizations
- **[Competitive-Analysis.md](Competitive-Analysis.md)** - Industry positioning and competitive landscape

### Visualizations
- **performance_dashboard.png** - Comprehensive performance overview
- **ivf_enhanced.png** - Enhanced IVF search performance chart
- **qr_enhanced.png** - Enhanced QR projection performance chart
- **orthogonality_enhanced.png** - Enhanced orthogonality operations chart
- **ivf_comparison.png** - Detailed IVF performance analysis

### Generation Scripts
- **generate_enhanced_charts.py** - Creates professional benchmark charts
- **generate_dashboard.py** - Generates comprehensive performance dashboard

### Raw Data
- **ivf.csv** - IVF search benchmark raw data
- **qr.csv** - QR projection benchmark raw data
- **orthogonality.csv** - Orthogonality operations raw data

## Quick Start

### View Results
```bash
# Open main results
open Results.md

# View competitive analysis  
open Competitive-Analysis.md
```

### Regenerate Charts
```bash
# Enhanced charts with competitive comparisons
python generate_enhanced_charts.py

# Performance dashboard
python generate_dashboard.py
```

### Run Benchmarks
```bash
# From project root
PYTHONPATH=python python python/metalfaiss/benchmarks/run_benchmarks.py
```

## Key Findings

### Performance Highlights
- **20x speedup** in specialized batched IVF scenarios
- **Competitive** QR projection performance (0.38ms)
- **Superior cost-performance** on consumer hardware

### Competitive Position
- **Apple Silicon Optimized**: Best-in-class performance for Metal acceleration
- **Pure Python**: Zero compilation complexity vs traditional FAISS
- **Developer Friendly**: Rapid iteration and customization capabilities

## Benchmark Methodology

### Test Configuration
- **Hardware**: Apple Silicon M-series with Metal acceleration
- **Framework**: MLX for GPU compute operations  
- **Metrics**: Median timing over 5 runs (1 warmup)
- **Precision**: float32 throughout

### Competitive Data Sources
- [Meta Engineering: FAISS cuVS Performance](https://engineering.fb.com/2025/05/08/data-infrastructure/accelerating-gpu-indexes-in-faiss-with-nvidia-cuvs/)
- [ANN-Benchmarks](https://ann-benchmarks.com/)
- Internal MetalFAISS benchmarks

## Interpretation Guide

### Chart Types
- **Bar Charts**: Direct performance comparisons
- **Dashboard**: Multi-dimensional analysis (performance, cost, complexity)
- **Comparison Charts**: Absolute vs relative performance views

### Performance Context
- **Specialized Workloads**: MetalFAISS excels in batched scenarios
- **General Workloads**: Competitive with consumer hardware advantages
- **Development**: Pure Python enables rapid experimentation

## Customization

### Modify Benchmarks
1. Edit CSV files for custom data points
2. Run generation scripts to update charts
3. Update Results.md with new insights

### Add New Comparisons
1. Research competitive data from literature
2. Add entries to `competitive_data` in generation scripts
3. Update analysis documents with context

## Related Documentation

- **[MLX Optimization Guide](../mlx/Comprehensive-MLX-Metal-Guide.md)** - Deep dive into Metal kernel optimization
- **[Implementation Analysis](../../MetalFaiss-Analysis.md)** - Technical implementation details
- **[Main README](../../README.md)** - Project overview and quick start

---

*Last Updated: September 2025 | Benchmarks performed on Apple Silicon with MLX framework*
