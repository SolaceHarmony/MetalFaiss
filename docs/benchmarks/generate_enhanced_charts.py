#!/usr/bin/env python3
"""
Enhanced chart generator for MetalFaiss benchmarks
Creates professional-looking charts with improved styling and competitive comparisons
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from pathlib import Path
import seaborn as sns

# Set up the plotting style
plt.style.use('default')
sns.set_palette("husl")

def setup_plot_style():
    """Configure matplotlib for professional-looking plots"""
    plt.rcParams.update({
        'figure.figsize': (12, 8),
        'font.size': 12,
        'font.family': 'sans-serif',
        'axes.linewidth': 1.2,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linewidth': 0.8,
        'axes.axisbelow': True,
    })

def create_enhanced_bar_chart(csv_file, title, ylabel, output_file, competitive_data=None):
    """Create an enhanced bar chart from CSV data"""
    setup_plot_style()
    
    # Read data
    df = pd.read_csv(csv_file)
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Define colors for different categories
    colors = ['#2E8B57', '#4169E1', '#FF6347', '#32CD32', '#FF4500', '#9370DB']
    
    # Create bars
    bars = ax.bar(df['label'], df['value'] * 1000, color=colors[:len(df)], 
                  alpha=0.8, edgecolor='white', linewidth=2)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{height:.2f}ms', ha='center', va='bottom', fontweight='bold')
    
    # Add competitive comparison if provided
    if competitive_data:
        # Add a comparison section
        ax2 = ax.twinx()
        ax2.set_ylim(ax.get_ylim())
        ax2.set_ylabel('Industry Comparison', fontsize=12, color='gray')
        
        # Add text annotations for competitive data
        y_pos = max(df['value']) * 1000 * 0.7
        for i, (name, value, note) in enumerate(competitive_data):
            ax.text(0.02, 0.85 - i*0.08, f'{name}: {value}ms {note}', 
                   transform=ax.transAxes, fontsize=10, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7))
    
    # Styling
    ax.set_title(title, fontsize=18, fontweight='bold', pad=20)
    ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
    ax.set_xlabel('Implementation', fontsize=14, fontweight='bold')
    
    # Rotate x-axis labels if needed
    plt.xticks(rotation=15, ha='right')
    
    # Add subtle gradient background
    ax.set_facecolor('#fafafa')
    
    # Add performance callouts
    min_val = df['value'].min() * 1000
    max_val = df['value'].max() * 1000
    fastest_idx = df['value'].idxmin()
    
    # Add "Winner" badge
    if len(df) > 1:
        winner_bar = bars[fastest_idx]
        ax.annotate('FASTEST', xy=(winner_bar.get_x() + winner_bar.get_width()/2, 
                                     winner_bar.get_height()), 
                   xytext=(0, 20), textcoords='offset points',
                   ha='center', fontsize=12, fontweight='bold', color='darkgreen',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

def create_comparison_chart(csv_file, title, output_file):
    """Create a comparison chart with speedup indicators"""
    setup_plot_style()
    
    df = pd.read_csv(csv_file)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Left plot: Absolute times
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    bars1 = ax1.bar(df['label'], df['value'] * 1000, color=colors[:len(df)], alpha=0.8)
    
    ax1.set_title('Absolute Performance', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Time (milliseconds)', fontsize=12)
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}ms', ha='center', va='bottom', fontweight='bold')
    
    # Right plot: Relative speedup
    baseline = df['value'].max()  # Use slowest as baseline
    speedups = baseline / df['value']
    
    bars2 = ax2.bar(df['label'], speedups, color=colors[:len(df)], alpha=0.8)
    ax2.set_title('Relative Speedup', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Speedup Factor', fontsize=12)
    ax2.tick_params(axis='x', rotation=45)
    
    # Add speedup labels
    for bar, speedup in zip(bars2, speedups):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{speedup:.1f}x', ha='center', va='bottom', fontweight='bold')
    
    plt.suptitle(title, fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

def main():
    """Generate all enhanced charts"""
    base_dir = Path(__file__).parent
    
    # QR Projection chart with competitive data
    qr_competitive = [
        ('FAISS SIMD (CPU)', '~0.1-0.3', '(estimated)'),
        ('MLX Optimized', '0.38', '(our baseline)'),
    ]
    
    create_enhanced_bar_chart(
        base_dir / 'qr.csv',
        'MetalFAISS QR Projection Performance (c = Qᵀv)\nMLX + Metal Optimization',
        'Execution Time (milliseconds)',
        base_dir / 'qr_enhanced.png',
        qr_competitive
    )
    
    # IVF Search chart with competitive data
    ivf_competitive = [
        ('FAISS Classic (H100)', '0.75', '(100M vectors)'),
        ('FAISS cuVS (H100)', '0.39', '(2.7x speedup)'),
        ('FAISS CPU', '~5-50', '(typical range)'),
    ]
    
    create_enhanced_bar_chart(
        base_dir / 'ivf.csv',
        'MetalFAISS IVF Search Performance (d=64, N=32k, nprobe=8)\nApple Silicon vs Industry Leaders',
        'Query Time (milliseconds)',
        base_dir / 'ivf_enhanced.png',
        ivf_competitive
    )
    
    # Orthogonality chart
    create_enhanced_bar_chart(
        base_dir / 'orthogonality.csv',
        'MetalFAISS Orthogonality Operations (m=1024, n=256)\nModified Gram-Schmidt Performance',
        'Computation Time (milliseconds)',
        base_dir / 'orthogonality_enhanced.png'
    )
    
    # Create comparison charts
    create_comparison_chart(
        base_dir / 'ivf.csv',
        'MetalFAISS IVF Search: Absolute vs Relative Performance',
        base_dir / 'ivf_comparison.png'
    )
    
    print("Enhanced charts generated successfully!")
    print("Files created:")
    print("  - qr_enhanced.png")
    print("  - ivf_enhanced.png") 
    print("  - orthogonality_enhanced.png")
    print("  - ivf_comparison.png")

if __name__ == "__main__":
    main()
