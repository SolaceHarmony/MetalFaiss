#!/usr/bin/env python3
"""
Performance Dashboard Generator for MetalFaiss
Creates a summary dashboard comparing MetalFaiss with industry standards
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path

def create_performance_dashboard():
    """Create a comprehensive performance dashboard"""
    
    # Set up the style
    plt.style.use('default')
    sns.set_palette("viridis")
    
    fig = plt.figure(figsize=(16, 12))
    
    # Create a grid layout
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.8], hspace=0.3, wspace=0.3)
    
    # 1. Absolute Performance Comparison
    ax1 = fig.add_subplot(gs[0, 0])
    libraries = ['FAISS cuVS\n(H100)', 'FAISS Classic\n(H100)', 'MetalFAISS\nBatched', 'MetalFAISS\nStandard']
    latencies = [0.39, 0.75, 1.52, 29.86]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    bars = ax1.bar(libraries, latencies, color=colors, alpha=0.8)
    ax1.set_ylabel('Latency (milliseconds)', fontweight='bold')
    ax1.set_title('Absolute Performance Comparison\nIVF Search (k=10)', fontweight='bold', fontsize=14)
    ax1.set_yscale('log')
    
    # Add value labels
    for bar, latency in zip(bars, latencies):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{latency:.2f}ms', ha='center', va='bottom', fontweight='bold')
    
    # 2. Cost-Performance Analysis
    ax2 = fig.add_subplot(gs[0, 1])
    hardware_costs = [30000, 30000, 3000, 3000]  # Approximate costs
    performance_scores = [1/0.39, 1/0.75, 1/1.52, 1/29.86]  # Inverse of latency
    cost_performance = [p/c * 1000000 for p, c in zip(performance_scores, hardware_costs)]
    
    bars2 = ax2.bar(libraries, cost_performance, color=colors, alpha=0.8)
    ax2.set_ylabel('Performance / Cost\n(QPS per $1000)', fontweight='bold')
    ax2.set_title('Cost-Performance Efficiency', fontweight='bold', fontsize=14)
    
    for bar, score in zip(bars2, cost_performance):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{score:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # 3. Deployment Complexity Radar
    ax3 = fig.add_subplot(gs[1, :], projection='polar')
    
    categories = ['Installation\nSimplicity', 'Hardware\nAccessibility', 
                 'Development\nVelocity', 'Customization\nEase', 'Raw\nPerformance']
    N = len(categories)
    
    # Scores out of 5
    faiss_scores = [2, 1, 2, 2, 5]  # FAISS
    metalfaiss_scores = [5, 4, 5, 5, 3]  # MetalFAISS
    
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Complete the circle
    
    faiss_scores += faiss_scores[:1]
    metalfaiss_scores += metalfaiss_scores[:1]
    
    ax3.plot(angles, faiss_scores, 'o-', linewidth=2, label='FAISS Classic', color='#ff7f0e')
    ax3.fill(angles, faiss_scores, alpha=0.25, color='#ff7f0e')
    ax3.plot(angles, metalfaiss_scores, 'o-', linewidth=2, label='MetalFAISS', color='#2ca02c')
    ax3.fill(angles, metalfaiss_scores, alpha=0.25, color='#2ca02c')
    
    ax3.set_xticks(angles[:-1])
    ax3.set_xticklabels(categories, fontsize=10)
    ax3.set_ylim(0, 5)
    ax3.set_title('Deployment & Development Comparison', fontweight='bold', 
                 fontsize=14, pad=30)
    ax3.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    ax3.grid(True)
    
    # 4. Summary Statistics Table
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis('off')
    
    table_data = [
        ['Metric', 'FAISS cuVS', 'FAISS Classic', 'MetalFAISS Batched', 'MetalFAISS Standard'],
        ['Latency (ms)', '0.39', '0.75', '1.52', '29.86'],
        ['Hardware Cost', '$30,000+', '$30,000+', '$2,000-4,000', '$2,000-4,000'],
        ['Installation', 'Complex', 'Complex', 'pip install', 'pip install'],
        ['Platform', 'CUDA/Linux', 'CUDA/Linux', 'Apple Silicon', 'Apple Silicon'],
        ['Language', 'C++/Python', 'C++/Python', 'Pure Python', 'Pure Python'],
        ['Best For', 'Max Performance', 'Data Centers', 'Development', 'Prototyping']
    ]
    
    table = ax4.table(cellText=table_data[1:], colLabels=table_data[0],
                     cellLoc='center', loc='center',
                     colWidths=[0.2, 0.2, 0.2, 0.2, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style the table
    for i in range(len(table_data)):
        for j in range(len(table_data[0])):
            cell = table[(i, j)]
            if i == 0:  # Header row
                cell.set_facecolor('#4CAF50')
                cell.set_text_props(weight='bold', color='white')
            elif j == 0:  # First column
                cell.set_facecolor('#E8F5E8')
                cell.set_text_props(weight='bold')
            else:
                cell.set_facecolor('#F9F9F9')
    
    plt.suptitle('MetalFAISS Performance Dashboard', fontsize=20, fontweight='bold', y=0.95)
    
    # Add footer
    fig.text(0.5, 0.02, 'Sources: Meta Engineering Blog (2025), Internal Benchmarks | Apple Silicon vs NVIDIA H100', 
             ha='center', fontsize=10, style='italic')
    
    plt.tight_layout()
    return fig

def main():
    """Generate the performance dashboard"""
    dashboard = create_performance_dashboard()
    
    # Save the dashboard
    output_path = Path(__file__).parent / 'performance_dashboard.png'
    dashboard.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print("Performance dashboard generated!")
    print(f"File saved: {output_path}")
    print("\nDashboard includes:")
    print("  - Absolute performance comparison")
    print("  - Cost-performance analysis")
    print("  - Deployment complexity radar chart")
    print("  - Summary statistics table")

if __name__ == "__main__":
    main()
