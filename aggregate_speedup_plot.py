#!/usr/bin/env python3
"""
Aggregate speedup plot generator.
Reads all *_tpch_speedup.csv files and creates:
1. An aggregate plot of all speedup curves (lines 1-64)
2. A 1D scatter plot of all runtimes (line 65)
"""

import os
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import savgol_filter


def read_speedup_csv(filepath):
    """
    Read a speedup CSV file.
    Returns:
        speedup: numpy array of 64 speedup values (lines 1-64)
        runtime: float runtime value (line 65)
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # First 64 lines are speedup values
    speedup = np.array([float(line.strip()) for line in lines[:64]])
    
    # Line 65 (index 64) is runtime
    runtime = float(lines[64].strip()) if len(lines) > 64 and lines[64].strip() else None
    
    return speedup, runtime


def get_query_label(filepath):
    """Extract query number/label from filename."""
    filename = os.path.basename(filepath)
    # Extract the prefix before '_tpch_speedup.csv'
    label = filename.replace('_tpch_speedup.csv', '')
    return label


def main():
    # Directory containing the CSV files
    queries_dir = Path(__file__).parent / 'queries'
    
    # Find all speedup CSV files (excluding inactive folder)
    csv_pattern = str(queries_dir / '*_tpch_speedup.csv')
    csv_files = sorted(glob.glob(csv_pattern))
    
    if not csv_files:
        print(f"No speedup CSV files found in {queries_dir}")
        return
    
    print(f"Found {len(csv_files)} speedup CSV files")
    
    # Read all data
    data = {}
    runtimes = {}
    for filepath in csv_files:
        label = get_query_label(filepath)
        speedup, runtime = read_speedup_csv(filepath)
        data[label] = speedup
        if runtime is not None:
            runtimes[label] = runtime
        print(f"  {label}: speedup range [{speedup.min():.2f}, {speedup.max():.2f}], runtime={runtime:.4f}s")
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), 
                                    gridspec_kw={'height_ratios': [3, 1]})
    
    # Color map for different queries
    colors = plt.cm.tab20(np.linspace(0, 1, len(data)))
    
    # Plot 1: Aggregate speedup curves
    x = np.arange(1, 65)  # 1 to 64 (number of threads/cores)
    
    # Savitzky-Golay filter parameters
    # window_length must be odd and <= data length
    # polyorder must be < window_length
    smooth_window = 11  # Window size (odd number)
    smooth_poly = 3     # Polynomial order
    
    for i, (label, speedup) in enumerate(sorted(data.items())):
        # Apply Savitzky-Golay smoothing
        speedup_smooth = savgol_filter(speedup, smooth_window, smooth_poly)
        ax1.plot(x, speedup_smooth, label=label, color=colors[i], alpha=0.8, linewidth=1.5)
    
    # Add ideal linear speedup reference line
    ax1.plot(x, x, 'k--', label='Ideal (linear)', alpha=0.5, linewidth=2)
    
    ax1.set_xlabel('Number of Threads', fontsize=12)
    ax1.set_ylabel('Speedup', fontsize=12)
    ax1.set_title('TPC-H Query Speedup Curves (Aggregate)', fontsize=14)
    ax1.legend(loc='upper left', fontsize=8, ncol=3)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(1, 64)
    ax1.set_ylim(0, None)
    
    # Plot 2: 1D scatter plot of runtimes
    sorted_runtimes = sorted(runtimes.items())
    labels = [item[0] for item in sorted_runtimes]
    runtime_values = [item[1] for item in sorted_runtimes]
    
    # Create scatter plot with labels
    scatter = ax2.scatter(runtime_values, [0] * len(runtime_values), 
                          c=range(len(runtime_values)), cmap='tab20', 
                          s=100, alpha=0.8, edgecolors='black', linewidth=0.5)
    
    # Add labels for each point
    for i, (label, runtime) in enumerate(sorted_runtimes):
        ax2.annotate(label, (runtime, 0), textcoords="offset points", 
                     xytext=(0, 10 + (i % 3) * 8), ha='center', fontsize=7,
                     rotation=45)
    
    ax2.set_xlabel('Runtime (seconds)', fontsize=12)
    ax2.set_title('TPC-H Query Runtimes (1D Scatter)', fontsize=14)
    ax2.set_yticks([])  # Hide y-axis ticks for 1D scatter
    ax2.grid(True, axis='x', alpha=0.3)
    ax2.set_ylim(-0.5, 1.5)  # Give some vertical space for labels
    
    plt.tight_layout()
    
    # Save the figure
    output_path = queries_dir / 'aggregate_speedup_plot.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved aggregate plot to: {output_path}")
    
    # Also save a separate detailed runtime plot
    fig2, ax3 = plt.subplots(figsize=(14, 4))
    
    # Bar chart of runtimes for better readability
    bars = ax3.bar(labels, runtime_values, color=colors[:len(labels)], 
                   edgecolor='black', linewidth=0.5)
    ax3.set_xlabel('Query', fontsize=12)
    ax3.set_ylabel('Runtime (seconds)', fontsize=12)
    ax3.set_title('TPC-H Query Runtimes', fontsize=14)
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, val in zip(bars, runtime_values):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                 f'{val:.1f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    output_path2 = queries_dir / 'runtime_bar_chart.png'
    plt.savefig(output_path2, dpi=150, bbox_inches='tight')
    print(f"Saved runtime bar chart to: {output_path2}")


if __name__ == '__main__':
    main()
