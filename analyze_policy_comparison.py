#!/usr/bin/env python3
"""
Policy Comparison Analysis Script
Compares greedy policy vs default policy runs in TPC-H benchmark results.

Greedy runs have prefix N, default runs have prefix N+3.
For example: 001 (greedy) pairs with 004 (default).
"""

import os
import re
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import argparse
from collections import defaultdict


@dataclass
class StatsData:
    """Parsed statistics data from a single stats file."""
    filename: str
    prefix: int
    policy: str  # 'greedy' or 'default'
    lambda1: float
    lambda2: float
    length: int
    seed: int
    total_queries: int
    overall: Dict[str, float]  # Count, Average, Min, Max, P50, P90, P95, P99
    per_query: Dict[str, Dict[str, float]]  # Query name -> stats


def parse_stats_file(filepath: str) -> Optional[StatsData]:
    """Parse a stats.txt file and extract all metrics."""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None
    
    filename = os.path.basename(filepath)
    
    # Extract prefix number
    prefix_match = re.match(r'^(\d+)_', filename)
    if not prefix_match:
        return None
    prefix = int(prefix_match.group(1))
    
    # Determine policy type
    if 'greedy' in filename:
        policy = 'greedy'
    elif 'default' in filename:
        policy = 'default'
    else:
        return None
    
    # Extract lambda values
    lambda_match = re.search(r'lam([\d.]+)_([\d.]+)', filename)
    if lambda_match:
        lambda1 = float(lambda_match.group(1))
        lambda2 = float(lambda_match.group(2))
    else:
        lambda1 = lambda2 = 0.0
    
    # Extract length and seed
    len_match = re.search(r'len(\d+)', filename)
    length = int(len_match.group(1)) if len_match else 0
    
    seed_match = re.search(r's(\d+)', filename)
    seed = int(seed_match.group(1)) if seed_match else 0
    
    # Parse total queries
    total_match = re.search(r'Total queries:\s*(\d+)', content)
    total_queries = int(total_match.group(1)) if total_match else 0
    
    # Parse overall statistics
    overall = {}
    overall_section = re.search(r'OVERALL STATISTICS.*?PER-QUERY', content, re.DOTALL)
    if overall_section:
        section_text = overall_section.group(0)
        for metric in ['Count', 'Average', 'Min', 'Max', 'P50', 'P90', 'P95', 'P99']:
            match = re.search(rf'{metric}:\s*([\d.]+)', section_text)
            if match:
                overall[metric] = float(match.group(1))
    
    # Parse per-query statistics
    per_query = {}
    query_pattern = re.compile(
        r'((?:p|np)_(?:greedy|default)_\d+):\s*\n'
        r'\s*Count:\s*([\d.]+)\s*\n'
        r'\s*Average:\s*([\d.]+)\s*ms\s*\n'
        r'\s*Min:\s*([\d.]+)\s*ms\s*\n'
        r'\s*Max:\s*([\d.]+)\s*ms\s*\n'
        r'\s*P50:\s*([\d.]+)\s*ms\s*\n'
        r'\s*P90:\s*([\d.]+)\s*ms\s*\n'
        r'\s*P95:\s*([\d.]+)\s*ms\s*\n'
        r'\s*P99:\s*([\d.]+)\s*ms',
        re.MULTILINE
    )
    
    for match in query_pattern.finditer(content):
        query_name = match.group(1)
        per_query[query_name] = {
            'Count': float(match.group(2)),
            'Average': float(match.group(3)),
            'Min': float(match.group(4)),
            'Max': float(match.group(5)),
            'P50': float(match.group(6)),
            'P90': float(match.group(7)),
            'P95': float(match.group(8)),
            'P99': float(match.group(9)),
        }
    
    return StatsData(
        filename=filename,
        prefix=prefix,
        policy=policy,
        lambda1=lambda1,
        lambda2=lambda2,
        length=length,
        seed=seed,
        total_queries=total_queries,
        overall=overall,
        per_query=per_query
    )


def find_pairs(stats_list: List[StatsData]) -> List[Tuple[StatsData, StatsData]]:
    """Find greedy-default pairs based on prefix difference of 3."""
    greedy_runs = {s.prefix: s for s in stats_list if s.policy == 'greedy'}
    default_runs = {s.prefix: s for s in stats_list if s.policy == 'default'}
    
    pairs = []
    for greedy_prefix, greedy_data in greedy_runs.items():
        default_prefix = greedy_prefix + 3
        if default_prefix in default_runs:
            default_data = default_runs[default_prefix]
            # Verify they have matching parameters (same lambda, length, seed)
            if (abs(greedy_data.lambda1 - default_data.lambda1) < 0.0001 and
                abs(greedy_data.lambda2 - default_data.lambda2) < 0.0001 and
                greedy_data.length == default_data.length and
                greedy_data.seed == default_data.seed):
                pairs.append((greedy_data, default_data))
    
    return sorted(pairs, key=lambda x: x[0].prefix)


def compute_comparison_metrics(greedy: StatsData, default: StatsData) -> Dict:
    """Compute comparison metrics between greedy and default runs."""
    metrics = {
        'greedy_prefix': greedy.prefix,
        'default_prefix': default.prefix,
        'lambda1': greedy.lambda1,
        'lambda2': greedy.lambda2,
        'total_lambda': greedy.lambda1 + greedy.lambda2,
        'lambda_ratio': greedy.lambda1 / greedy.lambda2 if greedy.lambda2 > 0 else float('inf'),
        'length': greedy.length,
        'seed': greedy.seed,
    }
    
    # Overall comparison
    for metric in ['Average', 'Min', 'Max', 'P50', 'P90', 'P95', 'P99']:
        greedy_val = greedy.overall.get(metric, 0)
        default_val = default.overall.get(metric, 0)
        metrics[f'greedy_{metric}'] = greedy_val
        metrics[f'default_{metric}'] = default_val
        metrics[f'diff_{metric}'] = greedy_val - default_val
        if default_val != 0:
            metrics[f'pct_diff_{metric}'] = ((greedy_val - default_val) / default_val) * 100
        else:
            metrics[f'pct_diff_{metric}'] = 0
    
    # Per-query type comparison (priority vs non-priority)
    greedy_p_avg = 0
    greedy_np_avg = 0
    default_p_avg = 0
    default_np_avg = 0
    
    for qname, qstats in greedy.per_query.items():
        if qname.startswith('p_'):
            greedy_p_avg = qstats.get('Average', 0)
        elif qname.startswith('np_'):
            greedy_np_avg = qstats.get('Average', 0)
    
    for qname, qstats in default.per_query.items():
        if qname.startswith('p_'):
            default_p_avg = qstats.get('Average', 0)
        elif qname.startswith('np_'):
            default_np_avg = qstats.get('Average', 0)
    
    metrics['greedy_priority_avg'] = greedy_p_avg
    metrics['greedy_nonpriority_avg'] = greedy_np_avg
    metrics['default_priority_avg'] = default_p_avg
    metrics['default_nonpriority_avg'] = default_np_avg
    
    metrics['priority_diff'] = greedy_p_avg - default_p_avg
    metrics['nonpriority_diff'] = greedy_np_avg - default_np_avg
    
    if default_p_avg != 0:
        metrics['priority_pct_diff'] = ((greedy_p_avg - default_p_avg) / default_p_avg) * 100
    else:
        metrics['priority_pct_diff'] = 0
        
    if default_np_avg != 0:
        metrics['nonpriority_pct_diff'] = ((greedy_np_avg - default_np_avg) / default_np_avg) * 100
    else:
        metrics['nonpriority_pct_diff'] = 0
    
    return metrics


def generate_summary_report(df: pd.DataFrame, output_dir: str):
    """Generate a text summary report."""
    report_path = os.path.join(output_dir, 'policy_comparison_summary.txt')
    
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("POLICY COMPARISON SUMMARY: GREEDY vs DEFAULT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Total pairs analyzed: {len(df)}\n\n")
        
        # Overall statistics
        f.write("-" * 60 + "\n")
        f.write("OVERALL LATENCY COMPARISON (ms)\n")
        f.write("-" * 60 + "\n\n")
        
        for metric in ['Average', 'P50', 'P90', 'P95', 'P99']:
            greedy_mean = df[f'greedy_{metric}'].mean()
            default_mean = df[f'default_{metric}'].mean()
            diff_mean = df[f'diff_{metric}'].mean()
            pct_diff_mean = df[f'pct_diff_{metric}'].mean()
            
            f.write(f"{metric}:\n")
            f.write(f"  Greedy Avg:  {greedy_mean:12.2f} ms\n")
            f.write(f"  Default Avg: {default_mean:12.2f} ms\n")
            f.write(f"  Diff:        {diff_mean:+12.2f} ms ({pct_diff_mean:+.2f}%)\n")
            f.write(f"  {'Greedy BETTER' if diff_mean < 0 else 'Default BETTER'}\n\n")
        
        # Priority vs Non-Priority breakdown
        f.write("-" * 60 + "\n")
        f.write("PRIORITY vs NON-PRIORITY QUERY BREAKDOWN\n")
        f.write("-" * 60 + "\n\n")
        
        f.write("Priority Queries:\n")
        f.write(f"  Greedy Avg:  {df['greedy_priority_avg'].mean():12.2f} ms\n")
        f.write(f"  Default Avg: {df['default_priority_avg'].mean():12.2f} ms\n")
        f.write(f"  Diff:        {df['priority_diff'].mean():+12.2f} ms ({df['priority_pct_diff'].mean():+.2f}%)\n\n")
        
        f.write("Non-Priority Queries:\n")
        f.write(f"  Greedy Avg:  {df['greedy_nonpriority_avg'].mean():12.2f} ms\n")
        f.write(f"  Default Avg: {df['default_nonpriority_avg'].mean():12.2f} ms\n")
        f.write(f"  Diff:        {df['nonpriority_diff'].mean():+12.2f} ms ({df['nonpriority_pct_diff'].mean():+.2f}%)\n\n")
        
        # Parameter impact analysis
        f.write("-" * 60 + "\n")
        f.write("PARAMETER IMPACT ANALYSIS\n")
        f.write("-" * 60 + "\n\n")
        
        # Lambda impact
        f.write("By Total Lambda (Arrival Rate):\n")
        df_sorted = df.sort_values('total_lambda')
        for _, row in df_sorted.iterrows():
            f.write(f"  λ={row['lambda1']:.4f}+{row['lambda2']:.4f}={row['total_lambda']:.4f}: "
                   f"Avg Diff={row['diff_Average']:+.2f} ms "
                   f"(P={row['priority_diff']:+.2f}, NP={row['nonpriority_diff']:+.2f})\n")
        f.write("\n")
        
        # Detailed pair-by-pair comparison
        f.write("-" * 60 + "\n")
        f.write("DETAILED PAIR-BY-PAIR COMPARISON\n")
        f.write("-" * 60 + "\n\n")
        
        for _, row in df.iterrows():
            f.write(f"Pair {int(row['greedy_prefix']):03d} (greedy) vs {int(row['default_prefix']):03d} (default)\n")
            f.write(f"  λ1={row['lambda1']:.6f}, λ2={row['lambda2']:.6f}\n")
            f.write(f"  Greedy Avg: {row['greedy_Average']:.2f} ms\n")
            f.write(f"  Default Avg: {row['default_Average']:.2f} ms\n")
            f.write(f"  Diff: {row['diff_Average']:+.2f} ms ({row['pct_diff_Average']:+.2f}%)\n")
            f.write(f"  Winner: {'Greedy' if row['diff_Average'] < 0 else 'Default'}\n\n")
    
    print(f"Summary report saved to: {report_path}")


def create_visualizations(df: pd.DataFrame, output_dir: str):
    """Create visualization plots for the comparison."""
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # 1. Average Latency Comparison Bar Chart
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Plot 1: Average latency comparison
    ax = axes[0, 0]
    x = range(len(df))
    width = 0.35
    bars1 = ax.bar([i - width/2 for i in x], df['greedy_Average'], width, label='Greedy', color='#2ecc71')
    bars2 = ax.bar([i + width/2 for i in x], df['default_Average'], width, label='Default', color='#3498db')
    ax.set_xlabel('Experiment Pair')
    ax.set_ylabel('Average Latency (ms)')
    ax.set_title('Average Latency: Greedy vs Default')
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(row['greedy_prefix']):03d}" for _, row in df.iterrows()], rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Latency difference percentage
    ax = axes[0, 1]
    colors = ['#2ecc71' if x < 0 else '#e74c3c' for x in df['pct_diff_Average']]
    ax.bar(x, df['pct_diff_Average'], color=colors)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Experiment Pair')
    ax.set_ylabel('Latency Difference (%)')
    ax.set_title('Latency Difference (Greedy - Default) %\nGreen=Greedy Better, Red=Default Better')
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(row['greedy_prefix']):03d}" for _, row in df.iterrows()], rotation=45)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Priority vs Non-Priority comparison
    ax = axes[1, 0]
    width = 0.2
    x = range(len(df))
    ax.bar([i - 1.5*width for i in x], df['greedy_priority_avg'], width, label='Greedy Priority', color='#27ae60')
    ax.bar([i - 0.5*width for i in x], df['default_priority_avg'], width, label='Default Priority', color='#2980b9')
    ax.bar([i + 0.5*width for i in x], df['greedy_nonpriority_avg'], width, label='Greedy Non-Priority', color='#2ecc71', alpha=0.6)
    ax.bar([i + 1.5*width for i in x], df['default_nonpriority_avg'], width, label='Default Non-Priority', color='#3498db', alpha=0.6)
    ax.set_xlabel('Experiment Pair')
    ax.set_ylabel('Average Latency (ms)')
    ax.set_title('Priority vs Non-Priority Query Latency')
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(row['greedy_prefix']):03d}" for _, row in df.iterrows()], rotation=45)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Lambda vs Latency Difference scatter
    ax = axes[1, 1]
    # Filter out infinite lambda ratios for color mapping
    finite_ratio = df['lambda_ratio'].replace([np.inf, -np.inf], np.nan)
    scatter = ax.scatter(df['total_lambda'], df['diff_Average'], 
                        c=finite_ratio, cmap='viridis', 
                        s=100, alpha=0.7, edgecolors='black')
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1, label='Equal Performance')
    ax.set_xlabel('Total Lambda (λ1 + λ2)')
    ax.set_ylabel('Latency Difference (Greedy - Default) ms')
    ax.set_title('Impact of Arrival Rate on Policy Difference')
    plt.colorbar(scatter, ax=ax, label='Lambda Ratio (λ1/λ2)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'policy_comparison_overview.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Percentile Distribution Comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    percentiles = ['P50', 'P90', 'P95', 'P99']
    greedy_means = [df[f'greedy_{p}'].mean() for p in percentiles]
    default_means = [df[f'default_{p}'].mean() for p in percentiles]
    
    x = range(len(percentiles))
    width = 0.35
    ax.bar([i - width/2 for i in x], greedy_means, width, label='Greedy', color='#2ecc71')
    ax.bar([i + width/2 for i in x], default_means, width, label='Default', color='#3498db')
    ax.set_xlabel('Percentile')
    ax.set_ylabel('Latency (ms)')
    ax.set_title('Percentile Distribution: Greedy vs Default (Averaged Across All Experiments)')
    ax.set_xticks(x)
    ax.set_xticklabels(percentiles)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'percentile_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. Heatmap of parameter impact
    if len(df) > 3:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Create pivot tables for heatmap
        df_pivot = df.copy()
        df_pivot['lambda1_str'] = df_pivot['lambda1'].apply(lambda x: f'{x:.4f}')
        df_pivot['lambda2_str'] = df_pivot['lambda2'].apply(lambda x: f'{x:.4f}')
        
        # Try to create a heatmap if we have enough variation
        try:
            pivot = df_pivot.pivot_table(values='diff_Average', 
                                         index='lambda1_str', 
                                         columns='lambda2_str', 
                                         aggfunc='mean')
            if pivot.size > 1:
                sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn_r', 
                           center=0, ax=axes[0])
                axes[0].set_title('Latency Diff by λ1 and λ2\n(+ve = Default Better)')
        except:
            axes[0].text(0.5, 0.5, 'Insufficient data variation\nfor heatmap', 
                        ha='center', va='center', transform=axes[0].transAxes)
            axes[0].set_title('Latency Diff Heatmap')
        
        # Lambda ratio impact
        ax = axes[1]
        df_sorted = df.sort_values('lambda_ratio')
        ax.plot(range(len(df_sorted)), df_sorted['diff_Average'], 'o-', color='#9b59b6', linewidth=2, markersize=8)
        ax.axhline(y=0, color='red', linestyle='--', linewidth=1)
        ax.set_xlabel('Experiments (sorted by λ1/λ2 ratio)')
        ax.set_ylabel('Latency Difference (ms)')
        ax.set_title('Impact of Lambda Ratio on Policy Difference')
        ax.grid(True, alpha=0.3)
        
        # Add ratio labels
        for i, (_, row) in enumerate(df_sorted.iterrows()):
            if row['lambda_ratio'] != float('inf'):
                ax.annotate(f"{row['lambda_ratio']:.2f}", (i, row['diff_Average']), 
                           textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'parameter_impact.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    # 4. Box plot of latency distributions
    fig, ax = plt.subplots(figsize=(10, 6))
    
    data_to_plot = [
        df['greedy_Average'], df['default_Average'],
        df['greedy_P50'], df['default_P50'],
        df['greedy_P99'], df['default_P99']
    ]
    labels = ['Greedy\nAvg', 'Default\nAvg', 'Greedy\nP50', 'Default\nP50', 'Greedy\nP99', 'Default\nP99']
    colors = ['#2ecc71', '#3498db'] * 3
    
    bp = ax.boxplot(data_to_plot, patch_artist=True, labels=labels)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel('Latency (ms)')
    ax.set_title('Latency Distribution Comparison')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'latency_boxplot.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Visualizations saved to: {output_dir}")


def analyze_directory(input_dir: str, output_dir: str = None):
    """Main analysis function for a single output directory."""
    if output_dir is None:
        output_dir = os.path.join(input_dir, 'analysis')
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all stats files
    stats_files = glob.glob(os.path.join(input_dir, '*_stats.txt'))
    
    if not stats_files:
        print(f"No stats files found in {input_dir}")
        return None
    
    print(f"Found {len(stats_files)} stats files")
    
    # Parse all files
    stats_list = []
    for filepath in stats_files:
        data = parse_stats_file(filepath)
        if data:
            stats_list.append(data)
    
    print(f"Successfully parsed {len(stats_list)} files")
    
    # Find pairs
    pairs = find_pairs(stats_list)
    print(f"Found {len(pairs)} greedy-default pairs")
    
    if not pairs:
        print("No matching pairs found!")
        return None
    
    # Compute comparison metrics
    comparison_data = []
    for greedy, default in pairs:
        metrics = compute_comparison_metrics(greedy, default)
        comparison_data.append(metrics)
    
    # Create DataFrame
    df = pd.DataFrame(comparison_data)
    
    # Save raw comparison data
    csv_path = os.path.join(output_dir, 'policy_comparison_data.csv')
    df.to_csv(csv_path, index=False)
    print(f"Comparison data saved to: {csv_path}")
    
    # Generate reports
    generate_summary_report(df, output_dir)
    
    # Create visualizations
    create_visualizations(df, output_dir)
    
    return df


def analyze_all_outputs(base_dir: str):
    """Analyze all subdirectories in the output folder."""
    subdirs = [d for d in os.listdir(base_dir) 
               if os.path.isdir(os.path.join(base_dir, d)) and d[0].isdigit()]
    
    all_results = {}
    for subdir in subdirs:
        input_dir = os.path.join(base_dir, subdir)
        print(f"\n{'='*60}")
        print(f"Analyzing: {subdir}")
        print('='*60)
        
        df = analyze_directory(input_dir)
        if df is not None:
            all_results[subdir] = df
    
    # Create combined analysis if multiple directories
    if len(all_results) > 1:
        combined_output = os.path.join(base_dir, 'combined_analysis')
        os.makedirs(combined_output, exist_ok=True)
        
        combined_df = pd.concat(
            [df.assign(experiment_batch=name) for name, df in all_results.items()],
            ignore_index=True
        )
        combined_df.to_csv(os.path.join(combined_output, 'all_experiments_comparison.csv'), index=False)
        print(f"\nCombined results saved to: {combined_output}")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description='Analyze policy comparison experiments')
    parser.add_argument('input', nargs='?', default=None,
                       help='Input directory (specific experiment or base output folder)')
    parser.add_argument('--all', action='store_true',
                       help='Analyze all subdirectories in the output folder')
    parser.add_argument('--output', '-o', default=None,
                       help='Output directory for analysis results')
    
    args = parser.parse_args()
    
    # Default to tpch/output if no input specified
    if args.input is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        args.input = os.path.join(script_dir, 'output')
    
    if not os.path.exists(args.input):
        print(f"Error: Input directory does not exist: {args.input}")
        return
    
    if args.all or (os.path.isdir(args.input) and 
                    any(d[0].isdigit() for d in os.listdir(args.input) 
                        if os.path.isdir(os.path.join(args.input, d)))):
        # Analyze all experiment directories
        analyze_all_outputs(args.input)
    else:
        # Analyze single directory
        analyze_directory(args.input, args.output)


if __name__ == '__main__':
    main()
