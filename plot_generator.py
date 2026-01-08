#!/usr/bin/env python3
"""
Standalone script to generate timeline plots from raw CSV output files.

Usage:
    # Generate plots for all raw CSVs in all folders under ./output
    python3 plot_generator.py

    # Generate plots for a specific output folder
    python3 plot_generator.py --output-dir ./output/20260108_143022

    # Generate plot for a single CSV file
    python3 plot_generator.py --csv path/to/schedule_raw.csv

    # Show plot interactively (single CSV mode only)
    python3 plot_generator.py --csv path/to/schedule_raw.csv --show
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
from dataclasses import dataclass
from typing import List, Dict

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# Color palette for different query IDs
QUERY_COLORS = [
    "#e63946",  # red
    "#457b9d",  # blue
    "#2a9d8f",  # teal
    "#e9c46a",  # yellow
    "#f4a261",  # orange
    "#9b5de5",  # purple
    "#00f5d4",  # cyan
    "#f15bb5",  # pink
]


@dataclass
class LatencyRecord:
    at_ms: int
    latency_ms: float
    qid: str


def read_raw_csv(path: str) -> tuple[List[LatencyRecord], List[str]]:
    """Read raw CSV file and return records and unique qid list (in order of appearance)."""
    records: List[LatencyRecord] = []
    qid_list: List[str] = []
    seen_qids: set = set()
    
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            at_ms = int(row["at_ms"])
            latency_ms = float(row["latency_ms"])
            qid = row["qid"].strip()
            records.append(LatencyRecord(at_ms=at_ms, latency_ms=latency_ms, qid=qid))
            
            if qid not in seen_qids:
                qid_list.append(qid)
                seen_qids.add(qid)
    
    return records, qid_list


def generate_timeline_plot(
    records: List[LatencyRecord],
    qid_list: List[str],
    title: str = "Query Timeline",
    output_path: str | None = None,
    show: bool = False,
) -> None:
    """Generate and optionally save/show a timeline plot."""
    if not records:
        print("No records to plot.")
        return
    
    # Sort records by arrival time
    sorted_records = sorted(records, key=lambda r: (r.at_ms, r.qid))
    
    # Build color map for query IDs
    color_map: Dict[str, str] = {}
    for i, qid in enumerate(qid_list):
        color_map[qid] = QUERY_COLORS[i % len(QUERY_COLORS)]
    
    # Convert to seconds for display
    starts = [r.at_ms / 1000.0 for r in sorted_records]
    durations = [r.latency_ms / 1000.0 for r in sorted_records]
    qids = [r.qid for r in sorted_records]
    
    # Create figure
    fig_height = max(4, 0.25 * len(sorted_records))
    fig, ax = plt.subplots(figsize=(12, fig_height))
    
    # Draw horizontal bars for each query
    for i in range(len(sorted_records)):
        color = color_map.get(qids[i], "#888888")
        ax.hlines(y=i, xmin=starts[i], xmax=starts[i] + durations[i], 
                  linewidth=6, colors=color)
    
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Query instances (earliest arrival at top)")
    ax.set_yticks([])
    ax.invert_yaxis()
    ax.grid(True, axis="x", linestyle="--", linewidth=0.6)
    ax.set_title(title)
    
    # Create legend
    legend_items = [
        Line2D([0], [0], color=color_map[qid], lw=6, label=qid)
        for qid in qid_list
    ]
    ax.legend(handles=legend_items, loc="upper right")
    
    plt.tight_layout()
    
    # Save if output path provided
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {output_path}")
    
    # Show if requested
    if show:
        plt.show()
    else:
        plt.close(fig)


def process_csv_file(csv_path: str, show: bool = False) -> None:
    """Process a single CSV file and generate its plot."""
    # Read the CSV
    records, qid_list = read_raw_csv(csv_path)
    
    # Determine output path
    if csv_path.endswith("_raw.csv"):
        output_path = csv_path[:-8] + "_timeline.png"
    elif csv_path.endswith(".csv"):
        output_path = csv_path[:-4] + "_timeline.png"
    else:
        output_path = csv_path + "_timeline.png"
    
    # Determine title
    basename = os.path.basename(csv_path)
    if basename.endswith("_raw.csv"):
        schedule_name = basename[:-8]
    elif basename.endswith(".csv"):
        schedule_name = basename[:-4]
    else:
        schedule_name = basename
    title = f"Query Timeline: {schedule_name}"
    
    # Generate the plot
    generate_timeline_plot(
        records=records,
        qid_list=qid_list,
        title=title,
        output_path=output_path if not show else None,
        show=show,
    )


def process_output_folder(folder_path: str) -> int:
    """Process all *_raw.csv files in a folder. Returns count of files processed."""
    csv_pattern = os.path.join(folder_path, "*_raw.csv")
    csv_files = sorted(glob.glob(csv_pattern))
    
    if not csv_files:
        return 0
    
    print(f"\nProcessing folder: {folder_path}")
    for csv_path in csv_files:
        print(f"  Processing: {os.path.basename(csv_path)}")
        try:
            process_csv_file(csv_path)
        except Exception as e:
            print(f"    Error: {e}")
    
    return len(csv_files)


def main():
    ap = argparse.ArgumentParser(
        description="Generate timeline plots from raw CSV output files."
    )
    ap.add_argument("--csv", help="Path to a single raw CSV file to process")
    ap.add_argument("--output-dir", help="Process a specific output folder")
    ap.add_argument("--base-dir", default="./output", 
                    help="Base directory containing output folders (default: ./output)")
    ap.add_argument("--show", action="store_true", 
                    help="Display plot interactively (only works with --csv)")
    ap.add_argument("--title", help="Custom title for the plot (only works with --csv)")
    args = ap.parse_args()
    
    # Mode 1: Single CSV file
    if args.csv:
        if not os.path.exists(args.csv):
            print(f"Error: File not found: {args.csv}")
            return
        
        records, qid_list = read_raw_csv(args.csv)
        print(f"Loaded {len(records)} records with {len(qid_list)} query type(s): {', '.join(qid_list)}")
        
        # Determine output path
        if args.csv.endswith("_raw.csv"):
            output_path = args.csv[:-8] + "_timeline.png"
        elif args.csv.endswith(".csv"):
            output_path = args.csv[:-4] + "_timeline.png"
        else:
            output_path = args.csv + "_timeline.png"
        
        # Determine title
        if args.title:
            title = args.title
        else:
            basename = os.path.basename(args.csv)
            if basename.endswith("_raw.csv"):
                schedule_name = basename[:-8]
            elif basename.endswith(".csv"):
                schedule_name = basename[:-4]
            else:
                schedule_name = basename
            title = f"Query Timeline: {schedule_name}"
        
        generate_timeline_plot(
            records=records,
            qid_list=qid_list,
            title=title,
            output_path=output_path if not args.show else None,
            show=args.show,
        )
        if not args.show:
            print(f"Plot saved to: {output_path}")
        return
    
    # Mode 2: Specific output folder
    if args.output_dir:
        if not os.path.isdir(args.output_dir):
            print(f"Error: Directory not found: {args.output_dir}")
            return
        
        count = process_output_folder(args.output_dir)
        print(f"\nProcessed {count} CSV file(s) in {args.output_dir}")
        return
    
    # Mode 3: All folders under base directory
    if not os.path.isdir(args.base_dir):
        print(f"Error: Base directory not found: {args.base_dir}")
        return
    
    # Find all subdirectories in base_dir
    subdirs = sorted([
        os.path.join(args.base_dir, d) 
        for d in os.listdir(args.base_dir) 
        if os.path.isdir(os.path.join(args.base_dir, d))
    ])
    
    if not subdirs:
        print(f"No output folders found in {args.base_dir}")
        return
    
    print(f"Found {len(subdirs)} output folder(s) in {args.base_dir}")
    
    total_count = 0
    for folder in subdirs:
        count = process_output_folder(folder)
        total_count += count
    
    print(f"\n{'='*60}")
    print(f"Done! Generated {total_count} plot(s) across {len(subdirs)} folder(s)")


if __name__ == "__main__":
    main()
