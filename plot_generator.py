#!/usr/bin/env python3
"""
Standalone script to generate timeline plots from raw CSV output files.

Usage:
    python3 plot_generator.py path/to/schedule_raw.csv
    python3 plot_generator.py path/to/schedule_raw.csv --output my_plot.png
    python3 plot_generator.py path/to/schedule_raw.csv --show
"""
from __future__ import annotations

import argparse
import csv
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
        print(f"Plot saved to: {output_path}")
    
    # Show if requested
    if show:
        plt.show()
    else:
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser(
        description="Generate timeline plots from raw CSV output files."
    )
    ap.add_argument("csv_file", help="Path to the raw CSV file (e.g., schedule_raw.csv)")
    ap.add_argument("--output", "-o", help="Output path for the plot image (default: <csv_basename>_timeline.png)")
    ap.add_argument("--show", action="store_true", help="Display the plot interactively")
    ap.add_argument("--title", help="Custom title for the plot")
    args = ap.parse_args()
    
    if not os.path.exists(args.csv_file):
        print(f"Error: File not found: {args.csv_file}")
        return
    
    # Read the CSV
    records, qid_list = read_raw_csv(args.csv_file)
    print(f"Loaded {len(records)} records with {len(qid_list)} query type(s): {', '.join(qid_list)}")
    
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        # Default: replace _raw.csv with _timeline.png or append _timeline.png
        base = args.csv_file
        if base.endswith("_raw.csv"):
            output_path = base[:-8] + "_timeline.png"
        elif base.endswith(".csv"):
            output_path = base[:-4] + "_timeline.png"
        else:
            output_path = base + "_timeline.png"
    
    # Determine title
    if args.title:
        title = args.title
    else:
        # Extract schedule name from filename
        basename = os.path.basename(args.csv_file)
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
        output_path=output_path if not args.show or args.output else None,
        show=args.show,
    )
    
    # If show was requested but no explicit output, still save by default
    if args.show and not args.output:
        # Don't save if only showing
        pass
    elif not args.show:
        # Already saved above
        pass


if __name__ == "__main__":
    main()
