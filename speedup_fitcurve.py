#!/usr/bin/env python3
"""
Speedup Curve Fitting Tool

Fits Amdahl's law to sparse speedup samples and generates a complete speedup curve.

Input format (stdin or file): Each line contains "speedup cores" (space or comma separated)
Output format: One speedup value per line (64 lines by default)

Amdahl's Law: Speedup(N) = 1 / (s + (1-s)/N)
where s is the serial fraction of the workload.

If the speedup increment becomes too low (plateau detected), the fitted curve
is flattened to that plateau value.

Usage:
    # From stdin
    echo "1.0 1
    3.5 4
    7.2 8
    12.1 16" | python speedup_fitcurve.py
    
    # From file
    python speedup_fitcurve.py --input samples.txt --output speedup.csv
    
    # Custom max threads
    python speedup_fitcurve.py --input samples.txt --max-threads 32
    
    # Adjust plateau threshold
    python speedup_fitcurve.py --input samples.txt --plateau-threshold 0.1
"""

import argparse
import sys
from typing import List, Tuple, Optional

try:
    import numpy as np
    from scipy.optimize import curve_fit
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def amdahl_law(n: np.ndarray, s: float) -> np.ndarray:
    """
    Amdahl's Law speedup function.
    
    Args:
        n: Number of processors/threads
        s: Serial fraction (0 <= s <= 1)
    
    Returns:
        Speedup = 1 / (s + (1-s)/n)
    """
    return 1.0 / (s + (1.0 - s) / n)


def amdahl_law_with_overhead(n: np.ndarray, s: float, overhead: float) -> np.ndarray:
    """
    Modified Amdahl's Law with parallelization overhead.
    
    Args:
        n: Number of processors/threads
        s: Serial fraction (0 <= s <= 1)
        overhead: Per-thread overhead factor
    
    Returns:
        Speedup accounting for overhead
    """
    base_speedup = 1.0 / (s + (1.0 - s) / n)
    overhead_factor = 1.0 + overhead * (n - 1)
    return base_speedup / overhead_factor


def fit_amdahl(thread_counts: List[int], speedups: List[float], 
               use_overhead: bool = False) -> Tuple[float, Optional[float]]:
    """
    Fit Amdahl's law to observed speedup data.
    
    Args:
        thread_counts: List of thread counts sampled
        speedups: Corresponding observed speedups
        use_overhead: If True, fit model with overhead term
    
    Returns:
        (serial_fraction, overhead) - overhead is None if use_overhead=False
    """
    if not HAS_SCIPY:
        # Fallback: estimate s from asymptotic speedup
        max_speedup = max(speedups)
        s = 1.0 / max_speedup if max_speedup > 0 else 1.0
        return s, None
    
    x = np.array(thread_counts, dtype=float)
    y = np.array(speedups, dtype=float)
    
    try:
        if use_overhead:
            # Fit with overhead parameter
            popt, _ = curve_fit(
                amdahl_law_with_overhead, x, y,
                p0=[0.1, 0.001],  # Initial guess: 10% serial, small overhead
                bounds=([0.0, 0.0], [1.0, 0.1]),  # s in [0,1], overhead in [0, 0.1]
                maxfev=5000
            )
            return popt[0], popt[1]
        else:
            # Fit standard Amdahl's law
            popt, _ = curve_fit(
                amdahl_law, x, y,
                p0=[0.1],  # Initial guess: 10% serial fraction
                bounds=([0.0], [1.0]),  # s must be in [0, 1]
                maxfev=5000
            )
            return popt[0], None
    except Exception as e:
        print(f"Warning: Curve fitting failed ({e}), using fallback estimation", file=sys.stderr)
        # Fallback estimation
        max_speedup = max(speedups)
        s = 1.0 / max_speedup if max_speedup > 0 else 1.0
        return min(max(s, 0.0), 1.0), None


def detect_plateau(thread_counts: List[int], speedups: List[float], 
                   threshold: float = 0.05) -> Tuple[bool, int, float]:
    """
    Detect if speedup has plateaued (incremental gain below threshold).
    
    Args:
        thread_counts: List of thread counts
        speedups: Corresponding speedups
        threshold: Relative improvement threshold (e.g., 0.05 = 5%)
    
    Returns:
        (has_plateau, plateau_thread_count, plateau_speedup)
    """
    if len(speedups) < 2:
        return False, thread_counts[-1], speedups[-1]
    
    for i in range(1, len(speedups)):
        prev_speedup = speedups[i - 1]
        curr_speedup = speedups[i]
        
        if prev_speedup > 0:
            relative_gain = (curr_speedup - prev_speedup) / prev_speedup
            
            # If gain is below threshold (or negative), we've hit plateau
            if relative_gain < threshold:
                return True, thread_counts[i - 1], prev_speedup
    
    return False, thread_counts[-1], speedups[-1]


def generate_full_speedup_curve(
    max_threads: int,
    serial_fraction: float,
    overhead: Optional[float],
    plateau_thread: int,
    plateau_speedup: float,
    has_plateau: bool
) -> List[float]:
    """
    Generate full speedup curve from 1 to max_threads using fitted model.
    
    If plateau is detected, the curve flattens at the plateau point.
    """
    speedups = []
    
    for n in range(1, max_threads + 1):
        if has_plateau and n >= plateau_thread:
            # Flatten at plateau value
            speedup = plateau_speedup
        else:
            # Use fitted Amdahl's law
            if HAS_SCIPY:
                if overhead is not None:
                    speedup = float(amdahl_law_with_overhead(np.array([n]), serial_fraction, overhead)[0])
                else:
                    speedup = float(amdahl_law(np.array([n]), serial_fraction)[0])
            else:
                # Fallback without numpy
                speedup = 1.0 / (serial_fraction + (1.0 - serial_fraction) / n)
        
        speedups.append(speedup)
    
    return speedups


def parse_input(lines: List[str]) -> Tuple[List[int], List[float]]:
    """
    Parse input lines to extract thread counts and speedups.
    
    Each line should contain: speedup cores (space or comma separated)
    
    Returns:
        (thread_counts, speedups)
    """
    thread_counts = []
    speedups = []
    
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        
        # Split by space or comma
        parts = line.replace(',', ' ').split()
        if len(parts) >= 2:
            speedup = float(parts[0])
            cores = int(parts[1])
            speedups.append(speedup)
            thread_counts.append(cores)
    
    # Sort by thread count
    sorted_pairs = sorted(zip(thread_counts, speedups))
    thread_counts = [p[0] for p in sorted_pairs]
    speedups = [p[1] for p in sorted_pairs]
    
    return thread_counts, speedups


def fit_and_generate(
    thread_counts: List[int],
    speedups: List[float],
    max_threads: int,
    plateau_threshold: float,
    use_overhead: bool,
    verbose: bool
) -> Tuple[List[float], dict]:
    """
    Fit Amdahl's law and generate full speedup curve.
    
    Returns:
        (full_speedups, metadata_dict)
    """
    # Detect plateau
    has_plateau, plateau_thread, plateau_speedup = detect_plateau(
        thread_counts, speedups, plateau_threshold
    )
    
    if verbose:
        if has_plateau:
            print(f"Plateau detected at {plateau_thread} threads (speedup: {plateau_speedup:.2f})", file=sys.stderr)
        else:
            print("No plateau detected", file=sys.stderr)
    
    # Fit Amdahl's law
    serial_fraction, overhead = fit_amdahl(thread_counts, speedups, use_overhead)
    
    if verbose:
        print(f"Fitted serial fraction: {serial_fraction:.4f}", file=sys.stderr)
        if overhead is not None:
            print(f"Fitted overhead: {overhead:.6f}", file=sys.stderr)
        theoretical_max = 1.0 / serial_fraction if serial_fraction > 0 else float('inf')
        print(f"Theoretical max speedup (Amdahl): {theoretical_max:.2f}", file=sys.stderr)
    
    # Generate full speedup curve
    full_speedups = generate_full_speedup_curve(
        max_threads, serial_fraction, overhead,
        plateau_thread, plateau_speedup, has_plateau
    )
    
    metadata = {
        'serial_fraction': serial_fraction,
        'overhead': overhead,
        'has_plateau': has_plateau,
        'plateau_thread': plateau_thread,
        'plateau_speedup': plateau_speedup,
        'sample_points': thread_counts,
        'sampled_speedups': speedups,
    }
    
    return full_speedups, metadata


def main():
    parser = argparse.ArgumentParser(
        description="Fit Amdahl's law to speedup samples and generate full curve"
    )
    
    # Input/output
    parser.add_argument("--input", "-i", help="Input file (default: stdin)")
    parser.add_argument("--output", "-o", help="Output file (default: stdout)")
    parser.add_argument("--meta-output", "-m", help="Metadata output file (optional)")
    
    # Fitting parameters
    parser.add_argument("--max-threads", "-t", type=int, default=64,
                        help="Maximum number of threads for output curve (default: 64)")
    parser.add_argument("--plateau-threshold", "-p", type=float, default=0.001,
                        help="Relative speedup improvement threshold for plateau detection (default: 0.001)")
    parser.add_argument("--with-overhead", action="store_true",
                        help="Use Amdahl's law model with parallelization overhead term")
    
    # Output options
    parser.add_argument("--verbose", "-v", action="store_true", 
                        help="Print fitting info to stderr")
    
    args = parser.parse_args()
    
    if not HAS_SCIPY:
        print("Warning: scipy not installed. Curve fitting will use fallback estimation.", file=sys.stderr)
    
    # Read input
    if args.input:
        with open(args.input, 'r') as f:
            lines = f.readlines()
    else:
        lines = sys.stdin.readlines()
    
    # Parse input
    thread_counts, speedups = parse_input(lines)
    
    if not thread_counts:
        print("Error: No valid input data found", file=sys.stderr)
        sys.exit(1)
    
    if args.verbose:
        print(f"Parsed {len(thread_counts)} sample points", file=sys.stderr)
        for tc, sp in zip(thread_counts, speedups):
            print(f"  {tc} cores: {sp:.2f}x speedup", file=sys.stderr)
    
    # Fit and generate
    full_speedups, metadata = fit_and_generate(
        thread_counts, speedups,
        args.max_threads, args.plateau_threshold, args.with_overhead, args.verbose
    )
    
    # Write output
    if args.output:
        with open(args.output, 'w') as f:
            for s in full_speedups:
                f.write(f"{s:.4f}\n")
    else:
        for s in full_speedups:
            print(f"{s:.4f}")
    
    # Write metadata if requested
    if args.meta_output:
        with open(args.meta_output, 'w') as f:
            f.write(f"serial_fraction: {metadata['serial_fraction']:.6f}\n")
            if metadata['overhead'] is not None:
                f.write(f"overhead: {metadata['overhead']:.6f}\n")
            f.write(f"has_plateau: {metadata['has_plateau']}\n")
            f.write(f"plateau_thread: {metadata['plateau_thread']}\n")
            f.write(f"plateau_speedup: {metadata['plateau_speedup']:.4f}\n")
            f.write(f"sample_points: {metadata['sample_points']}\n")
            f.write(f"sampled_speedups: {metadata['sampled_speedups']}\n")


# Exported functions for use as a library
def fit_speedup_curve(
    thread_counts: List[int],
    speedups: List[float],
    max_threads: int = 64,
    plateau_threshold: float = 0.001,
    use_overhead: bool = False
) -> Tuple[List[float], dict]:
    """
    Library function to fit speedup curve.
    
    Args:
        thread_counts: List of thread counts sampled
        speedups: Corresponding observed speedups
        max_threads: Maximum threads for output curve
        plateau_threshold: Threshold for plateau detection
        use_overhead: Use model with overhead term
    
    Returns:
        (full_speedups, metadata_dict)
    """
    return fit_and_generate(
        thread_counts, speedups,
        max_threads, plateau_threshold, use_overhead, verbose=False
    )


if __name__ == "__main__":
    main()
