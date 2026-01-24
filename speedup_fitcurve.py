#!/usr/bin/env python3
"""
Speedup Curve Fitting Tool

Fits scalability models to sparse speedup samples and generates a complete speedup curve.

Supported models:
- Amdahl's Law: Speedup(N) = 1 / (s + (1-s)/N)
  where s is the serial fraction of the workload.
  
- Universal Scalability Law (USL): Speedup(N) = N / (1 + o(N-1) + κ*N*(N-1))
  where o(sigma) is the contention parameter and κ (kappa) is the coherency parameter.
  USL can model speedup degradation at high thread counts (retrograde behavior).

Input format (stdin or file): Each line contains "speedup cores" (space or comma separated)
Output format: One speedup value per line (64 lines by default)
Usage:
    # From stdin (default: Amdahl's law)
    echo "1.0 1
    3.5 4
    7.2 8
    12.1 16" | python speedup_fitcurve.py
    
    # From file with USL model
    python speedup_fitcurve.py --input samples.txt --model usl
    
    # Custom max threads
    python speedup_fitcurve.py --input samples.txt --max-threads 32
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


# =============================================================================
# Scalability Models
# =============================================================================

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


def usl_law(n: np.ndarray, sigma: float, kappa: float) -> np.ndarray:
    """
    Universal Scalability Law (USL) speedup function.
    
    Args:
        n: Number of processors/threads
        sigma: Contention/serialization parameter (0 <= sigma <= 1)
        kappa: Coherency/crosstalk parameter (0 <= kappa)
    
    Returns:
        Speedup = N / (1 + sigma*(N-1) + kappa*N*(N-1))
    
    Note: USL can model retrograde behavior where speedup decreases at high N.
    """
    return n / (1.0 + sigma * (n - 1) + kappa * n * (n - 1))


def amdahl_law_scalar(n: float, s: float) -> float:
    """Scalar version of Amdahl's law for use without numpy."""
    return 1.0 / (s + (1.0 - s) / n)


def usl_law_scalar(n: float, sigma: float, kappa: float) -> float:
    """Scalar version of USL for use without numpy."""
    return n / (1.0 + sigma * (n - 1) + kappa * n * (n - 1))


# =============================================================================
# Curve Fitting
# =============================================================================

def fit_amdahl(thread_counts: List[int], speedups: List[float]) -> Tuple[float, dict]:
    """
    Fit Amdahl's law to observed speedup data.
    
    Args:
        thread_counts: List of thread counts sampled
        speedups: Corresponding observed speedups
    
    Returns:
        (serial_fraction, params_dict)
    """
    if not HAS_SCIPY:
        # Fallback: estimate s from asymptotic speedup
        max_speedup = max(speedups)
        s = 1.0 / max_speedup if max_speedup > 0 else 1.0
        return s, {'s': s}
    
    x = np.array(thread_counts, dtype=float)
    y = np.array(speedups, dtype=float)
    
    try:
        popt, _ = curve_fit(
            amdahl_law, x, y,
            p0=[0.1],  # Initial guess: 10% serial fraction
            bounds=([0.0], [1.0]),  # s must be in [0, 1]
            maxfev=5000
        )
        return popt[0], {'s': popt[0]}
    except Exception as e:
        print(f"Warning: Amdahl fitting failed ({e}), using fallback estimation", file=sys.stderr)
        max_speedup = max(speedups)
        s = 1.0 / max_speedup if max_speedup > 0 else 1.0
        s = min(max(s, 0.0), 1.0)
        return s, {'s': s}


def fit_usl(thread_counts: List[int], speedups: List[float]) -> Tuple[Tuple[float, float], dict]:
    """
    Fit Universal Scalability Law to observed speedup data.
    
    Args:
        thread_counts: List of thread counts sampled
        speedups: Corresponding observed speedups
    
    Returns:
        ((sigma, kappa), params_dict)
    """
    if not HAS_SCIPY:
        # Fallback: estimate sigma from max speedup, assume kappa=0
        max_speedup = max(speedups)
        max_n = max(thread_counts)
        # From USL with kappa=0: Speedup = N / (1 + sigma*(N-1))
        # At max_n: max_speedup = max_n / (1 + sigma*(max_n-1))
        # Solving: sigma = (max_n/max_speedup - 1) / (max_n - 1)
        if max_n > 1 and max_speedup > 0:
            sigma = (max_n / max_speedup - 1) / (max_n - 1)
            sigma = max(0.0, min(1.0, sigma))
        else:
            sigma = 0.1
        return (sigma, 0.0), {'sigma': sigma, 'kappa': 0.0}
    
    x = np.array(thread_counts, dtype=float)
    y = np.array(speedups, dtype=float)
    
    try:
        popt, _ = curve_fit(
            usl_law, x, y,
            p0=[0.1, 0.0001],  # Initial guess
            bounds=([0.0, 0.0], [1.0, 0.1]),  # sigma in [0,1], kappa in [0, 0.1]
            maxfev=5000
        )
        return (popt[0], popt[1]), {'sigma': popt[0], 'kappa': popt[1]}
    except Exception as e:
        print(f"Warning: USL fitting failed ({e}), using fallback estimation", file=sys.stderr)
        max_speedup = max(speedups)
        max_n = max(thread_counts)
        if max_n > 1 and max_speedup > 0:
            sigma = (max_n / max_speedup - 1) / (max_n - 1)
            sigma = max(0.0, min(1.0, sigma))
        else:
            sigma = 0.1
        return (sigma, 0.0), {'sigma': sigma, 'kappa': 0.0}

# =============================================================================
# Curve Generation
# =============================================================================

def generate_full_speedup_curve(
    max_threads: int,
    model: str,
    params: dict,
) -> List[float]:
    """
    Generate full speedup curve from 1 to max_threads using fitted model.
    """
    speedups = []
    
    for n in range(1, max_threads + 1):
        # Use fitted model
        if model == 'amdahl':
            s = params['s']
            speedup = amdahl_law_scalar(n, s)
        elif model == 'usl':
            sigma, kappa = params['sigma'], params['kappa']
            speedup = usl_law_scalar(n, sigma, kappa)
        else:
            raise ValueError(f"Unknown model: {model}")
        
        speedups.append(speedup)
    
    return speedups


# =============================================================================
# Input Parsing
# =============================================================================

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
            speedup = float(parts[1])
            cores = int(parts[0])
            speedups.append(speedup)
            thread_counts.append(cores)
    
    # Sort by thread count
    sorted_pairs = sorted(zip(thread_counts, speedups))
    thread_counts = [p[0] for p in sorted_pairs]
    speedups = [p[1] for p in sorted_pairs]
    
    return thread_counts, speedups


# =============================================================================
# Main Fitting Logic
# =============================================================================

def fit_and_generate(
    thread_counts: List[int],
    speedups: List[float],
    max_threads: int,
    model: str,
    verbose: bool
) -> Tuple[List[float], dict]:
    """
    Fit scalability model and generate full speedup curve.
    
    Returns:
        (full_speedups, metadata_dict)
    """

    # Fit model
    if model == 'amdahl':
        serial_fraction, params = fit_amdahl(thread_counts, speedups)
        if verbose:
            print(f"Model: Amdahl's Law", file=sys.stderr)
            print(f"Fitted serial fraction (s): {serial_fraction:.4f}", file=sys.stderr)
            theoretical_max = 1.0 / serial_fraction if serial_fraction > 0 else float('inf')
            print(f"Theoretical max speedup: {theoretical_max:.2f}", file=sys.stderr)
    elif model == 'usl':
        (sigma, kappa), params = fit_usl(thread_counts, speedups)
        if verbose:
            print(f"Model: Universal Scalability Law (USL)", file=sys.stderr)
            print(f"Fitted sigma (contention): {sigma:.6f}", file=sys.stderr)
            print(f"Fitted kappa (coherency): {kappa:.6f}", file=sys.stderr)
            # Peak speedup for USL occurs at N* = sqrt((1-sigma)/kappa) if kappa > 0
            if kappa > 0 and sigma < 1:
                n_peak = ((1 - sigma) / kappa) ** 0.5
                peak_speedup = usl_law_scalar(n_peak, sigma, kappa)
                print(f"Peak speedup: {peak_speedup:.2f} at N={n_peak:.1f} threads", file=sys.stderr)
            else:
                theoretical_max = 1.0 / sigma if sigma > 0 else float('inf')
                print(f"Theoretical max speedup (sigma only): {theoretical_max:.2f}", file=sys.stderr)
    else:
        raise ValueError(f"Unknown model: {model}")
    
    # Generate full speedup curve
    full_speedups = generate_full_speedup_curve(
        max_threads, model, params
    )
    
    metadata = {
        'model': model,
        'params': params,
        'sample_points': thread_counts,
        'sampled_speedups': speedups,
    }
    
    return full_speedups, metadata


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Fit scalability model to speedup samples and generate full curve"
    )
    
    # Input/output
    parser.add_argument("--input", "-i", help="Input file (default: stdin)")
    parser.add_argument("--output", "-o", help="Output file (default: stdout)")
    parser.add_argument("--meta-output", "-m", help="Metadata output file (optional)")
    
    # Model selection
    parser.add_argument("--model", choices=['amdahl', 'usl'], default='amdahl',
                        help="Scalability model to fit (default: amdahl)")
    
    # Fitting parameters
    parser.add_argument("--max-threads", "-t", type=int, default=64,
                        help="Maximum number of threads for output curve (default: 64)")
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
        args.max_threads,  args.model, args.verbose
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
            f.write(f"model: {metadata['model']}\n")
            for key, value in metadata['params'].items():
                f.write(f"{key}: {value:.6f}\n")
            f.write(f"sample_points: {metadata['sample_points']}\n")
            f.write(f"sampled_speedups: {metadata['sampled_speedups']}\n")


# =============================================================================
# Library API
# =============================================================================

def fit_speedup_curve(
    thread_counts: List[int],
    speedups: List[float],
    max_threads: int = 32,
    model: str = 'usl'
) -> Tuple[List[float], dict]:
    """
    Library function to fit speedup curve.
    
    Args:
        thread_counts: List of thread counts sampled
        speedups: Corresponding observed speedups
        max_threads: Maximum threads for output curve
        model: 'amdahl' or 'usl'
    
    Returns:
        (full_speedups, metadata_dict)
    """
    return fit_and_generate(
        thread_counts, speedups,
        max_threads, model, verbose=False
    )


if __name__ == "__main__":
    main()
