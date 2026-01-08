#!/usr/bin/env python3
# python3 poisson_schedule_generator.py --length 30 --lam1 0.02 --lam2 0.04

import argparse
import csv
import random
from dataclasses import dataclass

@dataclass(frozen=True)
class Event:
    at_ms: int
    qid: str

def gen_poisson_events(rate_qps: float, length: float, qid: str, rng: random.Random):
    t = 0.0
    out = []
    if rate_qps <= 0:
        return out
    while True:
        t += rng.expovariate(rate_qps)  # inter-arrival ~ Exp(rate)
        if t > length:
            break
        out.append((t, qid))
    return out

def main():
    ap = argparse.ArgumentParser(description="Generate sorted ClickHouse schedule.csv (at_ms,qid) for two Poisson classes.")
    # Remove the --out argument line
    ap.add_argument("--length", type=float, required=True, help="Total duration to generate (seconds)")
    ap.add_argument("--lam1", type=float, required=True, help="Class 1 rate (QPS)")
    ap.add_argument("--lam2", type=float, required=True, help="Class 2 rate (QPS)")
    ap.add_argument("--qid1", default="q0001", help="Class 1 qid (maps to queries/<qid>.sql)")
    ap.add_argument("--qid2", default="q0002", help="Class 2 qid (maps to queries/<qid>.sql)")
    ap.add_argument("--seed", type=int, default=123, help="RNG seed")
    ap.add_argument("--rounding", choices=["floor", "round"], default="round",
                    help="How to convert seconds to milliseconds")
    ap.add_argument("--prefix", default="", help="Prefix for output filename (e.g., '001_')")
    args = ap.parse_args()

    # Generate output filename from parameters
    # Format: [prefix]qid1_qid2_lam{lam1}_{lam2}_len{length}_s{seed}_{rounding}.csv
    lam1_str = f"{args.lam1:.6f}".rstrip('0').rstrip('.')
    lam2_str = f"{args.lam2:.6f}".rstrip('0').rstrip('.')
    length_str = f"{args.length:.6f}".rstrip('0').rstrip('.')
    out_filename = f"{args.prefix}{args.qid1}_{args.qid2}_lam{lam1_str}_{lam2_str}_len{length_str}_s{args.seed}_{args.rounding}.csv"
    outdir = "./schedules/" + out_filename

    rng = random.Random(args.seed)


    events = []
    events += gen_poisson_events(args.lam1, args.length, args.qid1, rng)
    events += gen_poisson_events(args.lam2, args.length, args.qid2, rng)

    # Convert to ms and sort
    out_events = []
    for t_s, qid in events:
        if args.rounding == "floor":
            at_ms = int(t_s * 1000.0)
        else:
            at_ms = int(round(t_s * 1000.0))
        out_events.append(Event(at_ms=at_ms, qid=qid))

    # Sort by timestamp; tie-break by qid for stable output
    out_events.sort(key=lambda e: (e.at_ms, e.qid))

    with open(outdir, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["at_ms", "qid"])
        for e in out_events:
            w.writerow([e.at_ms, e.qid])

    print(f"Wrote {len(out_events)} events to {outdir} (length={args.length}s, lam1={args.lam1}, lam2={args.lam2})")

if __name__ == "__main__":
    main()
