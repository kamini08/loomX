#!/usr/bin/env python3
"""
aggregate_results.py -- turn raw bench_harness.py CSV rows into the numbers
that actually go on a slide: per-benchmark speedup vs. sequential for each
config, and the geometric mean across the whole benchmark set per config.

Expects labels formatted as '<benchmark>__<config>', e.g.:
  gemm__seq, gemm__cpu_omp, gemm__gpu_naive, gemm__gpu_profitable

Usage:
  python3 aggregate_results.py results.csv --baseline seq
"""
import argparse
import csv
import math
from collections import defaultdict


def geomean(xs):
    xs = [x for x in xs if x > 0]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path")
    ap.add_argument("--baseline", default="seq")
    args = ap.parse_args()

    data = defaultdict(dict)  # bench -> config -> median_s
    with open(args.csv_path) as f:
        for row in csv.DictReader(f):
            bench, _, config = row["label"].partition("__")
            data[bench][config] = float(row["median_s"])

    configs = sorted({c for d in data.values() for c in d} - {args.baseline})
    speedups = defaultdict(list)

    header = ["benchmark", f"{args.baseline}_s"] + [f"{c}_speedup" for c in configs]
    print(",".join(header))
    for bench, d in sorted(data.items()):
        if args.baseline not in d:
            continue
        base = d[args.baseline]
        row = [bench, f"{base:.4f}"]
        for c in configs:
            if c in d and d[c] > 0:
                sp = base / d[c]
                speedups[c].append(sp)
                row.append(f"{sp:.3f}")
            else:
                row.append("")
        print(",".join(row))

    print()
    print("== Geometric mean speedup vs. baseline across benchmarks ==")
    for c in configs:
        print(f"  {c}: {geomean(speedups[c]):.3f}x   (n={len(speedups[c])})")


if __name__ == "__main__":
    main()
