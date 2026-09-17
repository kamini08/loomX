#!/usr/bin/env python3
"""
bench_harness.py -- run a benchmark binary N times, record wall-clock time,
optionally capture an Nsight Systems host<->device transfer/kernel time
breakdown, and append results to a CSV.

Label convention (required by aggregate_results.py):
    <benchmark>__<config>
e.g. gemm__seq, gemm__cpu_omp, gemm__gpu_naive, gemm__gpu_profitable

Examples:
  python3 bench_harness.py --binary bin/gemm__seq --args "2048" \
      --runs 10 --label gemm__seq --out results.csv

  # GPU configs: also capture H2D/D2H/kernel time (needs `nsys` in PATH)
  python3 bench_harness.py --binary bin/gemm__gpu_profitable --args "2048" \
      --runs 10 --label gemm__gpu_profitable --out results.csv --nsys
"""
import argparse
import csv
import os
import shlex
import sqlite3
import statistics
import subprocess
import sys
import tempfile
import time


def run_once(binary, args):
    t0 = time.perf_counter()
    proc = subprocess.run([binary, *args], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    t1 = time.perf_counter()
    if proc.returncode != 0:
        raise RuntimeError(f"{binary} exited {proc.returncode}: {proc.stderr.decode()[:400]}")
    return t1 - t0


def run_with_nsys(binary, args, tag):
    """Wall time via wall clock, GPU breakdown via Nsight Systems' sqlite export.

    NOTE: CUPTI table/column names have shifted across nsys versions. If the
    query below returns zeros, run:
        nsys export -t sqlite -o /tmp/probe report.nsys-rep
        sqlite3 /tmp/probe.sqlite ".tables"
    and adjust the table names (commonly CUPTI_ACTIVITY_KIND_MEMCPY /
    CUPTI_ACTIVITY_KIND_KERNEL) to match what your installed version emits.
    """
    with tempfile.TemporaryDirectory() as td:
        report = os.path.join(td, tag)
        t0 = time.perf_counter()
        subprocess.run(
            ["nsys", "profile", "-o", report, "--force-overwrite=true", "--stats=false",
             binary, *args],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        wall = time.perf_counter() - t0

        sqlite_path = report + ".sqlite"
        subprocess.run(
            ["nsys", "export", "-t", "sqlite", "-o", sqlite_path, report + ".nsys-rep"],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )

        h2d = d2h = kernel = 0.0
        try:
            con = sqlite3.connect(sqlite_path)
            cur = con.cursor()
            cur.execute(
                "SELECT copyKind, SUM(end-start) FROM CUPTI_ACTIVITY_KIND_MEMCPY GROUP BY copyKind"
            )
            for kind, ns in cur.fetchall():
                if kind == 1:      # CUPTI memcpy kind: 1 = host-to-device
                    h2d = ns / 1e9
                elif kind == 2:    # 2 = device-to-host
                    d2h = ns / 1e9
            cur.execute("SELECT SUM(end-start) FROM CUPTI_ACTIVITY_KIND_KERNEL")
            row = cur.fetchone()
            kernel = (row[0] or 0) / 1e9
        except Exception as e:
            print(f"  [warn] nsys sqlite breakdown unavailable ({e}); "
                  f"see NOTE in run_with_nsys() docstring", file=sys.stderr)
        return wall, h2d, d2h, kernel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", required=True)
    ap.add_argument("--args", default="")
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--label", required=True, help="format: <benchmark>__<config>")
    ap.add_argument("--out", default="results.csv")
    ap.add_argument("--nsys", action="store_true",
                     help="capture H2D/D2H/kernel breakdown via Nsight Systems")
    args = ap.parse_args()

    argv = shlex.split(args.args)
    times, h2ds, d2hs, kerns = [], [], [], []

    for i in range(args.runs):
        if args.nsys:
            wall, h2d, d2h, kern = run_with_nsys(args.binary, argv, f"{args.label}_{i}")
            h2ds.append(h2d)
            d2hs.append(d2h)
            kerns.append(kern)
        else:
            wall = run_once(args.binary, argv)
        times.append(wall)
        print(f"  run {i + 1}/{args.runs}: {wall:.4f}s", file=sys.stderr)

    median = statistics.median(times)
    stdev = statistics.stdev(times) if len(times) > 1 else 0.0

    new_file = not os.path.exists(args.out)
    with open(args.out, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["label", "binary", "args", "runs", "median_s", "stdev_s",
                        "median_h2d_s", "median_d2h_s", "median_kernel_s"])
        w.writerow([
            args.label, args.binary, args.args, args.runs,
            f"{median:.6f}", f"{stdev:.6f}",
            f"{statistics.median(h2ds):.6f}" if h2ds else "",
            f"{statistics.median(d2hs):.6f}" if d2hs else "",
            f"{statistics.median(kerns):.6f}" if kerns else "",
        ])
    print(f"[{args.label}] median={median:.4f}s stdev={stdev:.4f}s -> appended to {args.out}")


if __name__ == "__main__":
    main()
