#!/usr/bin/env python3
"""
check_dataracebench.py -- evaluate loomX's parallel-safety judgments against
DataRaceBench's yes/no ground truth.

DataRaceBench files are named like:
    DRB###-name-orig-yes.c   -- contains a data race (loomX should reject)
    DRB###-name-orig-no.c    -- race-free (loomX should accept)

We run loomX on each file and count "accepted" (any OpenMP pragma inserted in
a loop body) vs "rejected".  The script reports false positives, false
negatives, and an overall accuracy score.

Usage:
    python3 check_dataracebench.py \
        --loomx /path/to/loomX \
        --suite /path/to/dataracebench/micro-benchmarks \
        --output drb_results.csv
"""
import argparse
import csv
import os
import re
import subprocess
import sys
import tempfile


def run_loomx(loomx_path, src_path, mode="cpu-only"):
    """Run loomX on src_path and return True if any loop was parallelized.
    Output is written to a temporary directory so stale .loomx.c files do not
    affect later runs or get scanned as DRB sources.
    """
    with tempfile.TemporaryDirectory() as td:
        out_path = os.path.join(td, os.path.basename(src_path) + ".loomx.c")
        try:
            subprocess.run(
                [loomx_path, f"--{mode}", src_path, "-o", out_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=120,
                check=False,
            )
        except Exception as e:
            print(f"  [warn] loomX failed on {src_path}: {e}", file=sys.stderr)
            return False

        if not os.path.exists(out_path):
            return False

        with open(out_path) as f:
            content = f.read()
        # A parallelized loop will contain an OpenMP pragma.
        return "#pragma omp" in content


def parse_label(filename):
    """Return ('yes'|'no'|None, base_name) from a DRB filename."""
    m = re.match(r"(DRB\d+-.+)-(orig|omp)-(yes|no)\.c$", filename)
    if not m:
        return None, None
    return m.group(3), m.group(1) + "-" + m.group(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loomx", required=True, help="path to loomX translator")
    ap.add_argument("--suite", required=True, help="path to dataracebench micro-benchmarks dir")
    ap.add_argument("--mode", default="cpu-only", help="loomX mode: cpu-only, gpu-naive, gpu-profitable")
    ap.add_argument("--output", default="dataracebench_results.csv")
    args = ap.parse_args()

    results = []
    accepted = {"yes": 0, "no": 0}
    rejected = {"yes": 0, "no": 0}
    false_positives = []
    false_negatives = []

    files = sorted(f for f in os.listdir(args.suite) if f.endswith(".c"))
    print(f"Scanning {len(files)} DRB sources in {args.suite} ...")

    for filename in files:
        label, base = parse_label(filename)
        if label is None:
            continue

        src = os.path.join(args.suite, filename)
        parallelized = run_loomx(args.loomx, src, args.mode)

        if parallelized:
            accepted[label] += 1
            if label == "yes":
                false_positives.append(filename)
        else:
            rejected[label] += 1
            if label == "no":
                false_negatives.append(filename)

        results.append({
            "file": filename,
            "expected": "reject" if label == "yes" else "accept",
            "actual": "accept" if parallelized else "reject",
            "correct": (label == "yes" and not parallelized) or (label == "no" and parallelized),
        })

    total = len(results)
    correct = sum(1 for r in results if r["correct"])

    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "expected", "actual", "correct"])
        w.writeheader()
        w.writerows(results)

    print(f"\nMode: {args.mode}")
    print(f"Total evaluated: {total}")
    print(f"Correct:         {correct} ({100.0*correct/total:.1f}%)" if total else "N/A")
    print(f"False positives (accepted a -yes race): {len(false_positives)}")
    print(f"False negatives (rejected a -no safe):  {len(false_negatives)}")
    print(f"\nAccepted -no:  {accepted['no']}  Rejected -no:  {rejected['no']}")
    print(f"Accepted -yes: {accepted['yes']}  Rejected -yes: {rejected['yes']}")

    if false_positives:
        print("\nFalse positives:")
        for f in false_positives:
            print(f"  {f}")
    if false_negatives:
        print("\nFalse negatives:")
        for f in false_negatives[:20]:  # truncate long lists
            print(f"  {f}")
        if len(false_negatives) > 20:
            print(f"  ... and {len(false_negatives) - 20} more")

    print(f"\nDetailed results written to {args.output}")


if __name__ == "__main__":
    main()
