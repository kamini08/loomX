import argparse
import csv
import os
import re
import subprocess
import sys
import tempfile


def find_hot_loop_lines(src_path):
    """Return line numbers of for-loops originally preceded by #pragma omp."""
    hot_lines = []
    with open(src_path) as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.lstrip().startswith("#pragma omp"):
            # Scan forward for the associated for-statement.
            j = i + 1
            while j < len(lines):
                cur = lines[j].strip()
                if not cur or cur.startswith("//"):
                    j += 1
                    continue
                if cur.startswith("for ") or cur.startswith("for("):
                    hot_lines.append(j + 1)  # 1-based line number
                    break
                break
            i = j + 1 if j < len(lines) else i + 1
        else:
            i += 1
    return hot_lines


def strip_existing_pragmas(src_path):
    with open(src_path) as f:
        lines = f.readlines()
    stripped = [line for line in lines if not line.lstrip().startswith("#pragma omp")]
    tmp_path = src_path + ".stripped.c"
    with open(tmp_path, "w") as f:
        f.writelines(stripped)
    return tmp_path


def run_loomx(loomx_path, src_path, mode="cpu-only", strict=False):
    input_path = src_path
    extra_args = []
    if strict:
        input_path = strip_existing_pragmas(src_path)
        extra_args.append("--analyze-only")
    else:
        extra_args.append("--no-scalar-dep-check")

    try:
        with tempfile.TemporaryDirectory() as td:
            out_path = os.path.join(td, os.path.basename(src_path) + ".loomx.c")
            try:
                proc = subprocess.run(
                    [loomx_path, f"--{mode}"] + extra_args + [input_path, "-o", out_path],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=120,
                    check=False,
                )
            except Exception as e:
                print(f"  [warn] loomX failed on {src_path}: {e}", file=sys.stderr)
                return False

            if strict:
                # Strict mode: accept only if a loop originally marked parallel is
                # now parallelized. This avoids counting init/checksum loops.
                hot_lines = set(find_hot_loop_lines(src_path))
                if not hot_lines:
                    return False
                stdout = proc.stdout.decode("utf-8", errors="ignore")
                parallelized_lines = set()
                for line in stdout.splitlines():
                    m = re.match(r"PARALLELIZED line (\d+)", line.strip())
                    if m:
                        parallelized_lines.add(int(m.group(1)))
                return bool(hot_lines & parallelized_lines)
            else:
                if not os.path.exists(out_path):
                    return False
                with open(out_path) as f:
                    content = f.read()
                return "#pragma omp" in content
    finally:
        if strict:
            try:
                os.remove(input_path)
            except FileNotFoundError:
                pass


def parse_label(filename):
    m = re.match(r"(DRB\d+-.+)-(orig|omp)-(yes|no)\.c$", filename)
    if not m:
        return None, None
    return m.group(3), m.group(1) + "-" + m.group(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loomx", required=True)
    ap.add_argument("--suite", required=True)
    ap.add_argument("--mode", default="cpu-only")
    ap.add_argument("--strict", action="store_true")
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
        parallelized = run_loomx(args.loomx, src, args.mode, strict=args.strict)

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

    print(f"\nMode: {args.mode} (strict={args.strict})")
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
        for f in false_negatives[:20]:
            print(f"  {f}")
        if len(false_negatives) > 20:
            print(f"  ... and {len(false_negatives) - 20} more")

    print(f"\nDetailed results written to {args.output}")


if __name__ == "__main__":
    main()
