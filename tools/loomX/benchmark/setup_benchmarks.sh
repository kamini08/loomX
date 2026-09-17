#!/usr/bin/env bash
# Fetches the three benchmark suites used for loomX validation:
#   - PolyBench/C 4.2.1  -> GPU-profitability demonstration (regular, affine kernels)
#   - Rodinia            -> realistic apps, exposes host<->device transfer costs
#   - DataRaceBench       -> correctness ground truth (LLNL, paired yes/no race examples)
#
# Usage: ./setup_benchmarks.sh [target_dir]

set -euo pipefail
WORKDIR="${1:-./bench-suites}"
mkdir -p "$WORKDIR"
cd "$WORKDIR"

echo "== PolyBench/C 4.2.1 (Pouchet/Yuki, Ohio State mirror) =="
if [ ! -d polybench ]; then
  git clone --depth 1 https://github.com/MatthiasJReisinger/PolyBenchC-4.2.1.git polybench
else
  echo "  already present, skipping"
fi

echo "== Rodinia (OpenMP + CUDA source side by side) =="
if [ ! -d rodinia ]; then
  git clone --depth 1 https://github.com/yuhc/gpu-rodinia.git rodinia || \
  git clone --depth 1 https://github.com/ouankou/rodinia.git rodinia   # actively-maintained CMake fork, fallback
else
  echo "  already present, skipping"
fi

echo "== DataRaceBench (LLNL correctness ground truth, DRBxxx-*-orig-{yes,no}.c) =="
if [ ! -d dataracebench ]; then
  git clone --depth 1 https://github.com/LLNL/dataracebench.git dataracebench
else
  echo "  already present, skipping"
fi

cat <<'EOF'

Done. Layout:
  bench-suites/polybench/      -- linear-algebra/, stencils/, datamining/ kernels
  bench-suites/rodinia/        -- openmp/ and cuda/ variants per app
  bench-suites/dataracebench/  -- micro-benchmarks/DRB###-*-orig-{yes,no}.c

Notes:
- You still need to write your own "function call inside a hot loop"
  benchmarks (PolyBench/Rodinia mostly don't have these on purpose --
  see interproc-microbench/ once you add it). Put those alongside these
  three under a fourth directory, e.g. bench-suites/interproc-microbench/.
- DataRaceBench's P-labels (Y1..Y7 has-a-race, N1..N7 race-free) are your
  off-the-shelf ground truth for "did loomX correctly judge this loop
  parallel-safe" -- no need to hand-verify safety yourselves for these.
EOF
