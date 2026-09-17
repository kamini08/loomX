#!/usr/bin/env bash
# run_benchmarks.sh -- driver for the loomX benchmarking harness.
#
# Usage:
#   ./run_benchmarks.sh [suite]
#
# where suite is one of:
#   interproc-microbench   (default) -- hand-written interprocedural stress tests
#   polybench              -- a few representative PolyBench/C kernels
#   dataracebench          -- correctness pass over DataRaceBench yes/no pairs
#
# The script:
#   1. Ensures bench-suites/ has been fetched.
#   2. Generates seq/cpu_omp/gpu_naive/gpu_profitable source variants.
#   3. Compiles them.
#   4. Checks numerical correctness of parallel configs vs. seq.
#   5. Runs each config through bench_harness.py.
#   6. Aggregates results with aggregate_results.py.
#
# Override environment variables:
#   LOOMX            path to translator (default: ../loomX)
#   COMPILER         host compiler (default: clang)
#   GPU_ARCH         offload arch, e.g. sm_120 (default: auto-detected)
#   RUNS             number of timing runs (default: 10)
#   BENCH_ARGS       problem-size argument passed to binaries
#   LOCK_CLOCKS      "yes" to lock/reset GPU clocks (default: no)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Make the ROSE shared library and offload clang runtime discoverable for the
# translator and the GPU compiler. Respect any LD_LIBRARY_PATH the user already
# exported (e.g. for a custom ROSE install or libomptarget).
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-/home/kamini/projects/hackathon-rose/rose-install/lib}"

SUITE="${1:-interproc-microbench}"
LOOMX="${LOOMX:-$SCRIPT_DIR/../loomX}"
COMPILER_CPU="${COMPILER_CPU:-gcc}"
COMPILER_GPU="${COMPILER_GPU:-clang}"
RUNS="${RUNS:-10}"
BENCH_ARGS="${BENCH_ARGS:-}"
LOCK_CLOCKS="${LOCK_CLOCKS:-no}"

# Auto-detect GPU compute capability if not provided.
if [ -z "${GPU_ARCH:-}" ]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
        CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | tr -d '.')
        GPU_ARCH="sm_${CAP}"
    else
        GPU_ARCH="sm_80"
    fi
fi

echo "== loomX benchmark driver =="
echo "suite:        $SUITE"
echo "LOOMX:        $LOOMX"
echo "CPU compiler: $COMPILER_CPU"
echo "GPU compiler: $COMPILER_GPU"
echo "GPU arch:     $GPU_ARCH"
echo "runs:         $RUNS"
echo "args:         ${BENCH_ARGS:-<none>}"
echo "lock clocks:  $LOCK_CLOCKS"
echo

mkdir -p out bin results

# ---------------------------------------------------------------------------
# 1. Pick the benchmark list.
# ---------------------------------------------------------------------------
declare -a BENCHES
if [ "$SUITE" = "interproc-microbench" ]; then
    MICRO_DIR="$SCRIPT_DIR/interproc-microbench"
    if [ ! -d "$MICRO_DIR" ]; then
        echo "ERROR: $MICRO_DIR not found." >&2
        exit 1
    fi
    mapfile -t BENCHES < <(find "$MICRO_DIR" -maxdepth 1 -name '*.c' -printf '%f\n' | sed 's/\.c$//' | sort)
    BENCH_SRC_DIR="$MICRO_DIR"
elif [ "$SUITE" = "polybench" ]; then
    PB_DIR="$SCRIPT_DIR/bench-suites/polybench"
    if [ ! -d "$PB_DIR" ]; then
        echo "ERROR: $PB_DIR not found. Run ./setup_benchmarks.sh first." >&2
        exit 1
    fi
    BENCHES=(gemm syrk syr2k)
    BENCH_SRC_DIR="$PB_DIR/linear-algebra/blas"
    PB_UTILITIES_DIR="$PB_DIR/utilities"
else
    echo "ERROR: unknown suite '$SUITE'" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# 2. Generate variants.
# ---------------------------------------------------------------------------
generate_one() {
    local name="$1"
    local src="$2"
    mkdir -p "out/$name"

    cp "$src" "out/$name/${name}__seq.c"

    local loomx_args=()
    if [ "$SUITE" = "polybench" ]; then
        loomx_args+=(-I"$PB_UTILITIES_DIR" -I"$BENCH_SRC_DIR/$name")
    fi

    "$LOOMX" --cpu-only "${loomx_args[@]}" "$src" -o "out/$name/${name}__cpu_omp.c" >/dev/null 2>&1 || {
        echo "  WARN: loomX --cpu-only failed for $name"
        return 1
    }
    "$LOOMX" --gpu-naive "${loomx_args[@]}" "$src" -o "out/$name/${name}__gpu_naive.c" >/dev/null 2>&1 || {
        echo "  WARN: loomX --gpu-naive failed for $name"
        return 1
    }
    "$LOOMX" --gpu-profitable "${loomx_args[@]}" "$src" -o "out/$name/${name}__gpu_profitable.c" >/dev/null 2>&1 || {
        echo "  WARN: loomX --gpu-profitable failed for $name"
        return 1
    }
    return 0
}

echo "== Generating source variants =="
for name in "${BENCHES[@]}"; do
    if [ "$SUITE" = "polybench" ]; then
        src="$BENCH_SRC_DIR/$name/${name}.c"
    else
        src="$BENCH_SRC_DIR/${name}.c"
    fi
    echo "  $name"
    generate_one "$name" "$src" || true
done

# ---------------------------------------------------------------------------
# 3. Compile variants.
# ---------------------------------------------------------------------------
compile_one() {
    local name="$1"
    local cfg="$2"
    local src="out/$name/${name}__${cfg}.c"
    local bin="bin/${name}__${cfg}"
    [ -f "$src" ] || return 1

    local extra_flags=(-lm)
    if [ "$SUITE" = "polybench" ]; then
        # Kernel-specific header (e.g. gemm.h) lives next to the .c source.
        extra_flags+=(-DPOLYBENCH_TIME -DLARGE_DATASET -I"$PB_UTILITIES_DIR" -I"$BENCH_SRC_DIR/$name")
        # PolyBench kernels need utilities/polybench.c linked in.
        extra_flags+=("$PB_UTILITIES_DIR/polybench.c")
    elif [ "$SUITE" = "interproc-microbench" ]; then
        extra_flags+=(-I"$BENCH_SRC_DIR")
    fi

    case "$cfg" in
        seq)
            "$COMPILER_CPU" -O3 "$src" "${extra_flags[@]}" -o "$bin"
            ;;
        cpu_omp)
            "$COMPILER_CPU" -O3 -fopenmp "$src" "${extra_flags[@]}" -o "$bin"
            ;;
        gpu_naive|gpu_profitable)
            # Offload clang needs the OpenMP runtime headers (omp.h, omp-tools.h).
            local omp_include="$(dirname "$COMPILER_GPU")/../projects/openmp/runtime/src"
            [ -f "$omp_include/omp.h" ] || omp_include="$(dirname "$COMPILER_GPU")/../include"
            "$COMPILER_GPU" -O3 -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda \
                -Xopenmp-target -march="$GPU_ARCH" \
                -I"$omp_include" \
                "$src" "${extra_flags[@]}" -o "$bin"
            ;;
    esac
}

echo "== Compiling =="
for name in "${BENCHES[@]}"; do
    for cfg in seq cpu_omp gpu_naive gpu_profitable; do
        if compile_one "$name" "$cfg"; then
            echo "  ok  $name/$cfg"
        else
            echo "  skip $name/$cfg (source or compile failed)"
        fi
    done
done

# ---------------------------------------------------------------------------
# 4. Numerical correctness: each non-seq config vs. seq golden.
# ---------------------------------------------------------------------------
RESULTS_CSV="results/${SUITE}.csv"
rm -f "$RESULTS_CSV"

echo "== Correctness checks =="
for name in "${BENCHES[@]}"; do
    seq_bin="bin/${name}__seq"
    [ -x "$seq_bin" ] || continue

    golden="results/${name}__golden.out"
    ./"$seq_bin" $BENCH_ARGS > "$golden" 2>/dev/null || true

    for cfg in cpu_omp gpu_naive gpu_profitable; do
        cand="bin/${name}__${cfg}"
        [ -x "$cand" ] || continue
        cand_out="results/${name}__${cfg}.out"
        ./"$cand" $BENCH_ARGS > "$cand_out" 2>/dev/null || true
        if python3 check_correctness.py "$golden" "$cand_out" --rtol 1e-5 --atol 1e-8; then
            echo "  PASS $name/$cfg"
        else
            echo "  FAIL $name/$cfg"
        fi
    done
done

# ---------------------------------------------------------------------------
# 5. Timing harness.
# ---------------------------------------------------------------------------
[ "$LOCK_CLOCKS" = "yes" ] && ./lock_gpu_clocks.sh lock

echo "== Timing ($RUNS runs each) =="
for name in "${BENCHES[@]}"; do
    for cfg in seq cpu_omp gpu_naive gpu_profitable; do
        bin="bin/${name}__${cfg}"
        [ -x "$bin" ] || continue
        extra=""
        [[ "$cfg" == gpu_* ]] && extra="--nsys"
        python3 bench_harness.py \
            --binary "$bin" --args "$BENCH_ARGS" --runs "$RUNS" \
            --label "${name}__${cfg}" --out "$RESULTS_CSV" $extra || true
    done
done

[ "$LOCK_CLOCKS" = "yes" ] && ./lock_gpu_clocks.sh reset

# ---------------------------------------------------------------------------
# 6. Aggregate.
# ---------------------------------------------------------------------------
echo "== Aggregation =="
if [ -s "$RESULTS_CSV" ]; then
    python3 aggregate_results.py "$RESULTS_CSV" --baseline seq
else
    echo "No results collected."
fi

echo
echo "Raw results: $RESULTS_CSV"
