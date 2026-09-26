#!/usr/bin/env bash
# run_validation.sh -- numerical validation of loomX's code generators.
#
# Why this exists
# ---------------
# `run_tests.sh` compares process exit codes. Exit status is truncated to 8
# bits, so a checksum of 500500, 500756 and 501012 all present as exit code 20:
# the check passes for a wide band of wrong answers and cannot say how wrong.
# This script instead compiles the serial program as a golden reference, then
# compiles loomX's transformed output, runs both, and compares the full
# precision stdout with `validate_results.py`.
#
# It also probes the (mode x backend) matrix and records why a combination was
# skipped, so "we did not test CUDA" is distinguishable from "CUDA was wrong".
#
# Usage:
#   ./run_validation.sh [--threads N] [--rtol R] [--atol A] [--out DIR]
#
# Environment:
#   LOOMX    path to translator (default: ../build/loomX)
#   CC       host compiler for the golden reference (default: gcc)

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VALIDATE_DIR="$SCRIPT_DIR/validate"

THREADS="${THREADS:-8}"
RTOL="${RTOL:-1e-8}"
ATOL="${ATOL:-1e-12}"
CC="${CC:-gcc}"

# Resolve the translator robustly. The historical default in run_tests.sh
# ("../../build/loomX" from tests/) pointed outside the tree and only worked
# by accident via a /tmp fallback.
resolve_loomx() {
    if [ -n "${LOOMX:-}" ]; then
        printf '%s' "$LOOMX"
        return
    fi
    local candidate
    for candidate in "$SCRIPT_DIR/../build/loomX" \
                     "$SCRIPT_DIR/../../build/loomX" \
                     "$SCRIPT_DIR/../loomX"; do
        if [ -x "$candidate" ] && [ ! -d "$candidate" ]; then
            (cd "$(dirname "$candidate")" && printf '%s/loomX' "$(pwd)")
            return
        fi
    done
    printf '%s' "$SCRIPT_DIR/../build/loomX"
}

while [ $# -gt 0 ]; do
    case "$1" in
        --threads) THREADS="$2"; shift 2 ;;
        --rtol)    RTOL="$2";    shift 2 ;;
        --atol)    ATOL="$2";    shift 2 ;;
        --out)     OUT="$2";     shift 2 ;;
        -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

LOOMX_BIN="$(resolve_loomx)"
OUT="${OUT:-$SCRIPT_DIR/validate-out}"

if [ ! -x "$LOOMX_BIN" ]; then
    echo "ERROR: loomX not found or not executable: $LOOMX_BIN" >&2
    echo "       build it first, or set LOOMX=/path/to/loomX" >&2
    exit 1
fi

mkdir -p "$OUT"
SUMMARY="$OUT/summary.csv"
printf 'kernel,mode,backend,status,detail\n' > "$SUMMARY"

echo "== loomX numerical validation =="
echo "translator: $LOOMX_BIN"
echo "compiler:   $CC"
echo "threads:    $THREADS"
echo "tolerance:  rtol=$RTOL atol=$ATOL"
echo "workdir:    $OUT"
echo

# ---------------------------------------------------------------------------
# Toolchain probes. A missing toolchain must be reported as SKIP with a
# reason, never silently folded into a pass.
# ---------------------------------------------------------------------------
# The offload toolchain is configured in the benchmark harness, not on PATH.
# `command -v clang` returning nothing proves nothing here: on this machine
# clang-15 lives in /usr/lib/llvm-15/bin and nvcc in /usr/local/cuda-12.4/bin,
# neither of which is on the default PATH. Probe the configured paths instead.
BENCH_CONFIG="${BENCH_CONFIG:-$SCRIPT_DIR/../../../loomX-benchmarks/config.env}"
GPU_CC="${COMPILER_GPU:-}"
NVCC_BIN="${NVCC:-}"
GPU_ARCH="${GPU_ARCH:-sm_86}"
if [ -f "$BENCH_CONFIG" ]; then
    # config.env uses shell expansion (${COMPILER_GPU:-...}), so source it in a
    # subshell and read the values back rather than pattern-matching the file.
    _cfg="$(bash -c 'set -a; . "$1" >/dev/null 2>&1; \
        printf "%s\n%s\n%s" "${COMPILER_GPU:-}" "${CUDA_PATH:-}" "${GPU_ARCH:-}"' \
        _ "$BENCH_CONFIG" 2>/dev/null)"
    _cfg_gpu="$(printf '%s\n' "$_cfg" | sed -n 1p)"
    _cfg_cuda="$(printf '%s\n' "$_cfg" | sed -n 2p)"
    _cfg_arch="$(printf '%s\n' "$_cfg" | sed -n 3p)"
    GPU_CC="${GPU_CC:-$_cfg_gpu}"
    [ -z "$GPU_ARCH" ] && GPU_ARCH="$_cfg_arch"
    if [ -z "$NVCC_BIN" ] && [ -n "$_cfg_cuda" ] && [ -x "$_cfg_cuda/bin/nvcc" ]; then
        NVCC_BIN="$_cfg_cuda/bin/nvcc"
    fi
fi
[ -z "$GPU_CC" ] && [ -x /usr/lib/llvm-15/bin/clang ] && GPU_CC=/usr/lib/llvm-15/bin/clang
[ -z "$NVCC_BIN" ] && [ -x /usr/local/cuda-12.4/bin/nvcc ] && NVCC_BIN=/usr/local/cuda-12.4/bin/nvcc

probe_nvcc()   { [ -n "$NVCC_BIN" ] && [ -x "$NVCC_BIN" ]; }
probe_openacc(){ "$CC" -fopenacc -x c -c /dev/null -o /dev/null >/dev/null 2>&1; }
probe_openmp() { "$CC" -fopenmp  -x c -c /dev/null -o /dev/null >/dev/null 2>&1; }
probe_opencl_host() { ls /usr/lib/x86_64-linux-gnu/libOpenCL.so.1 >/dev/null 2>&1 \
                   || ls /usr/lib/libOpenCL.so.1 >/dev/null 2>&1; }

# Real GPU offload, probed by compiling AND running a tiny reduction. A
# present-but-broken offload setup would otherwise look like a pass.
GPU_OFFLOAD_FLAGS=""
probe_gpu_offload() {
    [ -n "$GPU_CC" ] && [ -x "$GPU_CC" ] || return 1
    local td="$OUT/.gputest.$$"
    mkdir -p "$td"
    printf '%s\n' \
        '#include <stdio.h>' \
        'int main(void){ double t=0.0;' \
        '#pragma omp target teams distribute parallel for reduction(+:t)' \
        '  for(int i=0;i<1000;i++) t += 1.0;' \
        '  printf("%.0f\n", t); return 0; }' > "$td/t.c"
    if "$GPU_CC" -O2 -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda \
            -Xopenmp-target -march="$GPU_ARCH" "$td/t.c" -o "$td/t" \
            > "$td/log" 2>&1 && [ -x "$td/t" ]; then
        local got
        got="$(timeout 180 "$td/t" 2>/dev/null | tr -d '[:space:]')"
        rm -rf "$td"
        [ "$got" = "1000" ] || return 1
        GPU_OFFLOAD_FLAGS="-O2 -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target -march=$GPU_ARCH"
        return 0
    fi
    rm -rf "$td"
    return 1
}

# Which compile command, and how deep a check the pair can support.
# Output is "<flags>|<level>" where level is one of:
#   numeric  compile, run, and compare values against the golden reference
#   compile  compile only -- the toolchain accepts the dialect but cannot
#            execute it, so no numerical claim may be made
#   skip:<reason>  toolchain absent
#
# The compile-only level exists because gcc's -fopenacc cannot actually run an
# OpenACC loop: it aborts with "libgomp: target function wasn't mapped". That
# was confirmed to be a gcc limitation rather than a loomX defect, by
# compiling a hand-written #pragma acc parallel loop that fails identically.
# So OpenACC rows are validated structurally and for syntax, never numerically.
compile_for() {
    local mode="$1" backend="$2"
    case "$backend" in
        omp)
            probe_openmp || { echo "|skip:no-openmp-flag"; return; }
            case "$mode" in
                cpu-only|cpu-forced)
                    # Plain `#pragma omp parallel for` on the host: fully
                    # executable, so this is a real numerical check.
                    echo "-O2 -fopenmp|numeric" ;;
                gpu-naive|gpu-profitable)
                    # These emit `#pragma omp target ...`, which needs a real
                    # offload toolchain to execute at all. With
                    # clang+libomptarget present this becomes a genuine
                    # numerical check running on the GPU. Otherwise fall back
                    # to gcc, which accepts the directives but does not
                    # implement the target data environment (it never copies
                    # values back out), so only syntax and pragma shape can
                    # honestly be claimed there.
                    if probe_gpu_offload; then
                        echo "GPU_CC|numeric"
                    else
                        echo "-O2 -fopenmp|compile"
                    fi ;;
                *)
                    echo "|skip:unknown-mode" ;;
            esac ;;
        openacc)
            probe_openacc || { echo "|skip:no-openacc-flag"; return; }
            echo "-O2 -fopenacc|compile" ;;
        cuda)
            probe_nvcc || { echo "|skip:no-nvcc"; return; }
            echo "NVCC|numeric" ;;
        opencl)
            probe_opencl_host || { echo "|skip:no-libOpenCL"; return; }
            echo "|skip:no-opencl-c-compiler" ;;
    esac
}

MODE_BACKENDS=(
    "cpu-only:omp"
    "cpu-forced:omp"
    "cpu-only:openacc"
    "cpu-forced:openacc"
    "gpu-naive:openacc"
    "gpu-naive:omp"
    "gpu-naive:cuda"
    "gpu-naive:opencl"
    "gpu-profitable:cuda"
    "gpu-profitable:opencl"
)

pass=0; fail=0; skip=0; compile_only=0
KERNELS=()
for f in "$VALIDATE_DIR"/*.c; do
    KERNELS+=("$(basename "$f" .c)")
done

printf '%-18s %-14s %-8s %s\n' KERNEL MODE BACKEND RESULT
printf '%-18s %-14s %-8s %s\n' "------------------" "--------------" "--------" "------"

for k in "${KERNELS[@]}"; do
    src="$VALIDATE_DIR/$k.c"

    # Golden reference: the untouched program, built with the plain compiler.
    if ! "$CC" -O2 "$src" -o "$OUT/$k.golden" -lm 2>"$OUT/$k.golden.log"; then
        echo "ERROR: golden build failed for $k" >&2
        sed -n 1,3p "$OUT/$k.golden.log" >&2
        exit 1
    fi
    "$OUT/$k.golden" > "$OUT/$k.golden.out" 2>/dev/null

    for combo in "${MODE_BACKENDS[@]}"; do
        mode="${combo%%:*}"
        backend="${combo##*:}"

        plan="$(compile_for "$mode" "$backend")"
        level="${plan##*|}"
        flags="${plan%|*}"
        if [ "$level" != "numeric" ] && [ "$level" != "compile" ]; then
            printf '%-18s %-14s %-8s %s\n' "$k" "$mode" "$backend" "${level#skip:}"
            printf '%s,%s,%s,skip,%s\n' "$k" "$mode" "$backend" "${level#skip:}" >> "$SUMMARY"
            skip=$((skip+1))
            continue
        fi

        tsrc="$OUT/$k.$mode.$backend.c"
        # The matrix stores bare mode names; loomX wants the dashed flag.
        # Passing the bare name makes loomX treat it as a positional argument
        # and silently fall back to the default mode, which would make every
        # row below look like a pass for the wrong configuration.
        case "$mode" in
            cpu-only)      mode_flag="--cpu-only" ;;
            cpu-forced)    mode_flag="--cpu-forced" ;;
            gpu-naive)     mode_flag="--gpu-naive" ;;
            gpu-profitable) mode_flag="--gpu-profitable" ;;
            *) echo "internal error: unknown mode '$mode'" >&2; exit 2 ;;
        esac
        if ! "$LOOMX_BIN" "$mode_flag" --backend "$backend" -rose:skipfinalCompileStep \
                "$src" -o "$tsrc" > "$OUT/$k.$mode.$backend.log" 2>&1; then
            printf '%-18s %-14s %-8s %s\n' "$k" "$mode" "$backend" "FAIL-transform"
            printf '%s,%s,%s,fail,transform-failed\n' "$k" "$mode" "$backend" >> "$SUMMARY"
            fail=$((fail+1))
            continue
        fi

        # Structural check runs for every row: a pragma on the wrong statement
        # still compiles and still computes the right answer, so it is
        # invisible to the numerical check.
        if ! shape="$(python3 "$SCRIPT_DIR/check_pragma_shape.py" "$tsrc" \
                        --backend "$backend" 2>&1)"; then
            printf '%-18s %-14s %-8s %s\n' "$k" "$mode" "$backend" "FAIL-shape"
            echo "$shape" | sed 's/^/      /'
            printf '%s,%s,%s,fail,pragma-shape\n' "$k" "$mode" "$backend" >> "$SUMMARY"
            fail=$((fail+1))
            continue
        fi

        bin="$OUT/$k.$mode.$backend.bin"
        if [ "$flags" = "GPU_CC" ]; then
            # shellcheck disable=SC2086
            "$GPU_CC" $GPU_OFFLOAD_FLAGS "$tsrc" -o "$bin" -lm \
                > "$OUT/$k.$mode.$backend.cc.log" 2>&1
        elif [ "$flags" = "NVCC" ]; then
            # -x cu is required: loomX emits a .c file, but it contains CUDA C++
            # (__global__, <<<>>>, atomicAdd) so nvcc would otherwise treat it
            # as plain C. -arch matters too: nvcc defaults to sm_52, where
            # atomicAdd(double*, double) does not exist, so reductions fail to
            # compile for a reason that has nothing to do with loomX.
            "$NVCC_BIN" -x cu -O2 -arch="$GPU_ARCH" "$tsrc" -o "$bin" -lm \
                > "$OUT/$k.$mode.$backend.cc.log" 2>&1
        else
            # shellcheck disable=SC2086
            "$CC" $flags "$tsrc" -o "$bin" -lm > "$OUT/$k.$mode.$backend.cc.log" 2>&1
        fi
        if [ ! -x "$bin" ]; then
            printf '%-18s %-14s %-8s %s\n' "$k" "$mode" "$backend" "FAIL-compile"
            printf '%s,%s,%s,fail,compile-failed\n' "$k" "$mode" "$backend" >> "$SUMMARY"
            fail=$((fail+1))
            continue
        fi

        if [ "$level" = "compile" ]; then
            # Compiles and is structurally sound, but this toolchain cannot
            # execute the dialect, so no numerical verdict is claimed.
            npragmas="$(printf '%s' "$shape" | sed -n 's/.*pragmas=\([0-9]*\).*/\1/p')"
            printf '%-18s %-14s %-8s %s\n' "$k" "$mode" "$backend" \
                   "COMPILE-ONLY (pragmas=${npragmas:-0})"
            printf '%s,%s,%s,compile-only,not-executable-by-toolchain\n' \
                   "$k" "$mode" "$backend" >> "$SUMMARY"
            compile_only=$((compile_only+1))
            continue
        fi

        OMP_NUM_THREADS="$THREADS" "$bin" > "$OUT/$k.$mode.$backend.out" 2>/dev/null
        rc=$?
        if [ "$rc" -ne 0 ]; then
            printf '%-18s %-14s %-8s %s\n' "$k" "$mode" "$backend" "FAIL-runtime(rc=$rc)"
            printf '%s,%s,%s,fail,runtime-rc-%s\n' "$k" "$mode" "$backend" "$rc" >> "$SUMMARY"
            fail=$((fail+1))
            continue
        fi

        if out="$(python3 "$SCRIPT_DIR/validate_results.py" \
                    "$OUT/$k.golden.out" "$OUT/$k.$mode.$backend.out" \
                    --rtol "$RTOL" --atol "$ATOL" \
                    --kernel "$k" --backend "$backend/$mode" 2>&1)"; then
            printf '%-18s %-14s %-8s %s\n' "$k" "$mode" "$backend" "PASS"
            printf '%s,%s,%s,pass,ok\n' "$k" "$mode" "$backend" >> "$SUMMARY"
            pass=$((pass+1))
        else
            printf '%-18s %-14s %-8s %s\n' "$k" "$mode" "$backend" "FAIL-mismatch"
            echo "$out" | sed 's/^/      /'
            printf '%s,%s,%s,fail,numerical-mismatch\n' "$k" "$mode" "$backend" >> "$SUMMARY"
            fail=$((fail+1))
        fi
    done
done

echo
echo "== Summary =="
echo "numerical pass: $pass   fail: $fail   compile-only: $compile_only   skipped: $skip"
echo "matrix: $SUMMARY"
echo
echo "Notes:"
echo "  * Toolchain discovery reads ../loomX-benchmarks/config.env. clang and"
echo "    nvcc are NOT on the default PATH, so \`command -v clang\` returning"
echo "    nothing does not mean they are missing. Override with GPU_CC / NVCC."
echo "  * omp+gpu-* rows are a real numerical check on the GPU whenever"
echo "    probe_gpu_offload succeeds (it compiles AND runs a reduction)."
echo "  * compile-only rows are NOT numerical passes. That level is used when"
echo "    only gcc is available, because gcc -fopenacc aborts at runtime"
echo "    (\"libgomp: target function wasn't mapped\") and gcc -fopenmp does not"
echo "    write values back out of an 'omp target' data environment. Both were"
echo "    confirmed against hand-written oracles that fail identically, so"
echo "    those rows are checked for syntax and pragma shape only."
echo "  * A skip is a toolchain gap, not a pass."

[ "$fail" -eq 0 ]
