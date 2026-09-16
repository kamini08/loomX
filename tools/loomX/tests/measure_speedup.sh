#!/bin/bash
# Speedup measurement script for loomX
# Measures sequential vs CPU-OpenMP vs GPU performance
# Uses date +%s.%N for timing (more reliable than /usr/bin/time in subshells)

set -e

GPU_PAR="/home/kamini/projects/hackathon-rose/loomX/loomX"
NVC="/home/kamini/projects/hackathon-rose/nvhpc_2025_253_Linux_x86_64_cuda_12.8/install_components/Linux_x86_64/25.3/compilers/bin/nvc"
TESTDIR="/home/kamini/projects/hackathon-rose/loomX/tests"
export LD_LIBRARY_PATH=/home/kamini/projects/hackathon-rose/rose-install/lib:/home/kamini/projects/hackathon-rose/boost-install/lib:/usr/lib/llvm-17/lib:$LD_LIBRARY_PATH

RUNS=5

# Timing function using date +%s.%N (nanosecond precision)
measure_time() {
    local cmd="$1"
    local total=0
    
    # Warm-up run
    $cmd > /dev/null 2>&1 || true
    
    for i in $(seq 1 $RUNS); do
        local start=$(date +%s.%N)
        $cmd > /dev/null 2>&1 || true
        local end=$(date +%s.%N)
        local elapsed=$(echo "$end - $start" | bc)
        total=$(echo "$total + $elapsed" | bc)
    done
    
    local avg=$(echo "scale=4; $total / $RUNS" | bc)
    echo "$avg"
}

run_benchmark() {
    local name=$1
    local src="${TESTDIR}/${name}.c"
    local rose_src="${TESTDIR}/rose_${name}.c"
    
    echo "=== Benchmark: ${name} ==="
    
    # Ensure loomX output exists
    if [ ! -f "$rose_src" ]; then
        echo "  Generating loomX output..."
        cd "$TESTDIR"
        "$GPU_PAR" -rose:skipfinalCompileStep "${name}.c" > /dev/null 2>&1
    fi
    
    # Compile sequential
    echo "  Compiling sequential..."
    "$NVC" -o "${TESTDIR}/${name}_seq" "$src" 2>&1 | grep -v "warning:\|Remark:" || true
    
    # Compile CPU OpenMP
    echo "  Compiling CPU OpenMP..."
    "$NVC" -mp=multicore -o "${TESTDIR}/${name}_cpu" "$rose_src" 2>&1 | grep -v "warning:\|Remark:" || true
    
    # Compile GPU
    echo "  Compiling GPU..."
    "$NVC" -mp=gpu -gpu=cc120 -o "${TESTDIR}/${name}_gpu" "$rose_src" 2>&1 | grep -v "warning:\|Remark:" || true
    
    # Measure times
    echo "  Measuring sequential..."
    local t_seq=$(measure_time "${TESTDIR}/${name}_seq")
    echo "  Measuring CPU OpenMP..."
    local t_cpu=$(measure_time "${TESTDIR}/${name}_cpu")
    echo "  Measuring GPU..."
    local t_gpu=$(measure_time "${TESTDIR}/${name}_gpu")
    
    # Calculate speedups (handle t_cpu or t_gpu being 0)
    local s_cpu="N/A"
    local s_gpu="N/A"
    if [ "$(echo "$t_cpu > 0" | bc)" -eq 1 ]; then
        s_cpu=$(echo "scale=2; $t_seq / $t_cpu" | bc)
    fi
    if [ "$(echo "$t_gpu > 0" | bc)" -eq 1 ]; then
        s_gpu=$(echo "scale=2; $t_seq / $t_gpu" | bc)
    fi
    
    printf "  Sequential: %.4fs\n" "$t_seq"
    printf "  CPU OpenMP: %.4fs (%sx)\n" "$t_cpu" "$s_cpu"
    printf "  GPU OpenMP: %.4fs (%sx)\n" "$t_gpu" "$s_gpu"
    echo ""
}

echo "=== loomX Speedup Measurement ==="
echo "Averaging over ${RUNS} runs after 1 warm-up"
echo ""

run_benchmark "perf_large"
run_benchmark "perf_interproc"
run_benchmark "perf_compute"
run_benchmark "polybench_gemm"

echo "=== Measurement complete ==="
