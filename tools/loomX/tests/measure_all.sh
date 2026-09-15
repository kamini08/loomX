#!/bin/bash
# Comprehensive speedup measurement for loomX
# Measures sequential, CPU-OpenMP, and GPU-OpenMP across benchmarks

set -e

GPU_PAR="/home/kamini/projects/hackathon-rose/loomX/loomX"
NVC="/home/kamini/projects/hackathon-rose/nvhpc_2025_253_Linux_x86_64_cuda_12.8/install_components/Linux_x86_64/25.3/compilers/bin/nvc"
TESTDIR="/home/kamini/projects/hackathon-rose/loomX/tests"
export LD_LIBRARY_PATH=/home/kamini/projects/hackathon-rose/rose-install/lib:/home/kamini/projects/hackathon-rose/boost-install/lib:/usr/lib/llvm-17/lib:$LD_LIBRARY_PATH

RUNS=5

measure_time() {
    local cmd="$1"
    local total=0
    
    $cmd > /dev/null 2>&1 || true
    
    for i in $(seq 1 $RUNS); do
        local start=$(date +%s.%N)
        $cmd > /dev/null 2>&1 || true
        local end=$(date +%s.%N)
        local elapsed=$(echo "$end - $start" | bc)
        total=$(echo "$total + $elapsed" | bc)
    done
    
    echo "scale=4; $total / $RUNS" | bc
}

run_benchmark() {
    local name=$1
    local src="${TESTDIR}/${name}.c"
    local rose_src="${TESTDIR}/rose_${name}.c"
    
    # Ensure rose output exists
    if [ ! -f "$rose_src" ]; then
        cd "$TESTDIR"
        "$GPU_PAR" -rose:skipfinalCompileStep "${name}.c" > /dev/null 2>&1
    fi
    
    # Compile all variants
    "$NVC" -o "${TESTDIR}/${name}_seq" "$src" 2>&1 | grep -v "warning:\|Remark:" || true
    "$NVC" -mp=multicore -o "${TESTDIR}/${name}_cpu" "$rose_src" 2>&1 | grep -v "warning:\|Remark:" || true
    "$NVC" -mp=gpu -gpu=cc120 -o "${TESTDIR}/${name}_gpu" "$rose_src" 2>&1 | grep -v "warning:\|Remark:" || true
    
    local t_seq=$(measure_time "${TESTDIR}/${name}_seq")
    local t_cpu=$(measure_time "${TESTDIR}/${name}_cpu")
    local t_gpu=$(measure_time "${TESTDIR}/${name}_gpu")
    
    local s_cpu="N/A"
    local s_gpu="N/A"
    if [ "$(echo "$t_cpu > 0" | bc)" -eq 1 ]; then
        s_cpu=$(echo "scale=2; $t_seq / $t_cpu" | bc)
    fi
    if [ "$(echo "$t_gpu > 0" | bc)" -eq 1 ]; then
        s_gpu=$(echo "scale=2; $t_seq / $t_gpu" | bc)
    fi
    
    printf "%-20s %10ss %10ss %6sx %10ss %6sx\n" "$name" "$t_seq" "$t_cpu" "$s_cpu" "$t_gpu" "$s_gpu"
}

echo "=== loomX Speedup Measurement ==="
echo "Averaging over ${RUNS} runs after 1 warm-up"
echo ""
printf "%-20s %10s %10s %6s %10s %6s\n" "Benchmark" "Seq(s)" "CPU(s)" "CPUx" "GPU(s)" "GPUx"
printf "%-20s %10s %10s %6s %10s %6s\n" "--------" "------" "------" "----" "------" "----"

run_benchmark "perf_large"
run_benchmark "perf_interproc"
run_benchmark "perf_compute"
run_benchmark "perf_heavy"
run_benchmark "perf_heavy5m"
run_benchmark "perf_heavy10m"

echo ""
echo "=== Notes ==="
echo "- CPU OpenMP: -mp=multicore (all available CPU cores)"
echo "- GPU OpenMP: -mp=gpu -gpu=cc120 (RTX 5060 Mobile, sm_120)"
echo "- GPUx < 1 means GPU is slower than sequential (offload overhead dominates)"
echo "- GPUx > 1 means GPU is faster than sequential"
echo ""
