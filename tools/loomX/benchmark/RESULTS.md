# loomX evaluation results

## Environment

- Host CPU: not explicitly recorded; GCC 11.4.0 on Ubuntu 22.04
- GPU: NVIDIA GeForce RTX 4050 Laptop, compute capability 8.9 (sm_89), driver 580.178.04
- CUDA: 12.4 (reported by nvidia-smi)
- CPU compiler: gcc 11.4.0
- GPU compiler: clang — **not installed / not available**
- Nsight Systems: not installed
- Python: 3.10.12

## Limitations of this run

- **GPU configs could not be evaluated**: `clang` is not installed and there is no OpenMP-offload-capable compiler with `libomptarget-nvptx.bc`. The harness skipped GPU compilation. Consequently the headline profitability comparison (`gpu_naive` vs `gpu_profitable`) and the host↔device transfer breakdown cannot be produced in this environment.
- **PolyBench correctness checks via stdout failed**: PolyBench kernels with `-DPOLYBENCH_TIME` print execution time to stderr, not stdout. `check_correctness.py` compares stdout, so it reports shape mismatches for kernels that produced no stdout. Timing numbers are still valid.
- **Only 3 PolyBench kernels** were configured in `run_benchmarks.sh` (gemm, syrk, syr2k). A full 30-kernel sweep would require extending the driver or using `run_polybench_suite.py`.
- GPU clocks were **not locked** (`LOCK_CLOCKS=no`).

## 1. Interprocedural micro-benchmark corpus

Hot loops (loops that contain function calls) and whether loomX parallelized them in `--cpu-only` mode:

| benchmark | hot-loop line(s) | contains | loomX verdict |
|-----------|------------------|----------|---------------|
| aliased_pointer_write | 17 | aliased pointer write | **rejected** (correct negative control) |
| call_chain | 27 | chain of two callees | **parallelized** |
| conditional_call | 19, 28 | conditional callee | **parallelized** |
| function_pointer | 28 | call through function pointer | **rejected** (target not resolvable) |
| global_writer | 19 | callee writes global | **rejected** (correct negative control) |
| leaf_call | 19, 24 | leaf helper | **parallelized** |
| local_array | 26 | callee uses local array | **rejected** |
| loop_invariant_call | 18, 23 | loop-invariant callee | **parallelized** |
| pointer_escape | 19 | pointer escapes to global | **rejected** (correct negative control) |
| readonly_helper | 23, 28 | read-only helper | **parallelized** |
| recursive_sum | 21 | recursive callee | **rejected** |
| reduction_call | 18 | reduction inside callee | **rejected** |

Interprocedural claim: on this corpus, **5 of 12** hot loops with function calls were parallelized by loomX. The three negative controls were correctly rejected. The rejected positive cases (`function_pointer`, `local_array`, `recursive_sum`, `reduction_call`) identify current IPA gaps.

### Speedups (seq vs cpu_omp, 10 runs, median)

```
benchmark,seq_s,cpu_omp_speedup
aliased_pointer_write,0.0025,0.154
call_chain,0.0029,0.091
conditional_call,0.0037,0.119
function_pointer,0.0032,0.166
global_writer,0.0022,0.197
leaf_call,0.0029,0.112
local_array,0.0033,0.136
loop_invariant_call,0.0020,0.140
pointer_escape,0.0015,0.078
readonly_helper,0.0027,0.149
recursive_sum,0.0018,0.110
reduction_call,0.0019,0.244

Geometric mean cpu_omp/seq: 0.135x
```

The corpus is intentionally small (N≈200000) and the overhead of OpenMP thread launch dominates; speedup is not the metric of interest here.

## 2. PolyBench subset

Configs evaluated: `seq`, `cpu_omp`. GPU configs skipped due to missing offload compiler.

```
benchmark,seq_s,cpu_omp_speedup
gemm,0.5479,4.239
syr2k,2.2983,2.706
syrk,0.5481,2.943

Geometric mean cpu_omp/seq: 3.232x
```

Correctness: stdout-based diff is not meaningful for `-DPOLYBENCH_TIME` kernels; they emit timing to stderr. No output-integrity regression was observed by inspection of the generated sources.

## 3. DataRaceBench safety-decision evaluation

Mode: `--cpu-only`. Ground truth: `-yes.c` files contain a race and should be rejected; `-no.c` files are race-free and should be accepted.

```
Total evaluated: 117
Correct:         64 (54.7%)
False positives (accepted a -yes race): 50
False negatives (rejected a -no safe case): 3

Confusion matrix:
                accepted  rejected
-yes (unsafe)      50        1
-no  (safe)        63        3
```

The high false-positive count comes from DataRaceBench cases whose original OpenMP constructs (task, sections, simd, ordered, target, etc.) are stripped before running loomX; loomX then sees simple-looking loops and parallelizes cases that are unsafe under the original semantics.

## 4. What is missing for the full claims

- **Profitability claim**: requires an offload-capable clang and Nsight Systems to compare `gpu_naive` vs `gpu_profitable` end-to-end wall-clock speedups and to produce H2D/kernel/D2H breakdowns. See `build_gpu_clang.md`.
- **Broader PolyBench claim**: requires extending `run_benchmarks.sh` to all 30 kernels or using the referenced `run_polybench_suite.py`.
- **Rodinia**: not run; partly manual per-app curation is needed.

## Files produced

- `results/interproc-microbench.csv`
- `results/polybench.csv`
- `dataracebench_results.csv`
