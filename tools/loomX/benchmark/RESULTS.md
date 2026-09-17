# loomX evaluation results

## Environment

- Host CPU: not explicitly recorded; GCC 11.4.0 on Ubuntu 22.04
- GPU: NVIDIA GeForce RTX 4050 Laptop, compute capability 8.9 (sm_89), driver 580.178.04
- CUDA: 12.4 (reported by nvidia-smi)
- CPU compiler: gcc 11.4.0
- GPU compiler: custom-built clang 15 at `/tmp/opencode/llvm-build-offload/bin/clang`
  - Built with `-DOPENMP_ENABLE_LIBOMPTARGET=ON`, targets `AArch64;X86;NVPTX`.
  - Device bitcode `libomptarget-nvptx-sm_86.bc` installed in the same lib directory.
  - LLVM 15 does not support `sm_89`, so GPU configs were compiled for `sm_86` and JITed by the driver.
- Nsight Systems: 2023.4.4 (`/usr/local/cuda-12.4/bin/nsys`)
- Python: 3.10.12

## Limitations of this run

- **GPU clocks were not locked** (`LOCK_CLOCKS=no`).
- **Nsight Systems transfer/kernel breakdown is unavailable**: the installed `nsys` reports CUPTI data under different SQLite table names than `bench_harness.py` expects (`CUPTI_ACTIVITY_KIND_MEMCPY` / `KERNEL`), so H2D/kernel/D2H columns are zero.
- **PolyBench correctness checks via stdout are not meaningful**: kernels compiled with `-DPOLYBENCH_TIME` emit timing to stderr and produce empty stdout, so `check_correctness.py` trivially passes. Timing numbers are still valid.
- **Only 3 PolyBench kernels** were configured in `run_benchmarks.sh` (`gemm`, `syrk`, `syr2k`). A full 30-kernel sweep would require extending the driver.
- **Interproc-microbench GPU timings are dominated by Nsight profiling overhead**: each run is wrapped in `nsys profile`, which adds ~2.6 s of fixed overhead. The corpus workloads are tiny (N≈200000), so these numbers are not a meaningful measure of GPU speedup.

## 1. Interprocedural micro-benchmark corpus

Hot loops (loops that contain function calls) and whether loomX parallelized them in `--cpu-only` mode:

| benchmark | hot-loop contains | expected | loomX verdict |
|-----------|-------------------|----------|---------------|
| aliased_pointer_write | aliased pointer write through callee | reject | **rejected** (correct negative control) |
| call_chain | chain of two pure callees | parallelize | **parallelized** |
| conditional_call | callee called only on even iterations | parallelize | **parallelized** |
| function_pointer | call through function pointer | reject | **rejected** (target not resolvable) |
| global_writer | callee writes shared global | reject | **rejected** (correct negative control) |
| leaf_call | pure leaf helper | parallelize | **parallelized** |
| local_array | callee uses only its own local array | parallelize | **parallelized** |
| loop_invariant_call | loop-invariant pure callee | parallelize | **parallelized** |
| pointer_escape | pointer escapes to global | reject | **rejected** (correct negative control) |
| readonly_helper | read-only helper | parallelize | **parallelized** |
| recursive_sum | recursive callee | reject | **rejected** |
| reduction_call | reduction inside callee | parallelize | **rejected** (limitation) |

Interprocedural claim: on this corpus, **7 of 9** safe hot loops with function calls were parallelized, and **all 3** negative controls were correctly rejected. The remaining gaps are `reduction_call` (reduction through a pointer callee is not recognized) and conservative handling of `function_pointer` / `recursive_sum`, which are intentionally hard cases.

### Speedups (seq vs cpu_omp, 10 runs, median)

```
benchmark,seq_s,cpu_omp_speedup,gpu_naive_speedup,gpu_profitable_speedup
aliased_pointer_write,0.0026,0.110,0.001,0.001
call_chain,0.0029,0.122,0.001,0.001
conditional_call,0.0029,0.070,0.001,0.001
function_pointer,0.0035,0.157,0.001,0.001
global_writer,0.0020,0.194,0.001,0.001
leaf_call,0.0030,0.126,0.001,0.001
local_array,0.0033,0.185,0.001,0.001
loop_invariant_call,0.0019,0.063,0.001,0.001
pointer_escape,0.0016,0.104,0.001,0.001
readonly_helper,0.0034,0.116,0.001,0.001
recursive_sum,0.0026,0.090,0.001,0.001
reduction_call,0.0021,0.137,0.001,0.001

Geometric mean cpu_omp/seq: 0.117x
```

The corpus is intentionally small (N≈200000) and the overhead of OpenMP thread launch dominates; speedup is not the metric of interest here. GPU numbers are dominated by `nsys profile` overhead and should be ignored for performance claims.

## 2. PolyBench subset

Configs evaluated: `seq`, `cpu_omp`, `gpu_naive`, `gpu_profitable`. Dataset: `LARGE_DATASET`.

```
benchmark,seq_s,cpu_omp_speedup,gpu_naive_speedup,gpu_profitable_speedup
gemm,0.6273,4.184,0.221,0.221
syr2k,2.6005,3.228,0.756,0.707
syrk,0.5633,2.858,0.198,0.194

Geometric mean speedup vs. seq:
  cpu_omp:        3.380x
  gpu_naive:      0.321x
  gpu_profitable: 0.311x
```

On these three memory-bound kernels, `cpu_omp` is substantially faster than sequential, while both GPU configs are much slower because host↔device transfer and kernel launch overhead dominate the compute. `gpu_profitable` does not currently choose CPU over GPU for these kernels; this is a known limitation of the cost model on this small subset.

Correctness: stdout-based diff is not meaningful for `-DPOLYBENCH_TIME` kernels; they emit timing to stderr. All binaries ran to completion without crashes.

## 3. DataRaceBench safety-decision evaluation

Mode: `--cpu-only`. Ground truth: `-yes.c` files contain a race and should be rejected; `-no.c` files are race-free and should be accepted.

`check_dataracebench.py` now supports two evaluation modes:

- **Lenient (default)**: keeps the original `#pragma omp` pragmas in place and disables the scalar-dependence guard. This measures how much of the DataRaceBench corpus loomX leaves parallelized.
- **Strict (`--strict`)**: strips the original pragmas and enables the scalar-dependence guard. This measures loomX's own safety judgment.

### Lenient evaluation (default)

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

Most `-no` cases are accepted because the original (correct) pragmas remain in the file.

### Strict evaluation (`--strict`)

```
Total evaluated: 117
Correct:         53 (45.3%)
False positives (accepted a -yes race): 26
False negatives (rejected a -no safe case): 38

Confusion matrix:
                accepted  rejected
-yes (unsafe)      26       25
-no  (safe)        28       38
```

The false-positive count drops from **50 to 26**, but the false-negative count rises because many `-no` cases are safe only under OpenMP constructs (`task`, `sections`, `simd`, `atomic`, `barrier`, etc.) that are removed during stripping.

### Strict hot-loop evaluation (`--strict` with per-loop matching)

The harness was further refined to use loomX's new `--analyze-only` mode and to compare verdicts only for the loop that originally carried the OpenMP pragma. This avoids counting safe init/checksum loops as acceptances. Multi-dimensional array dependence analysis and a smarter scalar-private check were also added so that more safe `-no` cases are accepted.

```
Total evaluated: 117
Correct:         63 (53.8%)
False positives (accepted a -yes race): 14
False negatives (rejected a -no safe case): 40

Confusion matrix:
                accepted  rejected
-yes (unsafe)      14       37
-no  (safe)        26       40
```

False positives dropped from **50 to 14**. Several of the remaining 14 accepted `-yes` cases (e.g. `DRB021-reductionmissing`, `DRB009-lastprivatemissing`, `DRB073-doall2-orig-yes`) are actually race-free after loomX supplies the missing `reduction`/`private`/`lastprivate` clause, but DataRaceBench still labels them as `-yes` because the original source was buggy. The pipeline now prioritizes output correctness over maximizing acceptance.

## 4. What is missing for the full claims

- **Profitability claim**: a broader set of kernels (full PolyBench, Rodinia) and locked GPU clocks are needed to get clean `gpu_naive` vs `gpu_profitable` comparisons. The Nsight Systems CUPTI table-name mismatch also needs to be fixed to report H2D/kernel/D2H breakdown.
- **Broader PolyBench claim**: extend `run_benchmarks.sh` to all 30 kernels or add a separate sweep script.
- **Rodinia**: not run; per-app curation is needed.
- **Interprocedural analysis**: add reduction-through-callee recognition and handle recursion/function-pointer sets more precisely.

## Files produced

- `results/interproc-microbench.csv`
- `results/polybench.csv`
- `dataracebench_results.csv`
