# AGENTS.md — loomX

loomX is a ROSE-based source-to-source tool in `tools/loomX/`. The rest of the repository is the ROSE compiler infrastructure; limit editing to `tools/loomX/` unless you are changing ROSE itself.

## Build

Standalone (preferred for development):

```bash
cd tools/loomX
mkdir build && cd build
cmake .. -DROSE_INSTALL_PREFIX=/path/to/rose/install
make -j$(nproc)
```

In-tree inside ROSE:

```bash
cd /path/to/rose-build
make tools_loomX_exe -j$(nproc)
```

The binary is named `loomX`. At runtime it needs `librose.so` and Clang/LLVM libraries on `LD_LIBRARY_PATH` / rpath.

## CLI modes

```bash
./loomX --cpu-only       input.c -o out.c   # CPU OpenMP where the model says it pays
./loomX --cpu-forced     input.c -o out.c   # CPU OpenMP for every safe loop (ablation)
./loomX --gpu-naive      input.c -o out.c   # offload everything safe (ablation)
./loomX --gpu-profitable input.c -o out.c   # default: profitability gate chooses CPU vs GPU
./loomX --no-scalar-dep-check input.c -o out.c  # disable scalar loop-carried-dependence guard
./loomX --cpu-only --analyze-only input.c    # print per-loop verdicts, no output file
./loomX -v input.c                           # verbose analysis output
```

`--cpu-only` and `--cpu-forced` are different on purpose. `--cpu-only` still
honours `minTotalFlopForCPUOpenMP` (10M by default), so small loops stay serial
because threading them costs more than it saves. `--cpu-forced` ignores the FLOP
thresholds and parallelizes anything that passed the safety checks; it is the
forced-CPU arm needed to measure offload decision regret, and the CPU-side
counterpart of `--gpu-naive`. It can be slower than sequential by design.

Tunable thresholds: `--min-gpu-speedup`, `--min-total-flop`, `--min-nested-flop`,
`--min-cpu-openmp-flop`, `--compute-bound-threshold`.

Always pass `-rose:skipfinalCompileStep` if you do not want ROSE to invoke the backend compiler.

### `--backend` only takes effect for GPU targets

`--backend` selects the code generator, but the generator is only consulted for
loops that were actually assigned a GPU target. If the cost model keeps a loop
on the CPU, `--backend cuda` and `--backend opencl` both emit plain
`#pragma omp parallel for`. This is easy to mistake for the flag being ignored.
To see real CUDA or OpenCL output, combine the backend with a mode that forces
offloading (`--gpu-naive` or a `--gpu-profitable` run that selects the GPU).

## Testing

```bash
cd tools/loomX
./tests/run_tests.sh
```

This runs two distinct checks, and they answer different questions:

1. **Smoke** (`run_tests.sh`) runs `loomX` over every `tests/*.c`, compiles
   original and transformed with `gcc -O2 -fopenmp -lm`, and compares **exit
   codes**. This is not a correctness test. POSIX truncates exit status to 8
   bits, so a checksum of 500500, 500756 and 501012 all present as exit code 20
   — a wide band of wrong answers passes, and the check cannot report how wrong.
2. **Numerical validation** (`run_validation.sh`, invoked by `run_tests.sh`)
   compiles the untouched program as a golden reference, then compiles and runs
   loomX's output and compares full-precision stdout with
   `validate_results.py`. Kernels live in `tests/validate/`; each prints
   `<key> <value>` lines. The comparison is keyed, so a dropped or renamed key
   fails rather than being flattened away.

`run_validation.sh` also probes the mode x backend matrix and records a reason
for every skip in `tests/validate-out/summary.csv`, so "not tested" stays
distinguishable from "tested and wrong`. It reports three distinct outcomes:

- `pass` -- compiled, executed, values matched.
- `compile-only` -- the toolchain accepts the dialect but cannot execute it, so
  no numerical claim is made. Two cases, both confirmed against a hand-written
  oracle that fails identically: `gcc -fopenacc` aborts on OpenACC at runtime
  ("libgomp: target function wasn't mapped"), and `gcc -fopenmp` does not copy
  values back out of an `omp target` data region. Use `--threads`, `--rtol`,
  `--atol` to tighten.
- `skip` -- no toolchain (no `nvcc` for CUDA, no OpenCL C compiler for OpenCL).

`check_pragma_shape.py` runs on every row and validates structure without a
compiler: a pragma must govern a loop, `reduction(...)` items must carry an
operator, and `collapse(N)` must not exceed the loop depth that follows. This
catches a pragma attached to the wrong statement, which still compiles and
still produces correct numbers because the loop merely stayed serial.

`run_shape_tests.sh` (wired into `run_tests.sh`) is a third, cheaper layer that
needs no GPU, no CUDA, and no OpenCL runtime. It drives loomX over the
`tests/validate_shape_ptr*.c` fixtures and asserts pointer arguments are never
mapped as a bare name -- a bare `map(tofrom:a)` transfers the pointer value
rather than the array it points at, which used to kill every generated binary
with an illegal memory access while still looking like reasonable code. Run it
directly to check codegen shape without waiting on the numerical matrix.

## Evaluation harness

The benchmark harness does **not** live in this tree. It was moved out to a
standalone repository so that benchmark artifacts and large vendored suites do
not pollute the compiler checkout:

```
../loomX-benchmarks/
```

If that sibling directory is absent, this checkout simply has no harness. Do
not restore one under `tools/loomX/benchmark/`: that duplicates the canonical
harness and reintroduces the coupling the split was meant to remove.

The harness covers PolyBench/C, Rodinia, DataRaceBench, NPB, Parboil, the LLVM
Test Suite, AutoParBench and Loop-Fission, and configures its own toolchain in
`config.env`. Read that file before concluding a compiler is unavailable:
`clang` and `nvcc` are typically **not on the default PATH**, so
`command -v clang` returns nothing even on a fully provisioned machine. Probe
the configured paths, not `PATH`.

Machine-readable results land in `../loomX-benchmarks/results/`.

## Important gotchas

- **ROSE library path**: the harness sets `LD_LIBRARY_PATH` from
  `ROSE_INSTALL_PREFIX` in `../loomX-benchmarks/config.env`. Point that at your
  ROSE install if `librose.so` is not found.
- **Translator path**: the harness reads `LOOMX` from
  `../loomX-benchmarks/config.env`, which should point at
  `tools/loomX/build/loomX`. Rebuild that binary after changing the compiler;
  the harness will not rebuild it for you.
- **Reduction results on `omp target` are mapped explicitly**:
  `OpenMPCodeGen::insertGPUPragma` folds reduction variables into the map
  clause, emitting `map(tofrom:x)` alongside `reduction(+:x)`. This is
  defensive, not a fix for an observed miscompile: with clang-15
  `-fopenmp-targets=nvptx64-nvidia-cuda` on sm_86 the unmapped form already
  returns the correct result. The mapping is emitted because correctness
  otherwise rests on a compiler-specific reading of reduction-on-target, and
  because gcc does not implement the writeback at all (see the gcc note
  below). Do not treat its absence as a known bug.
- **Generated CUDA/OpenCL host stubs fail loudly**: every `cudaMalloc`,
  `cudaMemcpy`, and kernel launch is checked and aborts via `__loomx_cuda_fail`
  on error. This is deliberate. The stubs used to be `if (rc != cudaSuccess)
  return;` on every site, which left each host-side output variable at its
  pre-call value and exited 0, so a device-side failure produced a
  plausible-looking wrong answer. Do not "simplify" those checks back into bare
  `return;`: a numerical mismatch caused that way is indistinguishable from a
  real codegen bug.
- **Kernel index mapping assumes nothing about stride**: `KernelCodeGen`
  records each collapsed loop's step and derives both the trip count and the
  index from it. A unit-step trip count makes a strided loop visit a contiguous
  much larger range than the source loop did -- `for (j = 1; j <= NY; j += 97)`
  summed a full slab instead of the strided sample, which is numerically wrong
  while still compiling cleanly.
- **Subscript linearisation is outermost-dim-first**: `linearizeSubscripts`
  walks the `SgPntrArrRefExp` chain and reverses it so subscripts line up with
  `arrayDims()`. For `u[0][j][i]` the wrong order yields
  `j*1026*1026 + 0*1026 + i` instead of `0*1026*1026 + j*1026 + i`. On a square
  array the mis-indexed address is still in bounds, so it corrupts values
  silently; on a non-square array it walks off the allocation.
- **Non-collapsed inner loops need their induction variables declared**: a
  kernel that collapses `i` and `j` but keeps `for (k ...)` in the body must
  emit `int k;` in the prologue, because the function-scope declaration is not
  carried into the kernel text. `collapsedIndexVars` (the chain only) decides
  this, not `indexVars` (which also holds inner-loop indices, so they stay out
  of the *parameter* list).
- **Two cost-model passes are expected**: `decideTarget` runs once from local
  information and again after callee work is folded in, so `[GpuProfitability]`
  prints twice per loop. The log lines are labelled `pass=local` and
  `pass=callee-augmented`; the second verdict is the one that is acted on.
- **Rejection reasons are specific**: `--analyze-only` distinguishes unsafe
  (irregular access, divergence, unsafe callee) from cost (trip count or FLOP
  threshold), because a safety rejection and a cost rejection need different
  fixes.
- **GPU offload compiler**: GPU configs require a clang built with `openmp` and
  `libomptarget-nvptarget` support, taken from `COMPILER_GPU` in `config.env`
  (clang-15 here). If
  `clang -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda` fails with "no library
  'libomptarget-nvptx.bc' found", GPU configs are skipped with that reason
  recorded. The harness records such skips in a failures CSV rather than
  dropping them silently.
- **gcc is not an offload oracle**: it compiles `omp target` directives but
  does not implement the data environment, and `-fopenacc` aborts at runtime.
  Both were confirmed with hand-written reference programs that fail the same
  way. Use gcc for CPU OpenMP numerics only.
- **GPU architecture**: `GPU_ARCH` in `../loomX-benchmarks/config.env` pins the
  offload target. The RTX 4050 in this machine is sm_89, but `sm_86` is used
  because it is within the arch range the CUDA 12.4 / clang-15 combination
  supports and the driver JITs it upward. Override it if a suite needs a
  different arch.
- **Clock locking**: Set `LOCK_CLOCKS=yes` when benchmarking for reproducible
  timing. Requires root for `nvidia-smi -lgc`.
- **DataRaceBench evaluation**: `check_dataracebench.py` judges acceptance by
  checking whether the generated source contains `#pragma omp`. Use `--strict`
  to strip input pragmas, enable the scalar-dependence guard, and use
  `--analyze-only` to match verdicts against the original hot loop. The default
  lenient mode leaves pragmas in place. It reports false positives (accepted a
  `-yes` race) and false negatives (rejected a `-no` safe case).
- **Reporting convention**: report full end-to-end wall clock (median of >=10
  runs), geometric mean across benchmarks, and failure/regression counts from
  `failures.csv`. Do not report kernel-only time or arithmetic mean. Report
  forced-arm configurations as ablations, not as recommended settings.
- **Interprocedural micro-benchmarks** and PolyBench/Rodinia/DataRaceBench
  sources live in `../loomX-benchmarks/suites/`; `setup.sh` there verifies the
  vendored copies.

## Architecture pointers

- `src/main.cpp` — entry point, CLI parsing, drives IPA → loop discovery → dependence check → summary → codegen.
- `src/InterproceduralAnalysis.{h,cpp}` — whole-program side-effect summaries and call-graph safety.
- `src/GpuProfitability.{h,cpp}` — cost model and CPU/GPU/sequential target decision.
- `src/ComputeIntensityEstimator.{h,cpp}` — FLOP/memory-op counting, including a function-body estimator used by IPA.
- `src/OpenMPCodeGen.cpp` — pragma insertion and target-data-region post-processing.
- `src/LoopDependenceAnalysis.{h,cpp}` — loop-carried dependence check before parallelization.
