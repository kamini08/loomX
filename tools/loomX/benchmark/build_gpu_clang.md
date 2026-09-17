# Building a clang that supports OpenMP GPU offload

The GPU configs in the harness need a clang that includes the OpenMP runtime
and the `libomptarget-nvptx.bc` device bitcode library.  The default project
build instructions only enabled `clang`, so offload compilation fails with:

```
clang: error: no library 'libomptarget-nvptx.bc' found in the default clang lib directory
```

## CMake configuration

Add `openmp` to `LLVM_ENABLE_PROJECTS` and turn on libomptarget:

```bash
cmake -S llvm-project/llvm -B llvm-build-offload -G Ninja \
  -DLLVM_ENABLE_PROJECTS="clang;openmp" \
  -DLLVM_ENABLE_RUNTIMES="openmp" \
  -DOPENMP_ENABLE_LIBOMPTARGET=ON \
  -DLLVM_TARGETS_TO_BUILD="AArch64;X86;NVPTX" \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DLLVM_ENABLE_ASSERTIONS=On \
  -DLLVM_OPTIMIZED_TABLEGEN=ON \
  -DLLVM_BUILD_LLVM_DYLIB=ON \
  -DLLVM_LINK_LLVM_DYLIB=ON \
  -DLLVM_BUILD_TOOLS=OFF \
  -DLLVM_USE_LINKER=lld \
  -DLLVM_CCACHE_BUILD=ON \
  -DLLVM_USE_SPLIT_DWARF=ON \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++
```

The `NVPTX` target is required for NVIDIA GPUs.  For AMD GPUs add `AMDGPU` and
use `-fopenmp-targets=amdgcn-amd-amdhsa`.

## Build

```bash
ninja -C llvm-build-offload clang omp
```

`libomptarget-nvptx.bc` is produced under `lib/libomptarget-nvptx.bc` in the
build tree.  Install or run from the build `bin/` directory.

## CUDA version vs. GPU architecture

The host CUDA toolkit must support the target architecture.  On the current
machine the GPU reports compute capability 12.0 but CUDA is 12.4, which clang
rejects for `sm_120`.  Two workarounds:

1. Install CUDA 12.8 or newer and compile with `-march=sm_120`.
2. Compile for an older architecture the driver can JIT (e.g. `sm_89`) and run
   it on the sm_120 GPU.

Set `GPU_ARCH` accordingly when running `run_benchmarks.sh`.

## Verify

```bash
clang -O3 -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda \
  -Xopenmp-target -march=sm_80 \
  -fopenmp -x c /dev/null -o /dev/null
```

If this compiles without the `libomptarget-nvptx.bc` error, the offload
compiler is usable.
