#pragma once
#include "rose.h"
#include <string>

namespace loomX {

// Target for parallelized loop.
enum class ParallelTarget { SEQUENTIAL, CPU_OPENMP, GPU_OFFLOAD };

// Tunable parameters for the GPU profitability cost model.
struct ProfitabilityConfig {
    // Default iteration count used when the loop bound is symbolic and cannot
    // be evaluated at compile time.
    long defaultIterationsForSymbolicBound = 100000;

    // Minimum iterations to consider any parallelization.
    long minIterationsForParallel = 100;

    // Minimum iterations for GPU offload (when compute-bound).
    long minIterationsForGPU = 100000;

    // Minimum iterations for CPU OpenMP when loop is irregular/divergent.
    long minIterationsForCPU = 1000;

    // Minimum total FLOPs for the nested-loop GPU heuristic.
    long long minNestedFlopForGPU = 1000000;

    // Minimum total FLOPs (flopCount * iterations) for any GPU offload.
    // This stops trivially large but computationally light loops (e.g. init,
    // checksum) from going to the GPU.
    long long minTotalFlopForGPU = 10000000;

    // FLOPs per memory-op threshold used by the compute-intensity estimator.
    double computeBoundThreshold = 16.0;

    // Abstract hardware throughput numbers (relative units). These are not
    // meant to model a specific GPU exactly; they give the cost model a
    // consistent shape so decisions improve as the loop gets larger or more
    // compute-intensive.
    // Defaults are tuned so that large compute-bound kernels (e.g. PolyBench
    // gemm) favour GPU offload while small or memory-bound loops stay on CPU.
    double cpuComputeThroughput = 5.0;     // GFLOP/s (scalar, single-thread)
    double gpuComputeThroughput = 5000.0;  // GFLOP/s
    double cpuMemoryBandwidth = 20.0;      // GB/s
    double gpuMemoryBandwidth = 500.0;     // GB/s
    double pcieBandwidth = 32.0;           // GB/s
    double pcieLatency = 5.0;              // microseconds per transfer direction
    double kernelLaunchOverhead = 10.0;    // microseconds

    // GPU must be at least this many times faster than CPU to justify offload.
    double minGpuSpeedup = 1.2;

    // ----- Second-order heuristics -----

    // GPU hardware sizing for utilization penalty.  A loop with too few
    // iterations cannot fill the device, so effective compute throughput is
    // reduced.  These defaults describe a mid-sized NVIDIA GPU; override for
    // the actual target hardware.
    int gpuSMCount = 80;
    int gpuWarpsPerSM = 16;
    int gpuThreadsPerWarp = 32;

    // Fraction of peak FP throughput a typical kernel actually achieves
    // (memory latency, instruction mix, occupancy, etc.).
    double gpuComputeEfficiency = 0.5;

    // Heavy math (sin/cos/sqrt/exp/log) is expensive on both host and device,
    // but the relative penalty differs.  This factor inflates GPU compute time
    // for kernels containing such calls.
    double heavyMathCostFactor = 8.0;

    // Fraction of total memory traffic that crosses PCIe.  With target-data
    // hoisting this can be much less than 1.0 because data is resident on the
    // device across multiple kernels.
    double pcieTransferFactor = 1.0;

    // CPU cache reuse factor for unit-stride/sequential accesses.  The naive
    // memory-time estimate assumes every access misses in cache; this factor
    // scales down CPU memory traffic when the access pattern is cache-friendly.
    double cpuCacheReuseFactor = 0.25;

    // Extra overhead per reduction variable on the GPU (tree reduction).
    double gpuReductionOverhead = 1e-6;
};

// Dominant memory-access pattern for a loop, used to model cache reuse on
// the CPU and coalescing on the GPU.
enum class AccessPattern {
    UNKNOWN,
    UNIT_STRIDE,    // Index is exactly the loop variable (plus optional cast)
    STRIDED,        // Linear function of the loop variable, e.g. a[i*2]
    IRREGULAR       // Index involves other variables, function calls, etc.
};

// Canonical form classification for a for-loop.
enum class CanonicalForm {
    UNKNOWN,        // Not analyzed yet
    CANONICAL,      // Recognized canonical for-loop
    NON_CANONICAL   // Does not match canonical form
};

// Reason a loop was rejected as non-canonical.
enum class NonCanonicalReason {
    NONE,
    NO_INIT,
    NO_TEST,
    NO_INCREMENT,
    NON_INTEGER_ITERATOR,
    NON_LITERAL_OR_PARAMETER_BOUND,
    NON_UNIT_STRIDE,
    NON_IDIOMATIC_INCREMENT,
    LOOP_VAR_MODIFIED_IN_BODY,
    MULTIPLE_ITERATORS
};

// Result of canonical-form analysis.
struct CanonicalResult {
    CanonicalForm form = CanonicalForm::UNKNOWN;
    NonCanonicalReason reason = NonCanonicalReason::NONE;
    SgInitializedName* indexVar = nullptr;
    SgExpression* lowerBound = nullptr;
    SgExpression* upperBound = nullptr;
    SgExpression* stride = nullptr;
    bool isLowerInclusive = true;   // Canonical ROSE form is usually i=lb; i<ub; i++
    bool isUpperExclusive = true;
    std::string note;               // Human-readable detail
};

// Reduction operator kinds detected by the reduction analyzer.
enum class ReductionOp {
    UNKNOWN,
    ADD,
    SUB,
    MUL,
    MIN,
    MAX,
    BIT_AND,
    BIT_OR,
    BIT_XOR,
    LOGICAL_AND,
    LOGICAL_OR
};

// Description of a single reduction variable.
struct ReductionInfo {
    SgInitializedName* variable = nullptr;
    ReductionOp op = ReductionOp::UNKNOWN;
    std::string opString;  // OpenMP reduction operator string
    SgStatement* statement = nullptr;  // Statement where reduction was detected
};

// Divergence classification.
enum class DivergenceKind {
    NONE,
    DATA_DEPENDENT_IF,      // if (A[i] > 0)
    DATA_DEPENDENT_SWITCH,  // switch (A[i])
    EARLY_EXIT,             // break/continue/return inside loop
    INNER_LOOP,             // nested loop causes thread divergence
    FUNCTION_CALL           // call that may have divergent internal control flow
};

// Result of divergence analysis.
struct DivergenceResult {
    bool isDivergent = false;
    DivergenceKind kind = DivergenceKind::NONE;
    SgNode* source = nullptr;
    std::string description;
};

// Compute-intensity classification.
enum class IntensityClass {
    UNKNOWN,
    MEMORY_BOUND,    // More memory ops than compute
    BALANCED,        // Comparable compute and memory
    COMPUTE_BOUND    // Compute dominates
};

// Result of compute-intensity estimation.
struct ComputeIntensityResult {
    IntensityClass classification = IntensityClass::UNKNOWN;
    AccessPattern accessPattern = AccessPattern::UNKNOWN;
    double flopsPerMemoryOp = 0.0;   // FLOPs per memory access
    long long flopCount = 0;         // Static estimate of FP operations
    long long memoryOpCount = 0;     // Static estimate of memory accesses
    long long integerOpCount = 0;    // Static estimate of integer operations
    bool hasHeavyMath = false;       // sin/cos/sqrt/exp/log/etc.
    std::string note;
};

} // namespace loomX
