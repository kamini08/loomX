#pragma once
#include "rose.h"
#include <string>

namespace loomX {

// Target for parallelized loop.
enum class ParallelTarget { SEQUENTIAL, CPU_OPENMP, GPU_OFFLOAD };

// Tunable parameters for the GPU profitability cost model.
struct ProfitabilityConfig {
    // Minimum iterations to consider any parallelization.
    long minIterationsForParallel = 100;

    // Minimum iterations for GPU offload (when compute-bound).
    long minIterationsForGPU = 100000;

    // Minimum iterations for CPU OpenMP when loop is irregular/divergent.
    long minIterationsForCPU = 1000;

    // Minimum total FLOPs for the nested-loop GPU heuristic.
    long long minNestedFlopForGPU = 1000000;

    // FLOPs per memory-op threshold used by the compute-intensity estimator.
    double computeBoundThreshold = 8.0;

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
    double kernelLaunchOverhead = 10.0;    // microseconds

    // GPU must be at least this many times faster than CPU to justify offload.
    double minGpuSpeedup = 1.2;
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
    double flopsPerMemoryOp = 0.0;   // FLOPs per memory access
    long long flopCount = 0;         // Static estimate of FP operations
    long long memoryOpCount = 0;     // Static estimate of memory accesses
    long long integerOpCount = 0;    // Static estimate of integer operations
    bool hasHeavyMath = false;       // sin/cos/sqrt/exp/log/etc.
    std::string note;
};

} // namespace loomX
