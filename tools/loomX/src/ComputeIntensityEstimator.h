#pragma once
#include "LoopAnalysisTypes.h"

namespace loomX {

// ISA-aware compute-intensity estimator.
// Counts floating-point and integer operations vs. memory accesses inside a
// loop and classifies the loop as memory-bound, balanced, or compute-bound.
class ComputeIntensityEstimator {
public:
    ComputeIntensityEstimator() = default;

    // Analyze the loop body.  The optional targetFLOPsPerMemOp threshold
    // defaults to a GPU-friendly value (8.0).
    ComputeIntensityResult analyze(SgForStatement* loop,
                                   double targetFLOPsPerMemOp = 8.0);

    // Static counts of FLOPs and memory operations.
    static void countOperations(SgForStatement* loop,
                                int& flops,
                                int& memOps,
                                int& intOps,
                                bool& heavyMath);

private:
    static bool isFloatingPointOp(SgBinaryOp* op);
    static bool isFloatingPointType(SgType* type);
    static bool isHeavyMathFunction(const std::string& name);
};

} // namespace loomX
