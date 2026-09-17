#pragma once
#include "IterationCountEstimator.h"
#include "LoopAnalysisTypes.h"

namespace loomX {

// ISA-aware compute-intensity estimator.
// Counts floating-point and integer operations vs. memory accesses inside a
// loop and classifies the loop as memory-bound, balanced, or compute-bound.
// When the loop contains nested canonical loops, their operation counts are
// multiplied by their estimated iteration counts.
class ComputeIntensityEstimator {
public:
    ComputeIntensityEstimator();

    // Analyze the loop body.  The optional targetFLOPsPerMemOp threshold
    // defaults to a GPU-friendly value (8.0).
    ComputeIntensityResult analyze(SgForStatement* loop,
                                   double targetFLOPsPerMemOp = 8.0);

    // Estimate the static work (FLOPs and memory ops) inside a function body.
    // Nested loops are multiplied by their trip counts, matching the behaviour
    // of analyze().  This is exposed so interprocedural analysis can include
    // callee work in loop profitability estimates.
    ComputeIntensityResult estimateFunctionWork(SgFunctionDefinition* def);

private:
    LoopCanonicalChecker canonicalChecker_;
    IterationCountEstimator iterationEstimator_;

    void countOperations(SgStatement* body,
                         long long tripCount,
                         long long& flops,
                         long long& memOps,
                         long long& intOps,
                         bool& heavyMath);

    static bool isFloatingPointOp(SgBinaryOp* op);
    static bool isFloatingPointType(SgType* type);
    static bool isHeavyMathFunction(const std::string& name);

    // Classify the dominant memory-access pattern in the loop body.
    AccessPattern classifyAccessPattern(SgForStatement* loop);

    // Return true if expr is exactly the loop iterator (possibly through a
    // cast), a constant multiple of it (STRIDED), or something else.
    AccessPattern classifyIndexExpression(SgExpression* index,
                                          SgInitializedName* loopVar) const;
};

} // namespace loomX
