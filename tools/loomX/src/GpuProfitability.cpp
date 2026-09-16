#include "GpuProfitability.h"
#include "LoopAnalysisUtil.h"
#include <iostream>

GpuProfitability::GpuProfitability()
    : iterationEstimator_(canonicalChecker_) {}

ParallelTarget GpuProfitability::classifyLoop(SgForStatement* loop) {
    using namespace loomX;

    // Canonical form is required for any parallelization.
    const CanonicalResult& canonical = canonicalChecker_.analyze(loop);
    if (canonical.form != CanonicalForm::CANONICAL) {
        std::cout << "[GpuProfitability] Loop at line "
                  << loop->get_file_info()->get_line()
                  << " non-canonical (" << canonical.note << ")\n";
        return ParallelTarget::SEQUENTIAL;
    }

    long iterations = iterationEstimator_.estimate(loop, false);
    bool regular = hasRegularAccessPattern(loop);
    DivergenceResult divergence = divergenceAnalyzer_.analyze(loop);
    bool divergent = divergence.isDivergent;
    ComputeIntensityResult intensity = intensityEstimator_.analyze(loop, 8.0);
    bool computeHeavy = (intensity.classification == IntensityClass::COMPUTE_BOUND);

    std::cout << "[GpuProfitability] Loop at line "
              << loop->get_file_info()->get_line()
              << " iterations=" << iterations
              << " regular=" << regular
              << " divergent=" << divergent
              << " computeHeavy=" << computeHeavy
              << " (" << intensity.note << ")"
              << "\n";

    if (iterations >= 0 && iterations < 100) {
        return ParallelTarget::SEQUENTIAL;
    }

    if (!regular || (divergent && divergence.kind != DivergenceKind::FUNCTION_CALL)) {
        // Irregular or strongly divergent: CPU only if large enough.
        if (iterations >= 1000) {
            return ParallelTarget::CPU_OPENMP;
        }
        return ParallelTarget::SEQUENTIAL;
    }

    if (iterations >= 100000 && computeHeavy && regular) {
        return ParallelTarget::GPU_OFFLOAD;
    }

    if (iterations >= 100) {
        return ParallelTarget::CPU_OPENMP;
    }

    return ParallelTarget::SEQUENTIAL;
}

long GpuProfitability::estimateIterationCount(SgForStatement* loop) {
    return iterationEstimator_.estimate(loop, false);
}

bool GpuProfitability::hasRegularAccessPattern(SgForStatement* loop) {
    // Check all array references in the loop.
    Rose_STL_Container<SgNode*> arrRefs =
        NodeQuery::querySubTree(loop, V_SgPntrArrRefExp);

    for (SgNode* node : arrRefs) {
        SgPntrArrRefExp* arrRef = isSgPntrArrRefExp(node);
        if (!arrRef) continue;

        SgExpression* index = arrRef->get_rhs_operand();
        if (!index) continue;

        // Function calls in array index => irregular.
        Rose_STL_Container<SgNode*> calls =
            NodeQuery::querySubTree(index, V_SgFunctionCallExp);
        if (!calls.empty()) return false;
    }

    return true;
}

bool GpuProfitability::hasDivergentControlFlow(SgForStatement* loop) {
    loomX::DivergenceResult result = divergenceAnalyzer_.analyze(loop);
    // Mild function-call divergence does not disqualify GPU for this legacy helper.
    if (result.kind == loomX::DivergenceKind::FUNCTION_CALL) return false;
    return result.isDivergent;
}

bool GpuProfitability::isComputeIntensive(SgForStatement* loop) {
    loomX::ComputeIntensityResult result = intensityEstimator_.analyze(loop, 8.0);
    return result.classification == loomX::IntensityClass::COMPUTE_BOUND;
}
