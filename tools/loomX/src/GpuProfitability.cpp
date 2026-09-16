#include "GpuProfitability.h"
#include "LoopAnalysisUtil.h"
#include "ReductionDetector.h"
#include <iostream>

using namespace loomX;

GpuProfitability::GpuProfitability()
    : iterationEstimator_(canonicalChecker_) {}

ParallelTarget GpuProfitability::classifyLoop(SgForStatement* loop) {
    return summarize(loop).target;
}

loomX::LoopSummary GpuProfitability::summarize(SgForStatement* loop) {
    using namespace loomX;

    LoopSummary summary;
    summary.loop = loop;

    // 1. Canonical form.
    summary.canonical = canonicalChecker_.analyze(loop);

    // 2. Iteration count.
    summary.iterationCount = iterationEstimator_.estimate(loop, false);

    // 3. Access regularity.
    summary.regularAccess = hasRegularAccessPattern(loop);

    // 4. Divergence.
    summary.divergence = divergenceAnalyzer_.analyze(loop);

    // 5. Compute intensity.
    summary.intensity = intensityEstimator_.analyze(loop, 8.0);

    // 6. Reductions.
    ReductionDetector reducer;
    summary.reductions = reducer.analyze(loop);

    // 7. Final target decision.
    summary.target = decideTarget(summary);

    return summary;
}

ParallelTarget GpuProfitability::decideTarget(const loomX::LoopSummary& summary) {
    SgForStatement* loop = summary.loop;

    // Non-canonical loops cannot be safely parallelized.
    if (summary.canonical.form != loomX::CanonicalForm::CANONICAL) {
        std::cout << "[GpuProfitability] Loop at line "
                  << loop->get_file_info()->get_line()
                  << " non-canonical (" << summary.canonical.note << ")\n";
        return ParallelTarget::SEQUENTIAL;
    }

    long iterations = summary.iterationCount;
    bool regular = summary.regularAccess;
    bool divergent = summary.divergence.isDivergent &&
                     summary.divergence.kind != loomX::DivergenceKind::FUNCTION_CALL;
    bool computeHeavy = (summary.intensity.classification == loomX::IntensityClass::COMPUTE_BOUND);

    std::cout << "[GpuProfitability] Loop at line "
              << loop->get_file_info()->get_line()
              << " iterations=" << iterations
              << " regular=" << regular
              << " divergent=" << divergent
              << " computeHeavy=" << computeHeavy
              << " (" << summary.intensity.note << ")"
              << "\n";

    if (iterations >= 0 && iterations < 100) {
        return ParallelTarget::SEQUENTIAL;
    }

    if (!regular || divergent) {
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
    if (result.kind == loomX::DivergenceKind::FUNCTION_CALL) return false;
    return result.isDivergent;
}

bool GpuProfitability::isComputeIntensive(SgForStatement* loop) {
    loomX::ComputeIntensityResult result = intensityEstimator_.analyze(loop, 8.0);
    return result.classification == loomX::IntensityClass::COMPUTE_BOUND;
}
