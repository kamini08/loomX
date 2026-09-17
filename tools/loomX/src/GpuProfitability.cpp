#include "GpuProfitability.h"
#include "LoopAnalysisUtil.h"
#include "ReductionDetector.h"
#include <iostream>
#include <algorithm>

using namespace loomX;

namespace {

// Convert a memory-op count into an approximate byte count. We assume
// double-precision (8 bytes) per access; this can be made type-aware later.
constexpr double BYTES_PER_MEM_OP = 8.0;

const char* accessPatternName(loomX::AccessPattern pat) {
    using namespace loomX;
    switch (pat) {
        case AccessPattern::UNIT_STRIDE: return "unit";
        case AccessPattern::STRIDED: return "strided";
        case AccessPattern::IRREGULAR: return "irregular";
        default: return "unknown";
    }
}

} // namespace

GpuProfitability::GpuProfitability()
    : iterationEstimator_(canonicalChecker_) {}

GpuProfitability::GpuProfitability(const loomX::ProfitabilityConfig& config)
    : iterationEstimator_(canonicalChecker_), config_(config) {}

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
    summary.intensity = intensityEstimator_.analyze(loop, config_.computeBoundThreshold);

    // 6. Reductions.
    ReductionDetector reducer;
    summary.reductions = reducer.analyze(loop);

    // 7. Final target decision.
    summary.target = decideTarget(summary);

    return summary;
}

double GpuProfitability::estimateCpuTime(const loomX::LoopSummary& summary) const {
    double flops = static_cast<double>(summary.intensity.flopCount);
    double memBytes = static_cast<double>(summary.intensity.memoryOpCount) * BYTES_PER_MEM_OP;

    // Unit-stride accesses benefit from cache line reuse on the CPU.
    double cacheReuse = 1.0;
    if (summary.intensity.accessPattern == AccessPattern::UNIT_STRIDE) {
        cacheReuse = config_.cpuCacheReuseFactor;
    } else if (summary.intensity.accessPattern == AccessPattern::STRIDED) {
        cacheReuse = 0.5;
    }

    double computeTime = flops / (config_.cpuComputeThroughput * 1e9); // seconds
    double memoryTime = (memBytes * cacheReuse) / (config_.cpuMemoryBandwidth * 1e9);

    return computeTime + memoryTime;
}

double GpuProfitability::estimateDataMovementBytes(const loomX::LoopSummary& summary) const {
    // Approximate data movement as the bytes touched by memory operations.
    // With target-data hoisting this is pessimistic for secondary loops, but
    // it correctly penalises loops that touch a lot of data relative to work.
    return static_cast<double>(summary.intensity.memoryOpCount) * BYTES_PER_MEM_OP;
}

double GpuProfitability::estimateGpuTime(const loomX::LoopSummary& summary) const {
    double flops = static_cast<double>(summary.intensity.flopCount);
    double memBytes = static_cast<double>(summary.intensity.memoryOpCount) * BYTES_PER_MEM_OP;
    double dataBytes = estimateDataMovementBytes(summary) * config_.pcieTransferFactor;

    long iterations = summary.iterationCount;
    if (iterations < 0) iterations = config_.defaultIterationsForSymbolicBound;

    // GPU utilization: small loops cannot fill the device.
    long totalThreadsNeeded = iterations;
    long threadsPerSM = config_.gpuWarpsPerSM * config_.gpuThreadsPerWarp;
    long threadsForFullUtil = config_.gpuSMCount * threadsPerSM;
    double utilization = std::min(1.0, static_cast<double>(totalThreadsNeeded) /
                                       static_cast<double>(threadsForFullUtil));
    // Require at least a few warps per SM to hide latency.
    utilization = std::max(utilization, 0.05);

    // Coalescing: unit-stride accesses use full bandwidth; strided/irregular
    // accesses waste bandwidth.
    double coalescing = 1.0;
    if (summary.intensity.accessPattern == AccessPattern::STRIDED) {
        coalescing = 0.5;
    } else if (summary.intensity.accessPattern == AccessPattern::IRREGULAR) {
        coalescing = 0.25;
    }

    // Heavy math runs at lower effective throughput on the GPU.
    double effectiveGpuCompute = config_.gpuComputeThroughput * config_.gpuComputeEfficiency;
    if (summary.intensity.hasHeavyMath) {
        effectiveGpuCompute /= config_.heavyMathCostFactor;
    }

    double launchTime = config_.kernelLaunchOverhead * 1e-6;
    double computeTime = flops / (effectiveGpuCompute * 1e9 * utilization);
    double memoryTime = memBytes / (config_.gpuMemoryBandwidth * 1e9 * coalescing);
    double transferTime = dataBytes / (config_.pcieBandwidth * 1e9);

    double reductionTime = summary.reductions.size() * config_.gpuReductionOverhead;

    return launchTime + computeTime + memoryTime + transferTime + reductionTime;
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

    // Mild divergence from function calls or nested loops does not prevent
    // GPU offloading; nested loops are a common source of GPU parallelism.
    // Only data-dependent control flow or early exits force CPU execution.
    bool stronglyDivergent = summary.divergence.isDivergent &&
                             summary.divergence.kind != loomX::DivergenceKind::FUNCTION_CALL &&
                             summary.divergence.kind != loomX::DivergenceKind::INNER_LOOP;
    bool computeHeavy = (summary.intensity.classification == loomX::IntensityClass::COMPUTE_BOUND);

    double cpuTime = estimateCpuTime(summary);
    double gpuTime = estimateGpuTime(summary);
    double speedup = (gpuTime > 0.0) ? (cpuTime / gpuTime) : 0.0;

    std::cout << "[GpuProfitability] Loop at line "
              << loop->get_file_info()->get_line()
              << " iterations=" << iterations
              << " regular=" << regular
              << " divergent=" << stronglyDivergent
              << " computeHeavy=" << computeHeavy
              << " flops=" << summary.intensity.flopCount
              << " memOps=" << summary.intensity.memoryOpCount
              << " accessPattern=" << accessPatternName(summary.intensity.accessPattern)
              << " cpuTime=" << cpuTime
              << " gpuTime=" << gpuTime
              << " speedup=" << speedup
              << " (" << summary.intensity.note << ")"
              << "\n";

    if (iterations >= 0 && iterations < config_.minIterationsForParallel) {
        return ParallelTarget::SEQUENTIAL;
    }

    if (!regular || stronglyDivergent) {
        if (iterations >= config_.minIterationsForCPU) {
            return ParallelTarget::CPU_OPENMP;
        }
        return ParallelTarget::SEQUENTIAL;
    }

    // Primary GPU rule: large, compute-bound loop with estimated speedup.
    if (iterations >= config_.minIterationsForGPU && computeHeavy &&
        speedup >= config_.minGpuSpeedup) {
        return ParallelTarget::GPU_OFFLOAD;
    }

    // Secondary GPU rule: outer loops with nested canonical loops and enough
    // total work that the device can exploit the parallelism. Still require a
    // positive estimated speedup so we do not offload trivially small regions.
    if (regular && hasNestedLoops(loop) &&
        summary.intensity.flopCount >= config_.minNestedFlopForGPU &&
        speedup >= config_.minGpuSpeedup) {
        return ParallelTarget::GPU_OFFLOAD;
    }

    if (iterations >= config_.minIterationsForParallel) {
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

bool GpuProfitability::hasNestedLoops(SgForStatement* loop) {
    if (!loop) return false;
    SgStatement* body = loop->get_loop_body();
    if (!body) return false;
    Rose_STL_Container<SgNode*> nested =
        NodeQuery::querySubTree(body, V_SgForStatement);
    return !nested.empty();
}
