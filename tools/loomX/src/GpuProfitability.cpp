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

    // CPU execution overlaps compute and memory when data fits cache; use the
    // larger of the two rather than their sum.
    return std::max(computeTime, memoryTime);
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

    double launchTime = config_.kernelLaunchOverhead * 1e-6; // seconds
    double computeTime = flops / (config_.gpuComputeThroughput * 1e9);
    double memoryTime = memBytes / (config_.gpuMemoryBandwidth * 1e9);

    // PCIe transfer: bandwidth plus a fixed latency in each direction.  For
    // small loops the latency dominates and keeps them on the CPU.
    double transferTime = dataBytes / (config_.pcieBandwidth * 1e9) +
                          2.0 * config_.pcieLatency * 1e-6;

    // On the GPU compute and device memory overlap; data movement does not.
    return launchTime + transferTime + std::max(computeTime, memoryTime);
}

void GpuProfitability::reevaluateTarget(loomX::LoopSummary& summary) {
    summary.target = decideTarget(summary);
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

    long long totalWork = totalFlops(summary);
    bool initLoop = isInitializationLoop(summary);
    bool reductionOnly = isReductionOnlyLoop(summary);

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
              << " totalFlops=" << totalWork
              << " initLoop=" << initLoop
              << " reductionOnly=" << reductionOnly
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

    // Gap-1 fix: initialization and reduction-only loops rarely benefit from
    // GPU offload because they are memory-bound and have little reuse. Force
    // them to CPU OpenMP (if large enough) or sequential, regardless of the
    // raw cost-model speedup.
    if (initLoop || reductionOnly) {
        if (iterations >= config_.minIterationsForParallel) {
            return ParallelTarget::CPU_OPENMP;
        }
        return ParallelTarget::SEQUENTIAL;
    }

    // Primary GPU rule: enough total work and estimated speedup.  The cost
    // model (including data-transfer latency) is what keeps small or memory-
    // bound loops on the CPU.
    if (totalWork >= config_.minTotalFlopForGPU &&
        speedup >= config_.minGpuSpeedup) {
        return ParallelTarget::GPU_OFFLOAD;
    }

    // Secondary GPU rule: compute-bound outer loops with nested canonical loops
    // and enough total work.  This catches kernels whose inner loops dominate
    // the work but where the outer loop trip count alone is modest.
    if (computeHeavy && regular && hasNestedLoops(loop) &&
        summary.intensity.flopCount >= config_.minNestedFlopForGPU &&
        totalWork >= config_.minTotalFlopForGPU &&
        speedup >= config_.minGpuSpeedup) {
        return ParallelTarget::GPU_OFFLOAD;
    }

    if (iterations >= config_.minIterationsForParallel) {
        return ParallelTarget::CPU_OPENMP;
    }

    return ParallelTarget::SEQUENTIAL;
}

long long GpuProfitability::totalFlops(const loomX::LoopSummary& summary) const {
    // The intensity estimator already scales flopCount by trip counts when it
    // can resolve them.  Only scale further when the trip count is unknown.
    if (summary.iterationCount < 0) {
        return summary.intensity.flopCount * 100000;
    }
    return summary.intensity.flopCount;
}

bool GpuProfitability::isInitializationLoop(const loomX::LoopSummary& summary) const {
    // An initialization loop typically has no reductions, touches at least one
    // array, and the array references are predominantly writes (e.g.
    // C[i][j] = ... or a[i] = 0).  This distinguishes init loops from compute
    // loops that read several arrays and write one.
    if (!summary.reductions.empty()) return false;
    if (summary.intensity.memoryOpCount == 0) return false;

    SgStatement* body = summary.loop->get_loop_body();
    if (!body) return false;

    Rose_STL_Container<SgNode*> arrRefs =
        NodeQuery::querySubTree(body, V_SgPntrArrRefExp);
    if (arrRefs.empty()) return false;

    // Count how many array references are on the LHS of an assignment.
    long long writeArrRefs = 0;
    long long readArrRefs = 0;
    for (SgNode* node : arrRefs) {
        SgPntrArrRefExp* arrRef = isSgPntrArrRefExp(node);
        if (!arrRef) continue;

        // Determine whether this array reference is on the LHS of an
        // assignment, either directly or through a multi-dimensional wrapper.
        bool isWrite = isLhsOfAssignment(arrRef);
        if (!isWrite) {
            SgNode* parent = arrRef->get_parent();
            while (parent && !isSgFunctionDefinition(parent)) {
                if (isLhsOfAssignment(isSgExpression(parent))) {
                    isWrite = true;
                    break;
                }
                if (!isSgPntrArrRefExp(parent) && !isSgCastExp(parent)) break;
                parent = parent->get_parent();
            }
        }

        if (isWrite) writeArrRefs++;
        else readArrRefs++;
    }

    // If all array references are writes and there are no reductions, it is
    // almost certainly an initialization loop.
    return writeArrRefs > 0 && readArrRefs == 0;
}

bool GpuProfitability::isReductionOnlyLoop(const loomX::LoopSummary& summary) const {
    // A reduction-only loop has at least one reduction, and the only variables
    // written inside the loop are the reduction variables.  This catches nested
    // checksum loops like:
    //   for(i) for(j) sum = sum + a[i][j];
    if (summary.reductions.empty()) return false;

    SgStatement* body = summary.loop->get_loop_body();
    if (!body) return false;

    std::set<SgInitializedName*> reductionVars = summary.getReductionVariables();

    std::vector<SgNode*> readRefs, writeRefs;
    SageInterface::collectReadWriteRefs(body, readRefs, writeRefs);

    // collectReadWriteRefs also returns loop-index variables (they are
    // assigned in the for-header).  Exclude them from the write-set check.
    std::set<SgInitializedName*> loopIndexVars;
    Rose_STL_Container<SgNode*> forLoops =
        NodeQuery::querySubTree(body, V_SgForStatement);
    for (SgNode* node : forLoops) {
        SgForStatement* forStmt = isSgForStatement(node);
        SgInitializedName* index = SageInterface::getLoopIndexVariable(forStmt);
        if (index) loopIndexVars.insert(index);
    }

    for (SgNode* ref : writeRefs) {
        SgVarRefExp* varRef = isSgVarRefExp(ref);
        if (!varRef) continue;
        SgInitializedName* var = varRef->get_symbol()->get_declaration();
        if (!var) continue;
        if (loopIndexVars.find(var) != loopIndexVars.end()) continue;
        if (reductionVars.find(var) == reductionVars.end()) {
            return false;  // A non-reduction variable is written.
        }
    }

    return true;
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
