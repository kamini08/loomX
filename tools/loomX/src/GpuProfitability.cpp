#include "GpuProfitability.h"
#include "LoopAnalysisUtil.h"
#include "LoopDependenceAnalysis.h"
#include "ReductionDetector.h"
#include <iostream>
#include <algorithm>
#include <functional>

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

// Collect the base names of the array variables read and written inside a
// loop body.  Multi-dimensional references are unwrapped to their outermost
// array variable.  Names are used instead of SgInitializedName* identity
// because helper functions (e.g. PolyBench's init_array(A, B, C)) declare the
// arrays as fresh formal parameters that must still alias the callers' arrays.
void collectArrayAccessVars(SgStatement* body,
                            std::set<std::string>& reads,
                            std::set<std::string>& writes) {
    if (!body) return;

    std::vector<SgNode*> readRefs, writeRefs;
    SageInterface::collectReadWriteRefs(body, readRefs, writeRefs);

    auto baseArrayName = [](SgNode* ref) -> std::string {
        SgPntrArrRefExp* arr = isSgPntrArrRefExp(ref);
        if (!arr) return "";
        SgExpression* base = arr->get_lhs_operand();
        while (SgPntrArrRefExp* nested = isSgPntrArrRefExp(base)) {
            base = nested->get_lhs_operand();
        }
        if (SgVarRefExp* varRef = isSgVarRefExp(base)) {
            return varRef->get_symbol()->get_declaration()->get_name().getString();
        }
        return "";
    };

    for (SgNode* ref : readRefs) {
        const std::string name = baseArrayName(ref);
        if (!name.empty()) reads.insert(name);
    }
    for (SgNode* ref : writeRefs) {
        const std::string name = baseArrayName(ref);
        if (!name.empty()) writes.insert(name);
    }
}

} // namespace

// Return the next loop in a perfectly-nested chain: the loop whose body is a
// single for-statement (either directly, or as the only statement of an
// enclosing basic block).  Returns null when the body is not a single loop,
// which is what OpenMP's collapse(N) effectively requires of the first N-1
// loops.
static SgForStatement* nextPerfectNestedLoop(SgStatement* body) {
    if (SgForStatement* f = isSgForStatement(body)) return f;
    if (SgBasicBlock* bb = isSgBasicBlock(body)) {
        SgStatementPtrList& stmts = bb->get_statements();
        if (stmts.size() == 1) return isSgForStatement(stmts[0]);
    }
    return nullptr;
}

// True if every array variable written by body references every variable in
// collapsedVars somewhere in its subscript.  A write whose target is invariant
// along some collapsed dimension would be performed by many iterations of the
// flattened space onto the same location (cross-iteration accumulation),
// which races when the loops are collapsed without a reduction clause.
static bool fanWritesCoverAllCollapsedVars(
    SgStatement* body, const std::set<SgInitializedName*>& collapsedVars) {
    std::vector<SgNode*> readRefs, writeRefs;
    SageInterface::collectReadWriteRefs(body, readRefs, writeRefs);

    for (SgNode* ref : writeRefs) {
        SgPntrArrRefExp* arr = isSgPntrArrRefExp(ref);
        if (!arr) continue;

        // Gather the loop variables appearing anywhere in the subscript
        // expressions (recursing through nested dimension wrappers).
        std::set<SgInitializedName*> indexedBy;
        std::function<void(SgExpression*)> walk = [&](SgExpression* e) {
            if (!e) return;
            if (SgPntrArrRefExp* nested = isSgPntrArrRefExp(e)) {
                walk(nested->get_lhs_operand());
                walk(nested->get_rhs_operand());
                return;
            }
            if (SgVarRefExp* vr = isSgVarRefExp(e)) {
                if (SgInitializedName* v = vr->get_symbol()->get_declaration()) {
                    if (collapsedVars.count(v)) indexedBy.insert(v);
                }
                return;
            }
            // Any other expression (casts, arithmetic, constants): recurse so
            // the loop variable found inside an index expression is caught.
            for (size_t i = 0; i < e->get_numberOfTraversalSuccessors(); ++i) {
                walk(isSgExpression(e->get_traversalSuccessorByIndex(i)));
            }
        };
        walk(arr->get_lhs_operand());
        walk(arr->get_rhs_operand());
        for (SgInitializedName* v : collapsedVars) {
            if (!indexedBy.count(v)) return false;
        }
    }
    return true;
}

int GpuProfitability::collapseDepthFor(SgForStatement* loop) {
    if (!loop) return 1;

    std::vector<SgForStatement*> chain;
    chain.push_back(loop);
    while (true) {
        SgForStatement* current = chain.back();
        SgForStatement* inner = nextPerfectNestedLoop(current->get_loop_body());
        if (!inner) break;

        // The collapsed loop must itself be canonical.
        if (canonicalChecker_.analyze(inner).form != CanonicalForm::CANONICAL) {
            break;
        }

        // The collapsed loop must be independent: loop-carried dependences
        // (e.g. accumulating into a shared C[i][j] over k as in gemm/syr2k)
        // must not be parallelized by the collapse.  This catches dependences
        // CARRIED BY THE INNER LOOP's own variable.
        LoopDependenceAnalysis dep;
        if (dep.analyze(inner).hasLoopCarriedDependence) break;

        chain.push_back(inner);
    }
    if (chain.size() < 2) return 1;

    // Collapsing can still race on writes that are invariant along some
    // collapsed dimension even when each inner loop is individually
    // "independent"; e.g. gemver's "for(i) for(j) w[i] += beta*A[j][i]*x[j]"
    // races on w[i] across j.  The LoopDependenceAnalysis above only sees
    // dependences carried by the innermost variable and misses reductions
    // accumulated over the OUTER (prefix) variables.  Conservative rule:
    // every array write in the fan body must mention EVERY collapsed loop
    // variable in its subscript, so its location varies with the whole
    // flattened iteration space.
    std::set<SgInitializedName*> fanVars;
    for (SgForStatement* L : chain) {
        if (SgInitializedName* iv = SageInterface::getLoopIndexVariable(L)) {
            fanVars.insert(iv);
        }
    }
    SgStatement* innerBody = chain.back()->get_loop_body();
    if (!fanWritesCoverAllCollapsedVars(innerBody, fanVars)) return 1;

    return static_cast<int>(chain.size());
}

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

    // Units: throughput in GFLOP/s and bandwidth in GB/s, both per the
    // config.  The CPU OpenMP config runs on multiple cores, so compare GPU
    // offload against the parallel CPU.  Compute throughput scales with core
    // count; memory bandwidth is shared and does not.
    double cpuComputeThroughput = config_.cpuComputeThroughput *
                                  static_cast<double>(config_.cpuCoreCount);

    // Unit-stride accesses benefit from cache line reuse on the CPU.
    double cacheReuse = 1.0;
    if (summary.intensity.accessPattern == AccessPattern::UNIT_STRIDE) {
        cacheReuse = config_.cpuCacheReuseFactor;
    } else if (summary.intensity.accessPattern == AccessPattern::STRIDED) {
        cacheReuse = 0.5;
    }

    double computeTime = flops / (cpuComputeThroughput * 1e9); // seconds
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
    double computeTime = flops / (config_.gpuComputeThroughput *
                                  config_.gpuComputeEfficiency * 1e9);
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

void GpuProfitability::phaseCoupleInitLoops(
    std::vector<loomX::LoopSummary>& summaries) {
    const size_t n = summaries.size();
    if (n < 2) return;

    // Per-loop sets of array variable names read and written.
    std::vector<std::set<std::string>> reads(n), writes(n);
    for (size_t i = 0; i < n; ++i) {
        if (summaries[i].loop) {
            collectArrayAccessVars(summaries[i].loop->get_loop_body(),
                                   reads[i], writes[i]);
        }
    }

    // Phase 1: identify every init loop whose writes a later GPU loop consumes.
    // Promotion is all-or-nothing shared decision for the whole chain, because
    // init_array blocks are usually a run of consecutive helper loops writing
    // the arrays that the kernel then consumes; if we decided each one in
    // isolation the earlier loops would always be blocked by the later,
    // still-unpromoted helper loops.
    std::vector<bool> isGpu(n, false), candidate(n, false);
    for (size_t i = 0; i < n; ++i) {
        isGpu[i] = (summaries[i].target == loomX::ParallelTarget::GPU_OFFLOAD);
    }
    for (size_t i = 0; i < n; ++i) {
        if (isGpu[i] || !isInitializationLoop(summaries[i])) continue;
        if (writes[i].empty()) continue;
        for (size_t j = i + 1; j < n; ++j) {
            if (!isGpu[j]) continue;
            bool consumed = false;
            for (const std::string& w : writes[i]) {
                if (reads[j].count(w) || writes[j].count(w)) {
                    consumed = true;
                    break;
                }
            }
            if (consumed) {
                candidate[i] = true;
                break;
            }
        }
    }

    // Phase 2: assume every candidate is promoted, then verify that no loop in
    // program order between a candidate and its consuming GPU loop still runs
    // on the CPU while touching the same arrays (it would read stale host
    // data).  Loops nested inside a promoted loop are covered by their parent.
    for (size_t i = 0; i < n; ++i) {
        if (!candidate[i]) continue;

        size_t gpuIdx = n;
        for (size_t j = i + 1; j < n; ++j) {
            if (!isGpu[j]) continue;
            bool consumed = false;
            for (const std::string& w : writes[i]) {
                if (reads[j].count(w) || writes[j].count(w)) {
                    consumed = true;
                    break;
                }
            }
            if (consumed) {
                gpuIdx = j;
                break;
            }
        }
        if (gpuIdx == n) continue;

        bool safe = true;
        for (size_t k = i + 1; k < gpuIdx; ++k) {
            if (isGpu[k] || candidate[k]) continue;
            for (const std::string& w : writes[i]) {
                if (reads[k].count(w) || writes[k].count(w)) {
                    safe = false;
                    break;
                }
            }
            if (!safe) break;
        }
        if (!safe) continue;

        summaries[i].target = loomX::ParallelTarget::GPU_OFFLOAD;
        std::cout << "[GpuProfitability] Phase-coupled init loop at line "
                  << summaries[i].loop->get_file_info()->get_line()
                  << " with GPU loop at line "
                  << summaries[gpuIdx].loop->get_file_info()->get_line()
                  << "\n";
    }
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

    // Helper: does the loop carry enough work to amortise CPU OpenMP
    // threading / reduction overhead?  Without this gate, tiny init, reduction,
    // or fallback loops are classified as CPU_OPENMP and run slower than the
    // sequential baseline because of parallel overhead.
    auto cpuOpenmpWorthwhile = [&]() {
        return totalWork >= config_.minTotalFlopForCPUOpenMP;
    };

    // CPU OpenMP is most effective on outermost loops.  Parallelising an inner
    // loop that is repeatedly invoked by an outer sequential loop creates a
    // new parallel region every iteration of the outer loop, which is usually
    // slower than just running the whole nest sequentially.
    auto cpuTarget = [&]() {
        return isOutermostLoop(loop) ? ParallelTarget::CPU_OPENMP
                                     : ParallelTarget::SEQUENTIAL;
    };

    if (!regular || stronglyDivergent) {
        if (iterations >= config_.minIterationsForCPU && cpuOpenmpWorthwhile()) {
            return cpuTarget();
        }
        return ParallelTarget::SEQUENTIAL;
    }

    // Gap-1 fix: initialization and reduction-only loops rarely benefit from
    // GPU offload because they are memory-bound and have little reuse. Force
    // them to CPU OpenMP (if large enough) or sequential, regardless of the
    // raw cost-model speedup.
    if (initLoop || reductionOnly) {
        if (iterations >= config_.minIterationsForParallel && cpuOpenmpWorthwhile()) {
            return cpuTarget();
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

    if (iterations >= config_.minIterationsForParallel && cpuOpenmpWorthwhile()) {
        return cpuTarget();
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

bool GpuProfitability::isOutermostLoop(SgForStatement* loop) {
    if (!loop) return false;
    SgNode* parent = loop->get_parent();
    while (parent) {
        if (isSgForStatement(parent)) return false;
        parent = parent->get_parent();
    }
    return true;
}
