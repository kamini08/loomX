#include "rose.h"
#include "CodeGen.h"
#include "InterproceduralAnalysis.h"
#include "GpuProfitability.h"
#include "OpenMPCodeGen.h"
#include "OpenACCCodeGen.h"
#include "CudaCodeGen.h"
#include "OpenCLCodeGen.h"
#include "KernelCodeGen.h"
#include "LoopSummary.h"
#include "LoopDependenceAnalysis.h"
#include "PragmaAnalysis.h"
#include "LoopAnalysisTypes.h"
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <memory>

using namespace loomX;

// Find all for-loops in the AST
class LoopCollector : public AstSimpleProcessing {
public:
    std::vector<SgForStatement*> loops;
    void visit(SgNode* node) override {
        if (SgForStatement* loop = isSgForStatement(node)) {
            loops.push_back(loop);
        }
    }
};

// Returns true if a variable is a scalar (non-pointer/non-array type).
static bool isScalarVariable(SgInitializedName* var) {
    if (!var) return false;
    SgType* type = var->get_type();
    if (!type) return false;
    type = type->stripType(SgType::STRIP_MODIFIER_TYPE | SgType::STRIP_TYPEDEF_TYPE);
    if (isSgPointerType(type) || isSgArrayType(type)) return false;
    return true;
}

static bool isSharedScalarStorage(SgInitializedName* var) {
    if (!var) return false;
    if (var->get_declaration() && SageInterface::isStatic(var->get_declaration())) {
        return true;
    }
    return isSgGlobal(var->get_scope()) != nullptr;
}

static bool hasUnprovenScalarWrites(SgForStatement* loop,
                                    const loomX::LoopSummary& summary) {
    SgStatement* loopBody = loop ? loop->get_loop_body() : nullptr;
    if (!loopBody) return false;

    SgInitializedName* loopVar = SageInterface::getLoopIndexVariable(loop);
    std::set<SgInitializedName*> reductionVars = summary.getReductionVariables();
    std::vector<SgNode*> readRefs, writeRefs;
    SageInterface::collectReadWriteRefs(loopBody, readRefs, writeRefs);

    for (SgNode* ref : writeRefs) {
        SgVarRefExp* varRef = isSgVarRefExp(ref);
        if (!varRef) continue;
        SgInitializedName* var = varRef->get_symbol()->get_declaration();
        if (!var || var == loopVar || reductionVars.count(var)) continue;
        if (!isScalarVariable(var)) continue;

        SgScopeStatement* scope = var->get_scope();
        SgNode* current = scope;
        bool declaredInsideLoop = false;
        while (current && !isSgFunctionDefinition(current)) {
            if (current == loopBody) {
                declaredInsideLoop = true;
                break;
            }
            current = current->get_parent();
        }
        if (!declaredInsideLoop) return true;
    }
    return false;
}

static bool hasScalarLiveOut(SgForStatement* loop) {
    if (!loop) return false;
    SgStatement* body = loop->get_loop_body();
    if (!body) return false;

    SgFunctionDefinition* function = nullptr;
    for (SgNode* current = loop; current; current = current->get_parent()) {
        if ((function = isSgFunctionDefinition(current))) break;
    }
    if (!function) return false;

    std::vector<SgNode*> loopReads, loopWrites;
    SageInterface::collectReadWriteRefs(body, loopReads, loopWrites);
    std::set<SgInitializedName*> writtenScalars;
    for (SgNode* ref : loopWrites) {
        SgVarRefExp* varRef = isSgVarRefExp(ref);
        if (!varRef) continue;
        SgInitializedName* var = varRef->get_symbol()->get_declaration();
        if (var && isScalarVariable(var)) writtenScalars.insert(var);
    }
    if (writtenScalars.empty()) return false;

    long loopEnd = loop->get_file_info() ? loop->get_file_info()->get_line() : 0;
    Rose_STL_Container<SgNode*> loopNodes = NodeQuery::querySubTree(loop, V_SgNode);
    for (SgNode* node : loopNodes) {
        if (node->get_file_info()) {
            loopEnd = std::max(loopEnd,
                               static_cast<long>(node->get_file_info()->get_line()));
        }
    }

    std::vector<SgNode*> functionReads, functionWrites;
    SageInterface::collectReadWriteRefs(function->get_body(), functionReads, functionWrites);
    for (SgNode* ref : functionReads) {
        SgVarRefExp* varRef = isSgVarRefExp(ref);
        if (!varRef) continue;
        SgInitializedName* var = varRef->get_symbol()->get_declaration();
        if (!var || !writtenScalars.count(var)) continue;
        if (varRef->get_file_info() && varRef->get_file_info()->get_line() > loopEnd) return true;
    }
    return false;
}

// Return the innermost SgStatement that contains the given AST node.
static SgStatement* getEnclosingStatement(SgNode* node) {
    while (node && !isSgStatement(node)) {
        node = node->get_parent();
    }
    return isSgStatement(node);
}

// Return the name of the function that contains the given node, or an empty
// string if the node is not inside a function definition.
static std::string getEnclosingFunctionName(SgNode* node) {
    while (node) {
        if (SgFunctionDefinition* def = isSgFunctionDefinition(node)) {
            SgFunctionDeclaration* decl = def->get_declaration();
            if (decl) return decl->get_name().getString();
            return "";
        }
        node = node->get_parent();
    }
    return "";
}

// Check for loop-carried dependences through scalar variables that are not
// recognized as reductions.
//
// A scalar declared outside the loop and written inside the loop body is safe
// to make private/lastprivate if:
//   - it has exactly one write statement in the loop body,
//   - no read of the variable occurs before that write in source order,
//   - the write statement is not inside a conditional/switch (so it dominates
//     the rest of the iteration).
// Otherwise the scalar is treated as carrying a loop-carried dependence.
static bool hasScalarLoopCarriedDependence(SgForStatement* loop,
                                           const loomX::LoopSummary& summary,
                                           std::string& description) {
    SgInitializedName* loopVar = SageInterface::getLoopIndexVariable(loop);
    SgStatement* loopBody = loop->get_loop_body();
    if (!loopBody) return false;

    std::set<SgInitializedName*> reductionVars = summary.getReductionVariables();

    std::vector<SgNode*> readRefs, writeRefs;
    SageInterface::collectReadWriteRefs(loopBody, readRefs, writeRefs);

    // Group candidate scalar variables by their declaration.
    std::set<SgInitializedName*> candidateVars;
    for (SgNode* ref : writeRefs) {
        SgVarRefExp* varRef = isSgVarRefExp(ref);
        if (!varRef) continue;
        SgInitializedName* var = varRef->get_symbol()->get_declaration();
        if (!var) continue;

        // Skip loop index variables.
        if (var == loopVar) continue;

        // Skip nested loop index variables.
        Rose_STL_Container<SgNode*> nestedLoops =
            NodeQuery::querySubTree(loopBody, V_SgForStatement);
        bool isNestedLoopVar = false;
        for (SgNode* nlNode : nestedLoops) {
            SgForStatement* nestedLoop = isSgForStatement(nlNode);
            if (!nestedLoop) continue;
            SgInitializedName* nestedVar = SageInterface::getLoopIndexVariable(nestedLoop);
            if (var == nestedVar) {
                isNestedLoopVar = true;
                break;
            }
        }
        if (isNestedLoopVar) continue;

        // Skip reduction variables.
        if (reductionVars.find(var) != reductionVars.end()) continue;

        // Static locals and globals retain one storage location across all
        // iterations. They cannot be made private by an OpenMP clause.
        if (isSharedScalarStorage(var) &&
            !summary.pragmas.hasDataSharingClauses &&
            !summary.pragmas.hasReduction && !summary.pragmas.hasThreadprivate) {
            description = "shared scalar storage may race: " +
                          var->get_name().getString();
            return true;
        }

        // Skip variables declared inside the loop body.
        SgScopeStatement* varScope = var->get_scope();
        bool declaredInsideLoop = false;
        SgNode* currentScope = varScope;
        while (currentScope && !isSgFunctionDefinition(currentScope)) {
            if (currentScope == loopBody) {
                declaredInsideLoop = true;
                break;
            }
            currentScope = currentScope->get_parent();
        }
        if (declaredInsideLoop) continue;

        // Skip non-scalars (arrays/pointers are handled by array dependence).
        if (!isScalarVariable(var)) continue;

        candidateVars.insert(var);
    }

    for (SgInitializedName* var : candidateVars) {
        // Collect all read and write references to this variable, together with
        // their enclosing statements, in source order.
        struct Ref {
            SgNode* node;
            SgStatement* stmt;
            bool isWrite;
        };
        std::vector<Ref> refs;
        auto collect = [&](const std::vector<SgNode*>& src, bool isWrite) {
            for (SgNode* node : src) {
                SgVarRefExp* varRef = isSgVarRefExp(node);
                if (!varRef) continue;
                if (varRef->get_symbol()->get_declaration() != var) continue;
                SgStatement* stmt = getEnclosingStatement(node);
                if (!stmt) continue;
                refs.push_back({node, stmt, isWrite});
            }
        };
        collect(readRefs, false);
        collect(writeRefs, true);

        std::sort(refs.begin(), refs.end(),
                  [](const Ref& a, const Ref& b) {
                      Sg_File_Info* fa = a.stmt->get_file_info();
                      Sg_File_Info* fb = b.stmt->get_file_info();
                      if (!fa || !fb) return a.stmt < b.stmt;
                      if (fa->get_line() != fb->get_line())
                          return fa->get_line() < fb->get_line();
                      return fa->get_col() < fb->get_col();
                  });

        // Find the first write statement.
        auto firstWriteIt = std::find_if(
            refs.begin(), refs.end(), [](const Ref& r) { return r.isWrite; });
        if (firstWriteIt == refs.end()) continue;

        // Any read before the first write means the read may see a value from a
        // previous iteration.
        for (auto it = refs.begin(); it != firstWriteIt; ++it) {
            if (!it->isWrite) {
                description = "loop-carried scalar dependence on " +
                              var->get_name().getString();
                return true;
            }
        }

        // Require exactly one write statement. Multiple writes make it hard to
        // guarantee the variable is iteration-private.
        std::set<SgStatement*> writeStmts;
        for (const Ref& r : refs) {
            if (r.isWrite) writeStmts.insert(r.stmt);
        }
        if (writeStmts.size() != 1) {
            description = "loop-carried scalar dependence on " +
                          var->get_name().getString();
            return true;
        }

        // The write must dominate the iteration: it cannot be inside a
        // conditional or switch, otherwise a read later in the body could see
        // a value from a previous iteration.
        SgStatement* writeStmt = *writeStmts.begin();
        SgNode* parent = writeStmt->get_parent();
        while (parent && parent != loopBody) {
            if (isSgIfStmt(parent) || isSgSwitchStatement(parent) ||
                isSgConditionalExp(parent)) {
                description = "loop-carried scalar dependence on " +
                              var->get_name().getString();
                return true;
            }
            parent = parent->get_parent();
        }
    }

    return false;
}

// Fill private-variable information for the summary.
// The loop-index variable of the parallelized loop is implicitly private in
// OpenMP parallel-for and must not be listed when it is declared in the
// for-init statement, because the pragma is inserted outside the loop's scope.
// Variables declared inside the loop body or any nested block are already
// local to each iteration and are also out of scope at the pragma location, so
// they are skipped as well.  Reduction variables use the reduction clause
// instead.
//
// Nested loop index variables (e.g. the inner 'i' when parallelizing the outer
// 'j' loop) must be made private explicitly when they are declared at function
// scope, otherwise concurrent threads share the same induction variable and
// races occur.
void fillPrivateVars(SgForStatement* loop, loomX::LoopSummary& summary) {
    SgStatement* loopBody = loop->get_loop_body();
    if (!loopBody) return;

    SgInitializedName* loopVar = SageInterface::getLoopIndexVariable(loop);
    std::set<SgInitializedName*> reductionVars = summary.getReductionVariables();

    // Collect nested loop index variables; they may need an explicit private
    // clause if they are declared at function scope.
    std::set<SgInitializedName*> nestedLoopVars;
    Rose_STL_Container<SgNode*> nestedLoops =
        NodeQuery::querySubTree(loopBody, V_SgForStatement);
    for (SgNode* nlNode : nestedLoops) {
        SgForStatement* nestedLoop = isSgForStatement(nlNode);
        if (!nestedLoop) continue;
        SgInitializedName* nestedVar = SageInterface::getLoopIndexVariable(nestedLoop);
        if (nestedVar) nestedLoopVars.insert(nestedVar);
    }

    std::vector<SgNode*> readRefs, writeRefs;
    SageInterface::collectReadWriteRefs(loopBody, readRefs, writeRefs);

    for (SgNode* ref : writeRefs) {
        SgVarRefExp* varRef = isSgVarRefExp(ref);
        if (!varRef) continue;
        SgInitializedName* var = varRef->get_symbol()->get_declaration();
        if (!var) continue;

        // Skip the parallelized loop's own index variable (handled implicitly
        // by OpenMP).
        if (var == loopVar) continue;

        // Skip reduction variables (handled by reduction clause).
        if (reductionVars.find(var) != reductionVars.end()) continue;

        // Skip variables declared inside the loop body or any nested block
        // within it. They are already per-iteration and must not appear in an
        // OpenMP private clause (the variable name is not in scope there).
        SgScopeStatement* varScope = var->get_scope();
        bool declaredInsideLoop = false;
        SgNode* currentScope = varScope;
        while (currentScope && !isSgFunctionDefinition(currentScope)) {
            if (currentScope == loopBody) {
                declaredInsideLoop = true;
                break;
            }
            currentScope = currentScope->get_parent();
        }
        if (declaredInsideLoop) continue;

        // Only privatise scalars. Arrays/pointers belong in map clauses for GPU
        // or are shared for CPU loops.
        if (!isScalarVariable(var)) continue;

        summary.privateVars.insert(var);
    }

    // Add function-scoped nested loop index variables to the private clause.
    for (SgInitializedName* nestedVar : nestedLoopVars) {
        if (nestedVar == loopVar) continue;
        if (reductionVars.find(nestedVar) != reductionVars.end()) continue;
        if (!isScalarVariable(nestedVar)) continue;

        // If the nested loop variable is declared inside the parallel loop body
        // it is already local to each thread; no clause needed.
        SgScopeStatement* varScope = nestedVar->get_scope();
        bool declaredInsideLoop = false;
        SgNode* currentScope = varScope;
        while (currentScope && !isSgFunctionDefinition(currentScope)) {
            if (currentScope == loopBody) {
                declaredInsideLoop = true;
                break;
            }
            currentScope = currentScope->get_parent();
        }
        if (declaredInsideLoop) continue;

        summary.privateVars.insert(nestedVar);
    }
}

// Fill map-clause information for GPU target regions.
void fillMapVariables(SgForStatement* loop, loomX::LoopSummary& summary) {
    std::set<SgInitializedName*> allVars;
    Rose_STL_Container<SgNode*> varRefs =
        NodeQuery::querySubTree(loop, V_SgVarRefExp);
    for (SgNode* node : varRefs) {
        SgVarRefExp* varRef = isSgVarRefExp(node);
        if (!varRef) continue;
        SgInitializedName* var = varRef->get_symbol()->get_declaration();
        if (var) allVars.insert(var);
    }

    std::vector<SgNode*> readRefs, writeRefs;
    SageInterface::collectReadWriteRefs(loop, readRefs, writeRefs);

    std::set<SgInitializedName*> readVars, writeVars;
    for (SgNode* ref : readRefs) {
        SgVarRefExp* varRef = isSgVarRefExp(ref);
        if (!varRef) continue;
        SgInitializedName* var = varRef->get_symbol()->get_declaration();
        if (var) readVars.insert(var);
    }
    for (SgNode* ref : writeRefs) {
        SgVarRefExp* varRef = isSgVarRefExp(ref);
        if (!varRef) continue;
        SgInitializedName* var = varRef->get_symbol()->get_declaration();
        if (var) writeVars.insert(var);
    }

    for (SgInitializedName* var : allVars) {
        // Skip loop index variable - it's handled implicitly.
        SgInitializedName* loopVar = SageInterface::getLoopIndexVariable(loop);
        if (var == loopVar) continue;

        // Skip nested loop index variables.
        Rose_STL_Container<SgNode*> nestedLoops =
            NodeQuery::querySubTree(loop->get_loop_body(), V_SgForStatement);
        bool isNestedLoopVar = false;
        for (SgNode* nlNode : nestedLoops) {
            SgForStatement* nestedLoop = isSgForStatement(nlNode);
            if (!nestedLoop) continue;
            SgInitializedName* nestedVar = SageInterface::getLoopIndexVariable(nestedLoop);
            if (var == nestedVar) {
                isNestedLoopVar = true;
                break;
            }
        }
        if (isNestedLoopVar) continue;

        // Skip variables declared inside the loop body or any nested block.
        SgScopeStatement* varScope = var->get_scope();
        SgStatement* loopBody = loop->get_loop_body();
        if (loopBody) {
            bool declaredInsideLoop = false;
            SgNode* currentScope = varScope;
            while (currentScope && !isSgFunctionDefinition(currentScope)) {
                if (currentScope == loopBody) {
                    declaredInsideLoop = true;
                    break;
                }
                currentScope = currentScope->get_parent();
            }
            if (declaredInsideLoop) continue;
        }

        bool isRead = readVars.find(var) != readVars.end();
        bool isWritten = writeVars.find(var) != writeVars.end();

        if (isRead && isWritten) {
            summary.mapClauses.insert({var, "tofrom"});
        } else if (isRead) {
            summary.mapClauses.insert({var, "to"});
        } else if (isWritten) {
            summary.mapClauses.insert({var, "from"});
        } else {
            summary.mapClauses.insert({var, "tofrom"});
        }
    }

    // Reduction variables are handled by the reduction clause, not map.
    std::set<SgInitializedName*> reductionVars = summary.getReductionVariables();
    for (SgInitializedName* redVar : reductionVars) {
        auto it = summary.mapClauses.begin();
        while (it != summary.mapClauses.end()) {
            if (it->first == redVar) {
                it = summary.mapClauses.erase(it);
            } else {
                ++it;
            }
        }
    }
}

// Build a complete LoopSummary for a loop, including interprocedural safety.
loomX::LoopSummary buildSummary(SgForStatement* loop,
                                GpuProfitability& profitability,
                                InterproceduralAnalysis& ipa,
                                SgInitializedName* loopVar) {
    loomX::LoopSummary summary = profitability.summarize(loop);
    PragmaAnalysis pragmaAnalysis;
    summary.pragmas = pragmaAnalysis.analyze(loop);

    // Interprocedural safety for function calls inside the loop.
    Rose_STL_Container<SgNode*> calls =
        NodeQuery::querySubTree(loop, V_SgFunctionCallExp);
    summary.hasFunctionCalls = !calls.empty();
    summary.allFunctionCallsSafe = true;
    for (SgNode* node : calls) {
        SgFunctionCallExp* call = isSgFunctionCallExp(node);
        if (call && !ipa.isSafeForParallelLoop(call, loopVar)) {
            summary.allFunctionCallsSafe = false;
            break;
        }
    }

    // Add callee work estimates to the loop's intensity.  The local body
    // analysis does not look inside function calls, so safe leaf/pure callees
    // would otherwise appear as trivially memory-bound.
    for (SgNode* node : calls) {
        SgFunctionCallExp* call = isSgFunctionCallExp(node);
        if (!call) continue;

        SgFunctionDeclaration* calleeDecl = call->getAssociatedFunctionDeclaration();
        if (!calleeDecl) continue;
        const FunctionSummary* callee = ipa.getSummary(calleeDecl->get_name().getString());
        if (!callee || callee->estimatedFlops < 0) continue;

        // Compute the product of trip counts for all loops enclosing the call
        // up to and including the target loop.
        long long tripProduct = 1;
        SgNode* current = call;
        while (current && current != loop) {
            if (SgForStatement* enclosing = isSgForStatement(current)) {
                long trip = profitability.getIterationEstimator().estimate(enclosing, false);
                if (trip > 0) tripProduct *= trip;
            }
            current = current->get_parent();
        }
        // The target loop itself is not visited by the walk above; account for it.
        long targetTrip = profitability.getIterationEstimator().estimate(loop, false);
        if (targetTrip > 0) tripProduct *= targetTrip;

        summary.intensity.flopCount += callee->estimatedFlops * tripProduct;
        summary.intensity.memoryOpCount += callee->estimatedMemOps * tripProduct;
    }

    // Recompute intensity classification and target decision now that callee
    // work has been included.  The initial summarize() decided the target
    // before interprocedural estimates were available.
    double computeBoundThreshold = profitability.getConfig().computeBoundThreshold;
    if (summary.intensity.memoryOpCount > 0) {
        summary.intensity.flopsPerMemoryOp =
            static_cast<double>(summary.intensity.flopCount) /
            static_cast<double>(summary.intensity.memoryOpCount);
        if (summary.intensity.hasHeavyMath ||
            summary.intensity.flopsPerMemoryOp >= computeBoundThreshold) {
            summary.intensity.classification = IntensityClass::COMPUTE_BOUND;
            summary.intensity.note = "High FLOP/memory ratio; compute-bound";
        } else if (summary.intensity.flopsPerMemoryOp >=
                   computeBoundThreshold / 4.0) {
            summary.intensity.classification = IntensityClass::BALANCED;
            summary.intensity.note = "Balanced compute and memory";
        } else {
            summary.intensity.classification = IntensityClass::MEMORY_BOUND;
            summary.intensity.note = "Low FLOP/memory ratio; memory-bound";
        }
    }
    profitability.reevaluateTarget(summary);

    // Fill codegen inputs.
    fillPrivateVars(loop, summary);

    return summary;
}

enum class TranslationMode {
    CPU_ONLY,        // Force CPU OpenMP for all safe loops.
    GPU_NAIVE,       // Offload every safe loop to GPU (profitability gate off).
    GPU_PROFITABLE,  // Use profitability model to choose CPU vs GPU (default).
    ANALYZE_ONLY,    // Report per-loop verdicts without transforming the source.
};

int main(int argc, char* argv[]) {
    ROSE_INITIALIZE;

    bool verbose = false;
    bool intraproceduralBaseline = false;
    bool scalarDepCheck = true;
    bool strictRaceSafety = false;
    bool phaseCouple = true;
    TranslationMode mode = TranslationMode::GPU_PROFITABLE;
    loomX::CodeGenBackend backend = loomX::CodeGenBackend::OMP;
    loomX::ProfitabilityConfig config;
    std::string explicitOutputFile;
    std::vector<std::string> args;
    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        if (arg == "-v" || arg == "--verbose") {
            verbose = true;
        } else if (arg == "--intraprocedural-baseline") {
            intraproceduralBaseline = true;
        } else if (arg == "--no-scalar-dep-check") {
            scalarDepCheck = false;
        } else if (arg == "--strict-race-safety") {
            strictRaceSafety = true;
        } else if (arg == "--cpu-only") {
            mode = TranslationMode::CPU_ONLY;
        } else if (arg == "--gpu-naive") {
            mode = TranslationMode::GPU_NAIVE;
        } else if (arg == "--gpu-profitable") {
            mode = TranslationMode::GPU_PROFITABLE;
        } else if (arg == "--backend" && i + 1 < argc) {
            std::string b = argv[++i];
            if (b == "omp") {
                backend = loomX::CodeGenBackend::OMP;
            } else if (b == "cuda") {
                backend = loomX::CodeGenBackend::CUDA;
            } else if (b == "opencl") {
                backend = loomX::CodeGenBackend::OPENCL;
            } else if (b == "openacc") {
                backend = loomX::CodeGenBackend::OPENACC;
            } else {
                std::cerr << "Unknown backend '" << b
                          << "' (expected omp|cuda|opencl|openacc)\n";
                return 1;
            }
        } else if (arg == "--no-phase-couple") {
            phaseCouple = false;
        } else if (arg == "--analyze-only") {
            mode = TranslationMode::ANALYZE_ONLY;
        } else if (arg == "--min-gpu-speedup" && i + 1 < argc) {
            config.minGpuSpeedup = std::stod(argv[++i]);
        } else if (arg == "--min-nested-flop" && i + 1 < argc) {
            config.minNestedFlopForGPU = std::stoll(argv[++i]);
        } else if (arg == "--min-total-flop" && i + 1 < argc) {
            config.minTotalFlopForGPU = std::stoll(argv[++i]);
        } else if (arg == "--min-cpu-openmp-flop" && i + 1 < argc) {
            config.minTotalFlopForCPUOpenMP = std::stoll(argv[++i]);
        } else if (arg == "--compute-bound-threshold" && i + 1 < argc) {
            config.computeBoundThreshold = std::stod(argv[++i]);
        } else if (arg == "--gpu-compute-efficiency" && i + 1 < argc) {
            config.gpuComputeEfficiency = std::stod(argv[++i]);
        } else if (arg == "--gpu-compute-throughput" && i + 1 < argc) {
            config.gpuComputeThroughput = std::stod(argv[++i]);
        } else if (arg == "--gpu-mem-bandwidth" && i + 1 < argc) {
            config.gpuMemoryBandwidth = std::stod(argv[++i]);
        } else if (arg == "--cpu-mem-bandwidth" && i + 1 < argc) {
            config.cpuMemoryBandwidth = std::stod(argv[++i]);
        } else if (arg == "--cpu-core-count" && i + 1 < argc) {
            config.cpuCoreCount = std::stoi(argv[++i]);
        } else if (arg == "--pcie-bandwidth" && i + 1 < argc) {
            config.pcieBandwidth = std::stod(argv[++i]);
        } else if (arg == "--pcie-latency" && i + 1 < argc) {
            config.pcieLatency = std::stod(argv[++i]);
        } else if (arg == "--kernel-launch-overhead" && i + 1 < argc) {
            config.kernelLaunchOverhead = std::stod(argv[++i]);
        } else if (arg == "--heavy-math-cost" && i + 1 < argc) {
            config.heavyMathCostFactor = std::stod(argv[++i]);
        } else if (arg == "--pcie-factor" && i + 1 < argc) {
            config.pcieTransferFactor = std::stod(argv[++i]);
        } else if (arg == "--cpu-cache-reuse" && i + 1 < argc) {
            config.cpuCacheReuseFactor = std::stod(argv[++i]);
        } else if (arg == "--gpu-reduction-overhead" && i + 1 < argc) {
            config.gpuReductionOverhead = std::stod(argv[++i]);
        } else if ((arg == "-o" || arg == "--output") && i + 1 < argc) {
            explicitOutputFile = argv[++i];
        } else {
            args.push_back(arg);
        }
    }

    if (args.empty()) {
        std::cerr << "Usage: " << argv[0]
                  << " [-v|--verbose] [--intraprocedural-baseline] [--no-scalar-dep-check]"
                  << " [--strict-race-safety]"
                  << " [--cpu-only|--gpu-naive|--gpu-profitable|--analyze-only]"
                  << " [--backend omp|cuda|opencl|openacc]"
                  << " [--no-phase-couple]"
                  << " [--min-gpu-speedup <f>] [--min-nested-flop <n>]"
                  << " [--min-total-flop <n>] [--min-cpu-openmp-flop <n>]"
                  << " [--compute-bound-threshold <f>]"
                  << " [--pcie-factor <f>] [--gpu-compute-throughput <f>]"
                  << " [--gpu-mem-bandwidth <f>] [--pcie-bandwidth <f>]"
                  << " [--cpu-core-count <n>] [--cpu-mem-bandwidth <f>]"
                  << " [--kernel-launch-overhead <f>] [--pcie-latency <f>]"
                  << " <input.c> [-o output.c]\n";
        return 1;
    }

    // Rebuild argc/argv for ROSE frontend (it expects a raw argv array).
    std::vector<char*> frontendArgv;
    frontendArgv.push_back(argv[0]);
    for (auto& a : args) frontendArgv.push_back(const_cast<char*>(a.c_str()));
    int frontendArgc = static_cast<int>(frontendArgv.size());

    // Build the AST
    SgProject* project = frontend(frontendArgc, frontendArgv.data());
    if (!project) {
        std::cerr << "Failed to parse input\n";
        return 1;
    }

    // Honor -o / --output explicitly so the harness can name outputs.
    if (!explicitOutputFile.empty()) {
        SgFilePtrList& files = project->get_fileList();
        if (!files.empty()) {
            SgSourceFile* firstFile = isSgSourceFile(files[0]);
            if (firstFile) {
                firstFile->set_unparse_output_filename(explicitOutputFile);
            }
        }
    }

    // Step 1: Interprocedural analysis
    std::cout << "=== Phase 1: Interprocedural Analysis ===\n";
    InterproceduralAnalysis ipa;
    ipa.setIntraproceduralBaseline(intraproceduralBaseline);
    ipa.analyzeProject(project);
    if (verbose || intraproceduralBaseline) {
        std::cout << "\n";
        ipa.printSummaries();
    }

    // Step 2: Find all loops
    std::cout << "\n=== Phase 2: Loop Discovery ===\n";
    LoopCollector collector;
    collector.traverse(project, preorder);
    std::cout << "Found " << collector.loops.size() << " for-loops\n";

    // Step 3: Analyze and transform each loop
    std::cout << "\n=== Phase 3: Parallelization ===\n";
    GpuProfitability profitability(config);
    std::unique_ptr<CodeGen> codegen;
    std::unique_ptr<loomX::KernelCodeGen> kernelCodegen;
    if (backend == loomX::CodeGenBackend::CUDA ||
        backend == loomX::CodeGenBackend::OPENCL) {
        // Kernel-extraction backends: GPU loops become device kernels; loops
        // that cannot be kernelized fall back to a CPU OpenMP pragma.
        kernelCodegen.reset(new loomX::KernelCodeGen(backend));
        codegen.reset(new OpenMPCodeGen());
    } else if (backend == loomX::CodeGenBackend::OPENACC) {
        codegen.reset(new OpenACCCodeGen());
    } else {
        codegen.reset(new OpenMPCodeGen());
    }
    bool openmpInOutput = false;  // CPU-OpenMP pragmas beside device kernels

    int parallelized = 0;
    int skipped = 0;
    std::set<SgForStatement*> parallelizedLoops;

    // Loops accepted for transformation, in program order, with a flag marking
    // helper (init_array) loops that should only be transformed if the
    // phase-coupling pass later promotes them to the GPU.
    std::vector<std::pair<loomX::LoopSummary, bool>> accepted;
    std::set<SgForStatement*> selectedLoops;

    for (SgForStatement* loop : collector.loops) {
        std::cout << "\nProcessing loop at line "
                  << loop->get_file_info()->get_line() << "\n";

        // Skip initialization / output helpers in benchmark suites; their
        // correctness matters for end-to-end validation and they frequently
        // contain reduction patterns (e.g. PolyBench init_array) that are not
        // the target of kernel parallelization.  init_array loops are kept as
        // candidates so the phase-coupling pass can offload them when a later
        // GPU loop consumes the data they write; output helpers are never
        // touched.
        std::string funcName = getEnclosingFunctionName(loop);
        if (funcName == "print_array") {
            std::cout << "  -> Skipped: loop inside " << funcName << "\n";
            skipped++;
            continue;
        }
        bool initHelperLoop = (funcName == "init_array");

        // Skip loops nested inside already-parallelized loops
        bool nested = false;
        SgNode* parent = loop->get_parent();
        while (parent) {
            if (SgForStatement* parentLoop = isSgForStatement(parent)) {
                if (selectedLoops.find(parentLoop) != selectedLoops.end()) {
                    std::cout << "  -> Skipped: nested inside already-parallelized loop\n";
                    nested = true;
                    break;
                }
            }
            parent = parent->get_parent();
        }
        if (nested) {
            skipped++;
            continue;
        }

        // Check for loop-carried dependences before parallelizing.
        SgInitializedName* loopVar = SageInterface::getLoopIndexVariable(loop);
        LoopDependenceAnalysis depAnalysis;
        DependenceResult depResult = depAnalysis.analyze(loop);
        if (depResult.hasLoopCarriedDependence) {
            std::cout << "  -> Skipped: " << depResult.description << "\n";
            skipped++;
            continue;
        }

        // Build the full loop summary (analyses + decision + codegen inputs).
        loomX::LoopSummary summary = buildSummary(loop, profitability, ipa, loopVar);

        if (summary.pragmas.blocksTransformation) {
            std::cout << "  -> Skipped: " << summary.pragmas.reason << "\n";
            skipped++;
            continue;
        }

        // Reject loops with scalar loop-carried dependences that are not
        // recognized reductions (unless disabled for lenient evaluation).
        if (scalarDepCheck) {
            std::string scalarDepDesc;
            if (hasScalarLoopCarriedDependence(loop, summary, scalarDepDesc)) {
                std::cout << "  -> Skipped: " << scalarDepDesc << "\n";
                skipped++;
                continue;
            }
        }

        // Reject loops with unsafe function calls.
        if (summary.hasFunctionCalls && !summary.allFunctionCallsSafe) {
            std::cout << "  -> Function call not safe for parallelization\n";
            skipped++;
            continue;
        }
        if (summary.hasFunctionCalls && summary.allFunctionCallsSafe) {
            std::cout << "  -> All function calls are safe (interprocedural analysis)\n";
        }

        if (strictRaceSafety && !summary.reductions.empty() &&
            !summary.pragmas.hasReduction) {
            std::cout << "  -> Skipped: strict race safety requires explicit reduction proof\n";
            skipped++;
            continue;
        }
        if (strictRaceSafety && hasUnprovenScalarWrites(loop, summary) &&
            !summary.pragmas.hasDataSharingClauses &&
            !summary.pragmas.hasThreadprivate && !summary.pragmas.hasReduction) {
            std::cout << "  -> Skipped: strict race safety requires explicit scalar privatization proof\n";
            skipped++;
            continue;
        }
        if (strictRaceSafety && hasScalarLiveOut(loop) &&
            !summary.pragmas.hasLastprivate) {
            std::cout << "  -> Skipped: strict race safety requires lastprivate for live-out scalar\n";
            skipped++;
            continue;
        }

        // Apply translation-mode override for ablation experiments.  init_array
        // helper loops are excluded: phase-coupling is a profitable-pipeline
        // concept, so the ablation modes keep their previous (untouched)
        // behavior for helper loops.
        if (!initHelperLoop) {
            if (mode == TranslationMode::CPU_ONLY) {
                if (summary.target != ParallelTarget::SEQUENTIAL) {
                    summary.target = ParallelTarget::CPU_OPENMP;
                }
            } else if (mode == TranslationMode::GPU_NAIVE) {
                // Ablation mode: bypass the profitability gate entirely.  Every
                // loop that passed the safety checks is offloaded, including
                // ones the cost model marked SEQUENTIAL (e.g. below the
                // CPU-OpenMP FLOP threshold).
                summary.target = ParallelTarget::GPU_OFFLOAD;
            }
        }

        // Compute data-motion clauses for any loop that will actually be
        // offloaded.  This runs after the mode override so that ablation runs
        // (--gpu-naive) get correct maps for loops the cost model would have
        // kept on the CPU.
        if (summary.target == ParallelTarget::GPU_OFFLOAD) {
            fillMapVariables(loop, summary);
        }

        // Apply the cost-model decision.
        if (summary.target == ParallelTarget::SEQUENTIAL) {
            if (!initHelperLoop) {
                if (mode == TranslationMode::ANALYZE_ONLY) {
                    std::cout << "REJECTED line "
                              << loop->get_file_info()->get_line()
                              << ": not profitable / not safe\n";
                } else {
                    std::cout << "  -> Not profitable to parallelize\n";
                }
                skipped++;
                continue;
            }
            // init_array loop: keep as coupling candidate.  It only receives a
            // pragma if phase-coupling later promotes it to the GPU.
            accepted.push_back({summary, true});
            continue;
        }

        // In analyze-only mode we report the verdict and stop.
        if (mode == TranslationMode::ANALYZE_ONLY) {
            std::cout << "PARALLELIZED line "
                      << loop->get_file_info()->get_line();
            if (summary.target == ParallelTarget::GPU_OFFLOAD) {
                std::cout << " GPU_OFFLOAD";
            } else {
                std::cout << " CPU_OPENMP";
            }
            std::cout << "\n";
            parallelized++;
            continue;
        }

        // Defer code generation until the phase-coupling pass has run (so it
        // can promote init_array loops to GPU and the same-function target-data
        // hoister can merge them with the consuming GPU loops).
        selectedLoops.insert(loop);
        accepted.push_back({summary, initHelperLoop});
    }

    // Phase-couple init loops with subsequent GPU loops, then emit all
    // transformed sources in program order.  Coupling is a profitable-pipeline
    // rule; the ablation modes keep their previous, untouched behavior for
    // init_array helpers.
    if (mode == TranslationMode::GPU_PROFITABLE && phaseCouple &&
        !accepted.empty()) {
        std::vector<loomX::LoopSummary> summaries;
        summaries.reserve(accepted.size());
        for (const auto& entry : accepted) summaries.push_back(entry.first);
        profitability.phaseCoupleInitLoops(summaries);

        for (size_t i = 0; i < accepted.size(); ++i) {
            accepted[i].first = summaries[i];
        }
    }

    for (auto& entry : accepted) {
            loomX::LoopSummary& summary = entry.first;
            bool coupleOnly = entry.second;

            // Untouched helper loop: never emitted a pragma before, keep it.
            if (coupleOnly && summary.target != ParallelTarget::GPU_OFFLOAD) {
                continue;
            }

            if (summary.target == ParallelTarget::GPU_OFFLOAD &&
                summary.mapClauses.empty()) {
                fillMapVariables(summary.loop, summary);
            }
            if (summary.target == ParallelTarget::GPU_OFFLOAD) {
                summary.collapseDepth = profitability.collapseDepthFor(summary.loop);
            }

// CUDA/OpenCL backends: extract GPU-offload loops into device
            // kernels; loops that cannot be kernelized fall back to the CPU.
            if (kernelCodegen && summary.target == ParallelTarget::GPU_OFFLOAD) {
                if (kernelCodegen->generate(summary.loop, summary)) {
                    parallelizedLoops.insert(summary.loop);
                    std::cout << "  -> "
                              << (backend == loomX::CodeGenBackend::CUDA
                                      ? "CUDA" : "OpenCL")
                              << " kernel extracted\n";
                    parallelized++;
                    continue;
                }
                summary.target = ParallelTarget::CPU_OPENMP;
                codegen->generatePragmas(summary);
                parallelizedLoops.insert(summary.loop);
                openmpInOutput = true;
                std::cout << "  -> fallback: CPU OpenMP pragma inserted\n";
                parallelized++;
                continue;
            }

            // Insert the selected backend's directives based on the summary.
            codegen->generatePragmas(summary);
            parallelizedLoops.insert(summary.loop);

            if (summary.target == ParallelTarget::CPU_OPENMP) {
                openmpInOutput = true;
            }

            if (summary.target == ParallelTarget::GPU_OFFLOAD) {
                std::cout << "  -> GPU offload pragma inserted\n";
            } else {
                std::cout << "  -> CPU OpenMP pragma inserted\n";
            }
            parallelized++;
    }

    std::cout << "\n=== Summary ===\n";
    std::cout << "Total loops: " << collector.loops.size() << "\n";
    std::cout << "Parallelized: " << parallelized << "\n";
    std::cout << "Skipped: " << skipped << "\n";

    // Generate output (skipped in analyze-only mode).
    if (mode != TranslationMode::ANALYZE_ONLY) {
        project->unparse();
    }

    // Post-process the unparsed source:
    //   1. Hoist target data regions around consecutive GPU loops.
    //   2. Prepend #include <omp.h> if not already present.
    if (mode != TranslationMode::ANALYZE_ONLY && parallelized > 0) {
        SgFilePtrList& files = project->get_fileList();
        if (!files.empty()) {
            SgSourceFile* firstFile = isSgSourceFile(files[0]);
            if (firstFile) {
                std::string outputName = firstFile->get_unparse_output_filename();
                if (outputName.empty()) {
                    outputName = "rose_" + firstFile->get_sourceFileNameWithoutPath();
                }
                std::ifstream inFile(outputName);
                if (inFile) {
                    std::string content((std::istreambuf_iterator<char>(inFile)),
                                         std::istreambuf_iterator<char>());
                    inFile.close();

                    if (kernelCodegen && kernelCodegen->kernelCount() > 0) {
                        // CUDA/OpenCL: splice kernel + launcher definitions at
                        // the bottom, prototypes near the top, and the device
                        // runtime include (plus <omp.h> for mixed-target loops).
                        kernelCodegen->finalizeOutput(content, openmpInOutput);
                    } else {
                        // Backend-specific post-processing (e.g. target-data
                        // hoisting for OpenMP, header injection for OpenACC).
                        codegen->postProcessSource(content);
                    }

                    std::ofstream outFile(outputName);
                    if (outFile) {
                        outFile << content;
                        outFile.close();
                    }
                }
            }
        }
    }

    return 0;
}
