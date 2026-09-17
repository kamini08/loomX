#include "rose.h"
#include "InterproceduralAnalysis.h"
#include "GpuProfitability.h"
#include "OpenMPCodeGen.h"
#include "LoopSummary.h"
#include "LoopDependenceAnalysis.h"
#include "LoopAnalysisTypes.h"
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>

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

// Fill private-variable information for the summary.
// Loop-index variables are implicitly private in OpenMP parallel-for and must
// not be listed when declared in the for-init statement, because the pragma is
// inserted outside the loop's scope.  Variables declared inside the loop body
// or any nested block are already local to each iteration and are also out of
// scope at the pragma location, so they are skipped as well.  Reduction
// variables use the reduction clause instead.
void fillPrivateVars(SgForStatement* loop, loomX::LoopSummary& summary) {
    SgStatement* loopBody = loop->get_loop_body();
    if (!loopBody) return;

    SgInitializedName* loopVar = SageInterface::getLoopIndexVariable(loop);
    std::set<SgInitializedName*> reductionVars = summary.getReductionVariables();

    // Collect nested loop index variables so we do not privatise them.
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

        // Skip loop index variables (handled implicitly by OpenMP).
        if (var == loopVar) continue;
        if (nestedLoopVars.find(var) != nestedLoopVars.end()) continue;

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
    if (summary.target == ParallelTarget::GPU_OFFLOAD) {
        fillMapVariables(loop, summary);
    }

    return summary;
}

enum class TranslationMode {
    CPU_ONLY,        // Force CPU OpenMP for all safe loops.
    GPU_NAIVE,       // Offload every safe loop to GPU (profitability gate off).
    GPU_PROFITABLE,  // Use profitability model to choose CPU vs GPU (default).
};

int main(int argc, char* argv[]) {
    ROSE_INITIALIZE;

    bool verbose = false;
    bool intraproceduralBaseline = false;
    TranslationMode mode = TranslationMode::GPU_PROFITABLE;
    loomX::ProfitabilityConfig config;
    std::string explicitOutputFile;
    std::vector<std::string> args;
    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        if (arg == "-v" || arg == "--verbose") {
            verbose = true;
        } else if (arg == "--intraprocedural-baseline") {
            intraproceduralBaseline = true;
        } else if (arg == "--cpu-only") {
            mode = TranslationMode::CPU_ONLY;
        } else if (arg == "--gpu-naive") {
            mode = TranslationMode::GPU_NAIVE;
        } else if (arg == "--gpu-profitable") {
            mode = TranslationMode::GPU_PROFITABLE;
        } else if (arg == "--min-gpu-speedup" && i + 1 < argc) {
            config.minGpuSpeedup = std::stod(argv[++i]);
        } else if (arg == "--min-nested-flop" && i + 1 < argc) {
            config.minNestedFlopForGPU = std::stoll(argv[++i]);
        } else if (arg == "--min-total-flop" && i + 1 < argc) {
            config.minTotalFlopForGPU = std::stoll(argv[++i]);
        } else if (arg == "--compute-bound-threshold" && i + 1 < argc) {
            config.computeBoundThreshold = std::stod(argv[++i]);
        } else if (arg == "--gpu-compute-efficiency" && i + 1 < argc) {
            config.gpuComputeEfficiency = std::stod(argv[++i]);
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
                  << " [-v|--verbose] [--intraprocedural-baseline]"
                  << " [--cpu-only|--gpu-naive|--gpu-profitable]"
                  << " [--min-gpu-speedup <f>] [--min-nested-flop <n>]"
                  << " [--min-total-flop <n>] [--compute-bound-threshold <f>]"
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
    if (verbose) {
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
    OpenMPCodeGen codegen;

    int parallelized = 0;
    int skipped = 0;
    std::set<SgForStatement*> parallelizedLoops;

    for (SgForStatement* loop : collector.loops) {
        std::cout << "\nProcessing loop at line "
                  << loop->get_file_info()->get_line() << "\n";

        // Skip loops nested inside already-parallelized loops
        bool nested = false;
        SgNode* parent = loop->get_parent();
        while (parent) {
            if (SgForStatement* parentLoop = isSgForStatement(parent)) {
                if (parallelizedLoops.find(parentLoop) != parallelizedLoops.end()) {
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

        // Reject loops with unsafe function calls.
        if (summary.hasFunctionCalls && !summary.allFunctionCallsSafe) {
            std::cout << "  -> Function call not safe for parallelization\n";
            skipped++;
            continue;
        }
        if (summary.hasFunctionCalls && summary.allFunctionCallsSafe) {
            std::cout << "  -> All function calls are safe (interprocedural analysis)\n";
        }

        // Apply translation-mode override for ablation experiments.
        if (mode == TranslationMode::CPU_ONLY) {
            if (summary.target != ParallelTarget::SEQUENTIAL) {
                summary.target = ParallelTarget::CPU_OPENMP;
            }
        } else if (mode == TranslationMode::GPU_NAIVE) {
            if (summary.target != ParallelTarget::SEQUENTIAL) {
                summary.target = ParallelTarget::GPU_OFFLOAD;
            }
        }

        // Apply the cost-model decision.
        if (summary.target == ParallelTarget::SEQUENTIAL) {
            std::cout << "  -> Not profitable to parallelize\n";
            skipped++;
            continue;
        }

        // Insert OpenMP directives based on the summary.
        codegen.generatePragmas(summary);
        parallelizedLoops.insert(loop);

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

    // Generate output
    project->unparse();

    // Post-process the unparsed source:
    //   1. Hoist target data regions around consecutive GPU loops.
    //   2. Prepend #include <omp.h> if not already present.
    if (parallelized > 0) {
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

                    // Hoist target data regions (text-based).
                    OpenMPCodeGen::hoistTargetDataRegions(content);

                    // Check if omp.h is already included.
                    if (content.find("#include <omp.h>") == std::string::npos &&
                        content.find("#include \"omp.h\"") == std::string::npos) {
                        content = std::string("#include <omp.h>\n\n") + content;
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
