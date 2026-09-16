#include "rose.h"
#include "InterproceduralAnalysis.h"
#include "GpuProfitability.h"
#include "OpenMPCodeGen.h"
#include "LoopSummary.h"
#include "LoopDependenceAnalysis.h"
#include "LoopAnalysisTypes.h"
#include <iostream>
#include <vector>

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

    // Fill codegen inputs.
    fillPrivateVars(loop, summary);
    if (summary.target == ParallelTarget::GPU_OFFLOAD) {
        fillMapVariables(loop, summary);
    }

    return summary;
}

int main(int argc, char* argv[]) {
    ROSE_INITIALIZE;

    bool verbose = false;
    bool intraproceduralBaseline = false;
    std::vector<std::string> args;
    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        if (arg == "-v" || arg == "--verbose") {
            verbose = true;
        } else if (arg == "--intraprocedural-baseline") {
            intraproceduralBaseline = true;
        } else {
            args.push_back(arg);
        }
    }

    if (args.empty()) {
        std::cerr << "Usage: " << argv[0] << " [-v|--verbose] <input.c> [-o output.c]\n";
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
    GpuProfitability profitability;
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

    // Post-process: prepend #include <omp.h> if we parallelized anything
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

                    // Check if omp.h is already included
                    if (content.find("#include <omp.h>") == std::string::npos &&
                        content.find("#include \"omp.h\"") == std::string::npos) {
                        std::ofstream outFile(outputName);
                        if (outFile) {
                            outFile << "#include <omp.h>\n\n" << content;
                            outFile.close();
                        }
                    }
                }
            }
        }
    }

    return 0;
}
