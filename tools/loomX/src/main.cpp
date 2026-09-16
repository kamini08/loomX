#include "rose.h"
#include "InterproceduralAnalysis.h"
#include "GpuProfitability.h"
#include "OpenMPCodeGen.h"
#include "ReductionDetector.h"
#include "LoopAnalysisTypes.h"
#include <iostream>
#include <vector>

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

// Check if loop contains function calls
bool hasFunctionCalls(SgForStatement* loop) {
    Rose_STL_Container<SgNode*> calls =
        NodeQuery::querySubTree(loop, V_SgFunctionCallExp);
    return !calls.empty();
}

// Collect variables that need to be explicitely listed in an OpenMP private
// clause.  Loop-index variables are implicitly private in OpenMP parallel-for
// and must not be listed when declared in the for-init statement, because the
// pragma is inserted outside the loop's scope.  Variables declared inside the
// loop body are already local to each iteration and are also out of scope at
// the pragma location, so they are skipped as well.
void collectPrivateVars(SgForStatement* loop,
                        std::set<SgInitializedName*>& privateVars) {
    SgInitializedName* loopVar = SageInterface::getLoopIndexVariable(loop);
    if (!loopVar) return;

    // If the loop variable is declared in the for-init statement, OpenMP
    // handles it implicitly; do not add it to the private clause.
    SgStatement* initStmt = loop->get_for_init_stmt();
    if (initStmt) {
        SgVariableDeclaration* initDecl = isSgVariableDeclaration(initStmt);
        if (initDecl) {
            SgInitializedNamePtrList& vars = initDecl->get_variables();
            for (SgInitializedName* v : vars) {
                if (v == loopVar) return;
            }
        }
    }

    // For C89-style loops (index declared before the loop), we still prefer to
    // rely on the implicit private semantics of the loop iteration variable in
    // OpenMP parallel-for, which is universally supported.  Listing it is
    // unnecessary and can trigger scope warnings with some compilers.
    //
    // If future heuristics need explicit privates for outer-scope temporaries,
    // they can be added here after liveness analysis.
}

// Collect variables that are live-in and live-out (need map clauses for GPU)
void collectMapVariables(SgForStatement* loop,
                         std::set<std::pair<SgInitializedName*, std::string>>& mapClauses) {
    // Collect all variable references including those inside array subscripts
    std::set<SgInitializedName*> allVars;
    Rose_STL_Container<SgNode*> varRefs =
        NodeQuery::querySubTree(loop, V_SgVarRefExp);
    for (SgNode* node : varRefs) {
        SgVarRefExp* varRef = isSgVarRefExp(node);
        if (!varRef) continue;
        SgInitializedName* var = varRef->get_symbol()->get_declaration();
        if (var) allVars.insert(var);
    }

    // Collect read/write references in the loop
    std::vector<SgNode*> readRefs, writeRefs;
    SageInterface::collectReadWriteRefs(loop, readRefs, writeRefs);

    // Track which variables are read, written, or both
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

    // For all variables in the loop, determine map direction
    for (SgInitializedName* var : allVars) {
        // Skip loop index variable - it's handled by private clause
        SgInitializedName* loopVar = SageInterface::getLoopIndexVariable(loop);
        if (var == loopVar) continue;

        // Skip loop index variables of nested loops - they are private to inner loops
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

        // Skip local variables declared inside the loop body itself
        SgScopeStatement* varScope = var->get_scope();
        SgStatement* loopBody = loop->get_loop_body();
        if (loopBody && varScope == loopBody->get_scope()) {
            continue;
        }

        bool isRead = readVars.find(var) != readVars.end();
        bool isWritten = writeVars.find(var) != writeVars.end();

        if (isRead && isWritten) {
            mapClauses.insert({var, "tofrom"});
        } else if (isRead) {
            mapClauses.insert({var, "to"});
        } else if (isWritten) {
            mapClauses.insert({var, "from"});
        } else {
            // Variable appears but read/write not detected (e.g., used only in sizeof)
            // Default to tofrom to be safe
            mapClauses.insert({var, "tofrom"});
        }
    }
}

// Detect reduction variables using the dedicated reduction analyzer.
void collectReductionVars(SgForStatement* loop,
                          std::set<SgInitializedName*>& reductionVars,
                          std::vector<loomX::ReductionInfo>& reductionDetails) {
    loomX::ReductionDetector detector;
    reductionDetails = detector.analyze(loop);
    for (const loomX::ReductionInfo& info : reductionDetails) {
        if (info.variable) reductionVars.insert(info.variable);
    }
}

int main(int argc, char* argv[]) {
    ROSE_INITIALIZE;

    bool verbose = false;
    std::vector<std::string> args;
    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        if (arg == "-v" || arg == "--verbose") {
            verbose = true;
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

        // Check canonical form
        if (!SageInterface::isCanonicalForLoop(loop)) {
            std::cout << "  -> Skipped: not a canonical loop\n";
            skipped++;
            continue;
        }

        // Check for function calls
        if (hasFunctionCalls(loop)) {
            // Try interprocedural analysis
            Rose_STL_Container<SgNode*> calls =
                NodeQuery::querySubTree(loop, V_SgFunctionCallExp);
            bool allSafe = true;
            for (SgNode* node : calls) {
                SgFunctionCallExp* call = isSgFunctionCallExp(node);
                if (!ipa.isSafeForParallelLoop(call)) {
                    std::cout << "  -> Function call not safe for parallelization\n";
                    allSafe = false;
                    break;
                }
            }
            if (!allSafe) {
                skipped++;
                continue;
            }
            std::cout << "  -> All function calls are safe (interprocedural analysis)\n";
        }

        // GPU profitability analysis
        ParallelTarget target = profitability.classifyLoop(loop);

        if (target == ParallelTarget::SEQUENTIAL) {
            std::cout << "  -> Not profitable to parallelize\n";
            skipped++;
            continue;
        }

        // Collect variable classifications
        std::set<SgInitializedName*> privateVars;
        std::set<SgInitializedName*> reductionVars;
        std::vector<loomX::ReductionInfo> reductionDetails;
        std::set<std::pair<SgInitializedName*, std::string>> mapClauses;

        collectPrivateVars(loop, privateVars);
        collectReductionVars(loop, reductionVars, reductionDetails);

        if (target == ParallelTarget::GPU_OFFLOAD) {
            collectMapVariables(loop, mapClauses);

            // Remove reduction variables from map clauses (they use reduction clause instead)
            for (SgInitializedName* redVar : reductionVars) {
                auto it = mapClauses.begin();
                while (it != mapClauses.end()) {
                    if (it->first == redVar) {
                        it = mapClauses.erase(it);
                    } else {
                        ++it;
                    }
                }
            }

            // Add omp declare target for functions called inside GPU region
            Rose_STL_Container<SgNode*> calls =
                NodeQuery::querySubTree(loop, V_SgFunctionCallExp);
            for (SgNode* node : calls) {
                SgFunctionCallExp* call = isSgFunctionCallExp(node);
                if (!call) continue;
                SgFunctionDeclaration* callee = call->getAssociatedFunctionDeclaration();
                if (!callee) continue;
                // Insert declare target pragma before function definition
                SgFunctionDefinition* def = callee->get_definition();
                if (def) {
                    SgFunctionDeclaration* decl = def->get_declaration();
                    if (decl) {
                        SgPragmaDeclaration* declareTarget =
                            SageBuilder::buildPragmaDeclaration("omp declare target",
                                                                 decl->get_scope());
                        SageInterface::insertStatementBefore(decl, declareTarget);
                    }
                }
            }
        }

        // Insert OpenMP directives
        codegen.generatePragmas(loop, target, privateVars, reductionVars, mapClauses, reductionDetails);
        parallelizedLoops.insert(loop);

        if (target == ParallelTarget::GPU_OFFLOAD) {
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
    // Use unparse instead of backend to preserve inserted pragmas
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
