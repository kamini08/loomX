#pragma once
#include "rose.h"
#include <map>
#include <set>
#include <string>
#include <vector>

// Summary of a function's side effects and call relationships.
// This is used both for direct safety checks and for transitive
// propagation through the call graph.
struct FunctionSummary {
    std::string name;
    SgFunctionDeclaration* decl = nullptr;

    // Basic facts
    bool hasDefinition = false;
    bool isLeaf = false;

    // Local side effects (do not depend on callees)
    bool hasIOSideEffects = false;       // printf, scanf, fprintf, ...
    bool hasGlobalWrites = false;        // writes to global/static variables
    bool writesThroughPointer = false;   // writes through pointer/array parameters

    // Transitive side effects (propagated from the whole call tree)
    bool hasTransitiveSideEffects = false;

    // Is this function part of a recursive cycle?
    bool inRecursiveCycle = false;

    // Estimated work per call (static).  local* is the function body's own
    // work; estimated* includes all non-recursive callees, propagated bottom-up.
    long long localFlops = 0;
    long long localMemOps = 0;
    long long estimatedFlops = 0;
    long long estimatedMemOps = 0;

    // Parameter access (by index)
    std::set<int> readParams;
    std::set<int> writtenParams;

    // Pointer parameters that are dereferenced (read or written)
    std::set<int> pointerReadParams;
    std::set<int> pointerWriteParams;

    // For each pointer-parameter write, record whether the index expression is
    // exactly another scalar parameter. If so, calls that pass the loop index
    // as that scalar parameter may be safe because the writes are per-iteration
    // disjoint.
    struct PointerWritePattern {
        int pointerParamIdx = -1;   // Index of the pointer parameter being written
        int indexParamIdx = -1;     // Index of the scalar parameter used as index (-1 if not)
    };
    std::vector<PointerWritePattern> pointerWritePatterns;

    // Call graph edges (by function name)
    std::set<std::string> callees;
    std::set<std::string> callers;
};

// Interprocedural side-effect and call-graph analysis.
//
// The analysis runs in three phases:
//   1. Build local summaries for every function (I/O, globals, pointer writes,
//      parameter MOD/REF, and immediate callees).
//   2. Build the call graph and detect recursive cycles.
//   3. Propagate side effects bottom-up through the call graph.
//
// After analysis, isSafeForParallelLoop() answers whether a function call
// inside a parallel loop is guaranteed to be side-effect free.
class InterproceduralAnalysis {
public:
    // Build summaries for all functions in the project.
    void analyzeProject(SgProject* project);

    // When true, reject all pointer-parameter writes (mimics an intraprocedural
    // baseline). Used for before/after demonstrations.
    void setIntraproceduralBaseline(bool enabled) {
        intraproceduralBaseline_ = enabled;
    }

    // Get summary for a function by name.
    const FunctionSummary* getSummary(const std::string& funcName) const;

    // Check if a function call is safe to call inside a parallel loop.
    // The loop-variable overload allows pointer-parameter writes indexed by
    // the loop iterator to be treated as per-iteration disjoint.
    bool isSafeForParallelLoop(SgFunctionCallExp* call) const;
    bool isSafeForParallelLoop(SgFunctionCallExp* call,
                               SgInitializedName* loopVar) const;

    // Check if a function (by name) is safe for parallel loops.
    bool isFunctionSafe(const std::string& funcName) const;

    // Print a human-readable summary of every function.
    void printSummaries(std::ostream& out = std::cout) const;

private:
    std::map<std::string, FunctionSummary> summaries_;
    bool intraproceduralBaseline_ = false;

    // Phase 1: local analysis
    void analyzeFunction(SgFunctionDeclaration* funcDecl);
    void collectParameterAccess(SgFunctionDeclaration* funcDecl,
                                FunctionSummary& summary);
    void collectCallees(SgFunctionDeclaration* funcDecl,
                        FunctionSummary& summary);

    // Phase 2: call graph and cycles
    void buildCallGraphEdges();
    void findRecursiveCycles();
    void dfsSCC(const std::string& node,
                std::map<std::string, int>& index,
                std::map<std::string, int>& lowlink,
                std::vector<std::string>& stack,
                std::set<std::string>& onStack,
                int& currentIndex);

    // Phase 3: bottom-up propagation
    void propagateSideEffects();

    // Helpers
    bool isStandardLibraryPureFunction(const std::string& name) const;
    bool isStandardLibraryImpureFunction(const std::string& name) const;
    bool isDereferenceWrite(SgNode* ref) const;
    SgNode* skipCasts(SgNode* node) const;

    // For a call argument expression, return the parameter/variable it refers
    // to after stripping casts, or nullptr if it is not a bare variable.
    SgInitializedName* getArgumentVariable(SgExpression* expr) const;

    // Check whether every pointer-parameter write in the callee is indexed by
    // loopVar in this specific call.
    bool pointerWritesAreLoopDisjoint(const FunctionSummary& summary,
                                      SgFunctionCallExp* call,
                                      SgInitializedName* loopVar) const;

    // True if the function has pointer writes that are not simple
    // parameter-index patterns and are therefore unsafe regardless of call site.
    bool hasUnanalyzablePointerWrites(const FunctionSummary& summary) const;
};
