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

    // Parameter access (by index)
    std::set<int> readParams;
    std::set<int> writtenParams;

    // Pointer parameters that are dereferenced (read or written)
    std::set<int> pointerReadParams;
    std::set<int> pointerWriteParams;

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

    // Get summary for a function by name.
    const FunctionSummary* getSummary(const std::string& funcName) const;

    // Check if a function call is safe to call inside a parallel loop.
    bool isSafeForParallelLoop(SgFunctionCallExp* call) const;

    // Check if a function (by name) is safe for parallel loops.
    bool isFunctionSafe(const std::string& funcName) const;

    // Print a human-readable summary of every function.
    void printSummaries(std::ostream& out = std::cout) const;

private:
    std::map<std::string, FunctionSummary> summaries_;

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
};
