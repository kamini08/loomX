#include "InterproceduralAnalysis.h"
#include "ComputeIntensityEstimator.h"
#include <iostream>
#include <sstream>
#include <stack>

// ---------------------------------------------------------------------------
// Standard library classification
// ---------------------------------------------------------------------------

// Functions that are mathematically pure: same output for same input,
// no memory writes, no I/O.  These are safe to call inside parallel loops.
static const std::set<std::string>& getPureStdlibFunctions() {
    static const std::set<std::string> pure = {
        // C math library (cmath/math.h)
        "sin", "sinf", "cos", "cosf", "tan", "tanf",
        "asin", "asinf", "acos", "acosf", "atan", "atanf", "atan2", "atan2f",
        "sinh", "sinhf", "cosh", "coshf", "tanh", "tanhf",
        "asinh", "asinhf", "acosh", "acoshf", "atanh", "atanhf",
        "exp", "expf", "exp2", "exp2f", "expm1", "expm1f",
        "log", "logf", "log2", "log2f", "log10", "log10f", "log1p", "log1pf",
        "sqrt", "sqrtf", "cbrt", "cbrtf", "fabs", "fabsf", "ceil", "ceilf",
        "floor", "floorf", "round", "roundf", "trunc", "truncf",
        "fmod", "fmodf", "remainder", "remainderf", "pow", "powf",
        "hypot", "hypotf", "ldexp", "ldexpf", "scalbn", "scalbnf",
        "fmax", "fmaxf", "fmin", "fminf",
        // Type conversions / limits
        "abs", "labs", "llabs",
        // Read-only string inspection
        "strlen", "strcmp", "strncmp", "strchr", "strrchr", "strstr",
        // Memory comparison
        "memcmp",
    };
    return pure;
}

// Functions known to perform I/O or otherwise be unsafe for parallel loops.
static const std::set<std::string>& getImpureStdlibFunctions() {
    static const std::set<std::string> impure = {
        "printf", "fprintf", "sprintf", "snprintf",
        "scanf", "fscanf", "sscanf",
        "puts", "gets", "fgets", "fputs",
        "getchar", "putchar",
        "malloc", "calloc", "realloc", "free",
        "memcpy", "memmove", "memset",
        "exit", "abort", "system",
    };
    return impure;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static bool isPointerOrArrayType(SgType* type) {
    if (!type) return false;
    if (isSgPointerType(type) || isSgArrayType(type)) return true;
    if (SgReferenceType* ref = isSgReferenceType(type))
        return isPointerOrArrayType(ref->get_base_type());
    if (SgTypedefType* td = isSgTypedefType(type))
        return isPointerOrArrayType(td->get_base_type());
    return false;
}

static bool isLhsOfAssignment(SgExpression* expr) {
    if (!expr) return false;
    SgNode* parent = expr->get_parent();
    if (!parent) return false;

    if (SgAssignOp* op = isSgAssignOp(parent)) return op->get_lhs_operand() == expr;
    if (SgPlusAssignOp* op = isSgPlusAssignOp(parent)) return op->get_lhs_operand() == expr;
    if (SgMinusAssignOp* op = isSgMinusAssignOp(parent)) return op->get_lhs_operand() == expr;
    if (SgMultAssignOp* op = isSgMultAssignOp(parent)) return op->get_lhs_operand() == expr;
    if (SgDivAssignOp* op = isSgDivAssignOp(parent)) return op->get_lhs_operand() == expr;
    if (SgModAssignOp* op = isSgModAssignOp(parent)) return op->get_lhs_operand() == expr;
    if (SgAndAssignOp* op = isSgAndAssignOp(parent)) return op->get_lhs_operand() == expr;
    if (SgIorAssignOp* op = isSgIorAssignOp(parent)) return op->get_lhs_operand() == expr;
    if (SgXorAssignOp* op = isSgXorAssignOp(parent)) return op->get_lhs_operand() == expr;
    if (SgLshiftAssignOp* op = isSgLshiftAssignOp(parent)) return op->get_lhs_operand() == expr;
    if (SgRshiftAssignOp* op = isSgRshiftAssignOp(parent)) return op->get_lhs_operand() == expr;

    // Pre/post increment/decrement are also writes.
    if (isSgPlusPlusOp(parent) || isSgMinusMinusOp(parent)) return true;

    return false;
}

// For an array/pointer dereference expression, return the underlying variable
// that is being dereferenced.  Handles chains like (*(arr + i)).
static SgInitializedName* getBaseVariable(SgExpression* expr) {
    if (!expr) return nullptr;

    if (SgVarRefExp* varRef = isSgVarRefExp(expr)) {
        return varRef->get_symbol()->get_declaration();
    }
    if (SgPntrArrRefExp* arr = isSgPntrArrRefExp(expr)) {
        return getBaseVariable(arr->get_lhs_operand());
    }
    if (SgPointerDerefExp* deref = isSgPointerDerefExp(expr)) {
        return getBaseVariable(deref->get_operand());
    }
    if (SgCastExp* cast = isSgCastExp(expr)) {
        return getBaseVariable(cast->get_operand());
    }
    if (SgBinaryOp* bin = isSgBinaryOp(expr)) {
        // e.g. (arr + i); try the left operand first.
        if (SgInitializedName* left = getBaseVariable(bin->get_lhs_operand()))
            return left;
        return getBaseVariable(bin->get_rhs_operand());
    }
    return nullptr;
}

// Strip casts from an expression tree (free-function variant).
static SgNode* stripCastsExpr(SgNode* node) {
    while (SgCastExp* cast = isSgCastExp(node)) {
        node = cast->get_operand();
    }
    return node;
}

// For a pointer dereference expression used as a write, try to extract the
// scalar index expression.  Recognises:
//   *p          -> nullptr (no index, not analysable)
//   *(p + idx)  -> idx
//   *(idx + p)  -> idx
static SgExpression* extractPointerArithmeticIndex(SgExpression* derefOperand,
                                                   SgInitializedName* baseVar) {
    if (!derefOperand) return nullptr;
    derefOperand = isSgExpression(stripCastsExpr(derefOperand));
    if (!derefOperand) return nullptr;

    if (SgAddOp* add = isSgAddOp(derefOperand)) {
        SgExpression* lhs = isSgExpression(stripCastsExpr(add->get_lhs_operand()));
        SgExpression* rhs = isSgExpression(stripCastsExpr(add->get_rhs_operand()));
        if (!lhs || !rhs) return nullptr;

        SgInitializedName* lhsVar = getBaseVariable(lhs);
        SgInitializedName* rhsVar = getBaseVariable(rhs);

        if (lhsVar == baseVar && rhsVar != baseVar) return rhs;
        if (rhsVar == baseVar && lhsVar != baseVar) return lhs;
    }
    return nullptr;
}

// For an array/pointer subscript expression, extract the index part,
// tolerating a constant offset: arr[idx], arr[idx + c], arr[c + idx],
// arr[idx - c].
static SgExpression* extractSubscriptIndex(SgExpression* indexExpr) {
    if (!indexExpr) return nullptr;
    indexExpr = isSgExpression(stripCastsExpr(indexExpr));
    if (!indexExpr) return nullptr;

    if (SgAddOp* add = isSgAddOp(indexExpr)) {
        SgExpression* lhs = isSgExpression(stripCastsExpr(add->get_lhs_operand()));
        SgExpression* rhs = isSgExpression(stripCastsExpr(add->get_rhs_operand()));
        if (!lhs || !rhs) return nullptr;

        bool lhsConst = isSgValueExp(lhs) != nullptr;
        bool rhsConst = isSgValueExp(rhs) != nullptr;
        if (lhsConst && !rhsConst) return rhs;
        if (rhsConst && !lhsConst) return lhs;
    } else if (SgSubtractOp* sub = isSgSubtractOp(indexExpr)) {
        SgExpression* lhs = isSgExpression(stripCastsExpr(sub->get_lhs_operand()));
        SgExpression* rhs = isSgExpression(stripCastsExpr(sub->get_rhs_operand()));
        if (lhs && rhs && isSgValueExp(rhs) && !isSgValueExp(lhs)) return lhs;
    }
    return indexExpr;
}

// Walk up from a reference and look for an enclosing function definition.
static SgFunctionDefinition* getEnclosingFunctionDefinition(SgNode* node) {
    SgNode* current = node;
    while (current) {
        if (SgFunctionDefinition* def = isSgFunctionDefinition(current))
            return def;
        current = current->get_parent();
    }
    return nullptr;
}

// ---------------------------------------------------------------------------
// Public interface
// ---------------------------------------------------------------------------

void InterproceduralAnalysis::analyzeProject(SgProject* project) {
    summaries_.clear();

    // Phase 1: collect local facts for every function declaration.
    Rose_STL_Container<SgNode*> funcDecls =
        NodeQuery::querySubTree(project, V_SgFunctionDeclaration);
    for (SgNode* node : funcDecls) {
        if (SgFunctionDeclaration* funcDecl = isSgFunctionDeclaration(node)) {
            analyzeFunction(funcDecl);
        }
    }

    // Phase 2: wire up the call graph and detect recursion.
    buildCallGraphEdges();
    findRecursiveCycles();

    // Phase 3: propagate side effects bottom-up.
    propagateSideEffects();
}

const FunctionSummary* InterproceduralAnalysis::getSummary(const std::string& funcName) const {
    auto it = summaries_.find(funcName);
    if (it != summaries_.end()) return &(it->second);
    return nullptr;
}

bool InterproceduralAnalysis::isSafeForParallelLoop(SgFunctionCallExp* call) const {
    // Without loop-variable context we cannot prove per-iteration disjoint
    // pointer writes, so we use the conservative function-level check.
    return isSafeForParallelLoop(call, nullptr);
}

bool InterproceduralAnalysis::isSafeForParallelLoop(SgFunctionCallExp* call,
                                                    SgInitializedName* loopVar) const {
    if (!call) return false;

    SgFunctionDeclaration* funcDecl = call->getAssociatedFunctionDeclaration();
    if (!funcDecl) {
        // Unknown target (e.g. function pointer we could not resolve).
        return false;
    }

    std::string name = funcDecl->get_name().getString();

    // Whitelisted pure standard library functions are always safe.
    if (isStandardLibraryPureFunction(name)) return true;

    const FunctionSummary* summary = getSummary(name);
    if (!summary) return false;

    if (!summary->hasDefinition) return false;
    if (summary->hasTransitiveSideEffects) return false;
    if (summary->hasGlobalWrites) return false;
    if (summary->inRecursiveCycle) return false;

    // If there are no pointer writes, the function is safe.
    if (!summary->writesThroughPointer) return true;

    // Baseline mode: mimic an intraprocedural tool by rejecting all
    // pointer-parameter writes outright.
    if (intraproceduralBaseline_) return false;

    // Pointer writes are present. Without a loop variable we must reject.
    if (!loopVar) return false;

    // Check whether the pointer writes are per-iteration disjoint when this
    // specific call is placed inside a loop over loopVar.
    return pointerWritesAreLoopDisjoint(*summary, call, loopVar);
}

bool InterproceduralAnalysis::pointerWritesAreLoopDisjoint(
    const FunctionSummary& summary, SgFunctionCallExp* call,
    SgInitializedName* loopVar) const {
    if (summary.pointerWritePatterns.empty()) return false;

    SgExprListExp* args = call->get_args();
    if (!args) return false;
    std::vector<SgExpression*> argList(args->get_expressions().begin(),
                                       args->get_expressions().end());

    for (const FunctionSummary::PointerWritePattern& pattern : summary.pointerWritePatterns) {
        if (pattern.indexParamIdx < 0) {
            // Index is not a simple scalar parameter; cannot prove disjointness.
            return false;
        }
        if (pattern.indexParamIdx >= static_cast<int>(argList.size())) {
            return false;
        }

        SgExpression* arg = argList[pattern.indexParamIdx];
        SgInitializedName* argVar = getArgumentVariable(arg);
        if (argVar != loopVar) {
            // The index argument is not the loop iterator.
            return false;
        }
    }

    return true;
}

SgInitializedName* InterproceduralAnalysis::getArgumentVariable(SgExpression* expr) const {
    if (!expr) return nullptr;
    expr = isSgExpression(skipCasts(expr));
    if (!expr) return nullptr;
    SgVarRefExp* varRef = isSgVarRefExp(expr);
    if (!varRef) return nullptr;
    return varRef->get_symbol()->get_declaration();
}

bool InterproceduralAnalysis::isFunctionSafe(const std::string& funcName) const {
    // Whitelisted pure standard library functions are always safe.
    if (isStandardLibraryPureFunction(funcName)) return true;

    const FunctionSummary* summary = getSummary(funcName);
    if (!summary) return false;  // No summary -> unknown -> unsafe.

    if (!summary->hasDefinition) return false;
    if (summary->hasTransitiveSideEffects) return false;
    if (summary->writesThroughPointer) return false;
    if (summary->inRecursiveCycle) return false;

    // Pointer parameters that are written through are already rejected above.
    // Reading through pointers is fine, but the pointer variable itself must
    // not be modified (that would be a write to a parameter, which we do not
    // currently track as a caller-visible side effect for scalar parameters).
    return true;
}

void InterproceduralAnalysis::printSummaries(std::ostream& out) const {
    out << "=== loomX Interprocedural Analysis Summaries ===\n";
    for (const auto& entry : summaries_) {
        const FunctionSummary& s = entry.second;
        out << "Function: " << s.name << "\n";
        out << "  hasDefinition: " << s.hasDefinition
            << "  isLeaf: " << s.isLeaf
            << "  inCycle: " << s.inRecursiveCycle << "\n";
        out << "  local IO: " << s.hasIOSideEffects
            << "  local global-write: " << s.hasGlobalWrites
            << "  local pointer-write: " << s.writesThroughPointer
            << "  transitive side-effects: " << s.hasTransitiveSideEffects << "\n";
        out << "  readParams: {";
        for (int p : s.readParams) out << p << " ";
        out << "} writtenParams: {";
        for (int p : s.writtenParams) out << p << " ";
        out << "}\n";
        out << "  pointerReadParams: {";
        for (int p : s.pointerReadParams) out << p << " ";
        out << "} pointerWriteParams: {";
        for (int p : s.pointerWriteParams) out << p << " ";
        out << "}\n";
        if (!s.pointerWritePatterns.empty()) {
            out << "  pointerWritePatterns: {";
            for (const auto& pat : s.pointerWritePatterns) {
                out << "p" << pat.pointerParamIdx << "[p" << pat.indexParamIdx << "] ";
            }
            out << "}\n";
        }
        out << "  callees: {";
        for (const std::string& c : s.callees) out << c << " ";
        out << "}\n";
        out << "  SAFE for parallel loop: " << (isFunctionSafe(s.name) ? "YES" : "NO") << "\n\n";
    }
}

// ---------------------------------------------------------------------------
// Phase 1: local analysis
// ---------------------------------------------------------------------------

void InterproceduralAnalysis::analyzeFunction(SgFunctionDeclaration* funcDecl) {
    std::string name = funcDecl->get_name().getString();

    // If we have already seen a declaration for this name, keep the one with
    // a definition if possible.
    auto existing = summaries_.find(name);
    if (existing != summaries_.end() && existing->second.hasDefinition) {
        return;
    }

    FunctionSummary summary;
    summary.name = name;
    summary.decl = funcDecl;
    summary.hasDefinition = (funcDecl->get_definition() != nullptr);

    SgFunctionDefinition* def = funcDecl->get_definition();

    // Standard library functions are treated specially.  If we see a
    // declaration for a known pure function, record a safe summary.
    if (isStandardLibraryPureFunction(name)) {
        summary.isLeaf = true;
        summaries_[name] = summary;
        return;
    }

    if (isStandardLibraryImpureFunction(name)) {
        summary.hasIOSideEffects = true;
        summary.isLeaf = true;
        summaries_[name] = summary;
        return;
    }

    if (!def) {
        // Unknown external function: conservative.
        summaries_[name] = summary;
        return;
    }

    // Parameter MOD/REF and pointer dereference tracking.
    collectParameterAccess(funcDecl, summary);

    // Immediate callees and leaf status.
    collectCallees(funcDecl, summary);

    // Local work estimate (FLOPs and memory ops) for the function body.
    if (def) {
        loomX::ComputeIntensityEstimator estimator;
        loomX::ComputeIntensityResult work = estimator.estimateFunctionWork(def);
        summary.localFlops = work.flopCount;
        summary.localMemOps = work.memoryOpCount;
        summary.estimatedFlops = work.flopCount;
        summary.estimatedMemOps = work.memoryOpCount;
    }

    // Local side effects are any effects visible to the caller.
    summary.hasIOSideEffects = [&]() {
        Rose_STL_Container<SgNode*> calls =
            NodeQuery::querySubTree(def, V_SgFunctionCallExp);
        for (SgNode* node : calls) {
            SgFunctionCallExp* call = isSgFunctionCallExp(node);
            if (!call) continue;
            SgFunctionDeclaration* callee = call->getAssociatedFunctionDeclaration();
            if (!callee) continue;
            if (isStandardLibraryImpureFunction(callee->get_name().getString()))
                return true;
        }
        return false;
    }();

    // Global/static writes.
    {
        std::vector<SgNode*> readRefs, writeRefs;
        SageInterface::collectReadWriteRefs(def->get_body(), readRefs, writeRefs);

        SgScopeStatement* body = def->get_body();
        SgInitializedNamePtrList& params = funcDecl->get_args();
        std::set<SgInitializedName*> paramSet(params.begin(), params.end());

        for (SgNode* ref : writeRefs) {
            SgVarRefExp* varRef = isSgVarRefExp(ref);
            if (!varRef) continue;
            SgInitializedName* var = varRef->get_symbol()->get_declaration();
            if (!var) continue;

            // Parameters are not global.
            if (paramSet.find(var) != paramSet.end()) continue;

            // Locals are not global.
            SgScopeStatement* varScope = var->get_scope();
            if (varScope == body) continue;

            // Pointer/array parameter dereferences are handled separately.
            if (paramSet.find(var) != paramSet.end() && isPointerOrArrayType(var->get_type()))
                continue;

            // Anything else (global, file-static, outer scope) counts.
            if (isSgGlobal(varScope) || varScope != body) {
                summary.hasGlobalWrites = true;
                break;
            }
        }
    }

    summaries_[name] = summary;
}

void InterproceduralAnalysis::collectParameterAccess(SgFunctionDeclaration* funcDecl,
                                                     FunctionSummary& summary) {
    SgFunctionDefinition* def = funcDecl->get_definition();
    if (!def) return;

    SgInitializedNamePtrList& params = funcDecl->get_args();
    std::vector<SgInitializedName*> paramList(params.begin(), params.end());
    if (paramList.empty()) return;

    std::vector<SgNode*> readRefs, writeRefs;
    SageInterface::collectReadWriteRefs(def->get_body(), readRefs, writeRefs);

    auto paramIndex = [&](SgInitializedName* var) -> int {
        for (size_t i = 0; i < paramList.size(); ++i) {
            if (paramList[i] == var) return static_cast<int>(i);
        }
        return -1;
    };

    for (SgNode* ref : readRefs) {
        SgInitializedName* var = nullptr;
        bool isDerefRead = false;

        if (SgVarRefExp* varRef = isSgVarRefExp(ref)) {
            var = varRef->get_symbol()->get_declaration();
            // Check whether the reference is inside a dereference.
            SgNode* current = varRef;
            while (current && !isSgFunctionDefinition(current)) {
                SgNode* parent = current->get_parent();
                if (isSgPntrArrRefExp(parent) || isSgPointerDerefExp(parent)) {
                    isDerefRead = true;
                    break;
                }
                current = parent;
            }
        } else if (SgPntrArrRefExp* arr = isSgPntrArrRefExp(ref)) {
            var = getBaseVariable(arr);
            isDerefRead = true;
        } else if (SgPointerDerefExp* deref = isSgPointerDerefExp(ref)) {
            var = getBaseVariable(deref);
            isDerefRead = true;
        }

        if (!var) continue;
        int idx = paramIndex(var);
        if (idx < 0) continue;

        summary.readParams.insert(idx);

        if (isPointerOrArrayType(var->get_type()) && isDerefRead) {
            summary.pointerReadParams.insert(idx);
        }
    }

    for (SgNode* ref : writeRefs) {
        SgInitializedName* var = nullptr;
        bool isDerefWrite = false;
        SgExpression* indexExpr = nullptr;

        if (SgVarRefExp* varRef = isSgVarRefExp(ref)) {
            var = varRef->get_symbol()->get_declaration();
            isDerefWrite = isDereferenceWrite(ref);
        } else if (SgPntrArrRefExp* arr = isSgPntrArrRefExp(ref)) {
            // The whole array reference is reported as a write.
            var = getBaseVariable(arr);
            isDerefWrite = true;
            indexExpr = extractSubscriptIndex(arr->get_rhs_operand());
        } else if (SgPointerDerefExp* deref = isSgPointerDerefExp(ref)) {
            var = getBaseVariable(deref);
            isDerefWrite = true;
            indexExpr = extractPointerArithmeticIndex(deref->get_operand(), var);
        }

        if (!var) continue;
        int idx = paramIndex(var);
        if (idx < 0) continue;

        summary.writtenParams.insert(idx);

        if (isPointerOrArrayType(var->get_type()) && isDerefWrite) {
            summary.pointerWriteParams.insert(idx);
            summary.writesThroughPointer = true;

            // Try to record a simple per-iteration disjoint pattern:
            // pointer parameter p is written at index given by scalar parameter q.
            FunctionSummary::PointerWritePattern pattern;
            pattern.pointerParamIdx = idx;
            if (indexExpr) {
                if (SgVarRefExp* idxVarRef = isSgVarRefExp(skipCasts(indexExpr))) {
                    SgInitializedName* idxVar = idxVarRef->get_symbol()->get_declaration();
                    int idxParam = paramIndex(idxVar);
                    if (idxParam >= 0 && !isPointerOrArrayType(idxVar->get_type())) {
                        pattern.indexParamIdx = idxParam;
                    }
                }
            }
            summary.pointerWritePatterns.push_back(pattern);
        }
    }
}

void InterproceduralAnalysis::collectCallees(SgFunctionDeclaration* funcDecl,
                                             FunctionSummary& summary) {
    SgFunctionDefinition* def = funcDecl->get_definition();
    if (!def) return;

    Rose_STL_Container<SgNode*> calls =
        NodeQuery::querySubTree(def, V_SgFunctionCallExp);

    for (SgNode* node : calls) {
        SgFunctionCallExp* call = isSgFunctionCallExp(node);
        if (!call) continue;
        SgFunctionDeclaration* callee = call->getAssociatedFunctionDeclaration();
        if (!callee) continue;

        std::string calleeName = callee->get_name().getString();

        // Standard library calls do not make a function non-leaf for our
        // purposes, because their effects are captured locally.
        if (isStandardLibraryPureFunction(calleeName) ||
            isStandardLibraryImpureFunction(calleeName)) {
            continue;
        }

        // Recursive self-calls still count as callees.
        summary.callees.insert(calleeName);
    }

    summary.isLeaf = summary.callees.empty();
}

// ---------------------------------------------------------------------------
// Phase 2: call graph and recursive cycle detection
// ---------------------------------------------------------------------------

void InterproceduralAnalysis::buildCallGraphEdges() {
    for (auto& entry : summaries_) {
        FunctionSummary& caller = entry.second;
        for (const std::string& calleeName : caller.callees) {
            auto it = summaries_.find(calleeName);
            if (it != summaries_.end()) {
                it->second.callers.insert(caller.name);
            }
        }
    }
}

void InterproceduralAnalysis::findRecursiveCycles() {
    std::map<std::string, int> index;
    std::map<std::string, int> lowlink;
    std::vector<std::string> stack;
    std::set<std::string> onStack;
    int currentIndex = 0;

    for (const auto& entry : summaries_) {
        const std::string& name = entry.first;
        if (index.find(name) == index.end()) {
            dfsSCC(name, index, lowlink, stack, onStack, currentIndex);
        }
    }
}

void InterproceduralAnalysis::dfsSCC(const std::string& node,
                                     std::map<std::string, int>& index,
                                     std::map<std::string, int>& lowlink,
                                     std::vector<std::string>& stack,
                                     std::set<std::string>& onStack,
                                     int& currentIndex) {
    index[node] = currentIndex;
    lowlink[node] = currentIndex;
    currentIndex++;
    stack.push_back(node);
    onStack.insert(node);

    auto it = summaries_.find(node);
    if (it != summaries_.end()) {
        for (const std::string& callee : it->second.callees) {
            if (summaries_.find(callee) == summaries_.end()) continue;

            if (index.find(callee) == index.end()) {
                dfsSCC(callee, index, lowlink, stack, onStack, currentIndex);
                lowlink[node] = std::min(lowlink[node], lowlink[callee]);
            } else if (onStack.find(callee) != onStack.end()) {
                lowlink[node] = std::min(lowlink[node], index[callee]);
            }
        }
    }

    if (lowlink[node] == index[node]) {
        std::set<std::string> component;
        std::string w;
        do {
            w = stack.back();
            stack.pop_back();
            onStack.erase(w);
            component.insert(w);
        } while (w != node);

        // A component with more than one node, or a self-loop, is recursive.
        bool isCycle = (component.size() > 1);
        if (!isCycle) {
            auto sit = summaries_.find(node);
            if (sit != summaries_.end() && sit->second.callees.count(node)) {
                isCycle = true;
            }
        }

        if (isCycle) {
            for (const std::string& member : component) {
                summaries_[member].inRecursiveCycle = true;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Phase 3: bottom-up side-effect propagation
// ---------------------------------------------------------------------------

void InterproceduralAnalysis::propagateSideEffects() {
    // Initialize: local side effects that are unsafe regardless of call site.
    // Pointer writes with a simple parameter-index pattern may be safe when the
    // call is inside a loop that passes the loop iterator as that parameter,
    // so they are not treated as unconditional side effects here.
    for (auto& entry : summaries_) {
        FunctionSummary& s = entry.second;
        s.hasTransitiveSideEffects =
            s.hasIOSideEffects || s.hasGlobalWrites || hasUnanalyzablePointerWrites(s);
    }

    // Worklist: start with functions that are locally unsafe.
    std::set<std::string> worklist;
    for (const auto& entry : summaries_) {
        if (entry.second.hasTransitiveSideEffects || entry.second.inRecursiveCycle) {
            worklist.insert(entry.first);
        }
    }

    // Propagate backwards through callers until fixed point.
    while (!worklist.empty()) {
        std::string current = *worklist.begin();
        worklist.erase(worklist.begin());

        auto it = summaries_.find(current);
        if (it == summaries_.end()) continue;

        for (const std::string& callerName : it->second.callers) {
            auto callerIt = summaries_.find(callerName);
            if (callerIt == summaries_.end()) continue;
            if (!callerIt->second.hasTransitiveSideEffects) {
                callerIt->second.hasTransitiveSideEffects = true;
                worklist.insert(callerName);
            }
        }
    }

    // Propagate work estimates bottom-up.  Recursive cycles invalidate callee
    // estimates, so we skip them.
    for (auto& entry : summaries_) {
        FunctionSummary& s = entry.second;
        if (s.inRecursiveCycle) {
            s.estimatedFlops = -1;
            s.estimatedMemOps = -1;
        }
    }

    // Keep propagating until no work estimate changes.  Recompute each
    // function's total from its local work plus the current totals of its
    // callees, so estimates converge to the transitive closure without
    // double-counting.
    bool changed = true;
    while (changed) {
        changed = false;
        for (auto& entry : summaries_) {
            FunctionSummary& caller = entry.second;
            if (caller.inRecursiveCycle) continue;

            long long newFlops = caller.localFlops;
            long long newMemOps = caller.localMemOps;
            for (const std::string& calleeName : caller.callees) {
                auto it = summaries_.find(calleeName);
                if (it == summaries_.end()) continue;
                const FunctionSummary& callee = it->second;
                if (callee.inRecursiveCycle) continue;
                if (callee.estimatedFlops < 0 || callee.estimatedMemOps < 0) continue;
                newFlops += callee.estimatedFlops;
                newMemOps += callee.estimatedMemOps;
            }

            if (newFlops != caller.estimatedFlops || newMemOps != caller.estimatedMemOps) {
                caller.estimatedFlops = newFlops;
                caller.estimatedMemOps = newMemOps;
                changed = true;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Helper predicates
// ---------------------------------------------------------------------------

bool InterproceduralAnalysis::isStandardLibraryPureFunction(const std::string& name) const {
    return getPureStdlibFunctions().count(name) > 0;
}

bool InterproceduralAnalysis::isStandardLibraryImpureFunction(const std::string& name) const {
    return getImpureStdlibFunctions().count(name) > 0;
}

bool InterproceduralAnalysis::isDereferenceWrite(SgNode* ref) const {
    SgVarRefExp* varRef = isSgVarRefExp(ref);
    if (!varRef) return false;

    SgInitializedName* var = varRef->get_symbol()->get_declaration();
    if (!var) return false;
    if (!isPointerOrArrayType(var->get_type())) return false;

    // Walk up from the variable reference.  If we encounter a pointer/array
    // dereference node that is the LHS of an assignment, this is a write
    // through the pointer.
    SgNode* current = varRef;
    while (current && !isSgFunctionDefinition(current)) {
        SgNode* parent = current->get_parent();

        if (SgPntrArrRefExp* arr = isSgPntrArrRefExp(parent)) {
            if (isLhsOfAssignment(arr)) return true;
        }
        if (SgPointerDerefExp* deref = isSgPointerDerefExp(parent)) {
            if (isLhsOfAssignment(deref)) return true;
        }

        current = parent;
    }
    return false;
}

SgNode* InterproceduralAnalysis::skipCasts(SgNode* node) const {
    while (SgCastExp* cast = isSgCastExp(node)) {
        node = cast->get_operand();
    }
    return node;
}

bool InterproceduralAnalysis::hasUnanalyzablePointerWrites(
    const FunctionSummary& summary) const {
    if (!summary.writesThroughPointer) return false;

    // If we detected pointer writes but no patterns, the writes are not
    // analyzable (e.g., *p = ...).
    if (summary.pointerWritePatterns.empty()) return true;

    // Any pattern whose index is not a simple scalar parameter is not
    // analyzable at the call site.
    for (const FunctionSummary::PointerWritePattern& pattern : summary.pointerWritePatterns) {
        if (pattern.indexParamIdx < 0) return true;
    }

    return false;
}
