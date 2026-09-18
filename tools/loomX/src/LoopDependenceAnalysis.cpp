#include "LoopDependenceAnalysis.h"
#include <algorithm>
#include <numeric>

namespace loomX {

namespace {

// Greatest common divisor.
long long gcd(long long a, long long b) {
    a = std::llabs(a);
    b = std::llabs(b);
    while (b != 0) {
        long long t = b;
        b = a % b;
        a = t;
    }
    return a;
}

// Skip casts and parentheses.
SgExpression* stripCastsAndParens(SgExpression* expr) {
    while (expr) {
        if (SgCastExp* cast = isSgCastExp(expr)) {
            expr = cast->get_operand();
        } else if (SgUnaryOp* uop = isSgUnaryOp(expr)) {
            // Only strip parentheses-like unary ops if any; for now just return.
            break;
        } else {
            break;
        }
    }
    return expr;
}

// Extract the base variable from a possibly multi-dimensional array reference.
// For A[i][j], the outermost SgPntrArrRefExp's lhs chain ends at A.
SgInitializedName* getBaseVariable(SgExpression* expr) {
    while (SgPntrArrRefExp* arrRef = isSgPntrArrRefExp(expr)) {
        expr = arrRef->get_lhs_operand();
    }
    SgVarRefExp* varRef = isSgVarRefExp(expr);
    if (!varRef) return nullptr;
    return varRef->get_symbol()->get_declaration();
}

// Collect the index variables of all for-loops nested immediately or deeply
// inside the given loop body.
std::set<SgInitializedName*> collectNestedLoopIndexVariables(SgForStatement* loop) {
    std::set<SgInitializedName*> vars;
    if (!loop) return vars;
    SgStatement* body = loop->get_loop_body();
    if (!body) return vars;

    Rose_STL_Container<SgNode*> nestedLoops =
        NodeQuery::querySubTree(body, V_SgForStatement);
    for (SgNode* node : nestedLoops) {
        if (SgForStatement* nested = isSgForStatement(node)) {
            if (nested == loop) continue;
            if (SgInitializedName* iv = SageInterface::getLoopIndexVariable(nested)) {
                vars.insert(iv);
            }
        }
    }
    return vars;
}

// True if expr references var.
bool subscriptContainsVar(SgExpression* expr, SgInitializedName* var) {
    if (!expr || !var) return false;
    Rose_STL_Container<SgNode*> refs = NodeQuery::querySubTree(expr, V_SgVarRefExp);
    for (SgNode* node : refs) {
        SgVarRefExp* ref = isSgVarRefExp(node);
        if (ref && ref->get_symbol()->get_declaration() == var) return true;
    }
    return false;
}

// True if expr references at least one variable from nestedLoopVars.
bool subscriptContainsAnyNestedLoopVar(
    SgExpression* expr,
    const std::set<SgInitializedName*>& nestedLoopVars) {
    if (!expr || nestedLoopVars.empty()) return false;
    Rose_STL_Container<SgNode*> refs = NodeQuery::querySubTree(expr, V_SgVarRefExp);
    for (SgNode* node : refs) {
        SgVarRefExp* ref = isSgVarRefExp(node);
        if (ref && nestedLoopVars.find(ref->get_symbol()->get_declaration()) != nestedLoopVars.end())
            return true;
    }
    return false;
}

} // anonymous namespace

DependenceResult LoopDependenceAnalysis::analyze(SgForStatement* loop) {
    DependenceResult result;
    if (!loop) {
        result.hasLoopCarriedDependence = true;
        result.description = "null loop";
        return result;
    }

    SgInitializedName* loopVar = SageInterface::getLoopIndexVariable(loop);
    if (!loopVar) {
        result.hasLoopCarriedDependence = true;
        result.description = "could not determine loop index variable";
        return result;
    }

    std::vector<ArrayReference> refs = collectArrayReferences(loop);

    // Conservative guard: if a write reference inside this loop is indexed by
    // a nested loop variable and the current loop variable does not appear in
    // any of its subscripts, different iterations of the current loop may touch
    // the same element.  Treat that as a loop-carried dependence so we don't
    // parallelize outer loops that contain reductions over inner-loop indices
    // (e.g. PolyBench atax/bicg).
    std::set<SgInitializedName*> nestedLoopVars = collectNestedLoopIndexVariables(loop);
    for (const ArrayReference& ref : refs) {
        if (!ref.isWrite) continue;
        if (!ref.baseVariable) continue;
        bool currentLoopVarInAnySubscript = false;
        bool nestedLoopVarInAnySubscript = false;
        for (SgExpression* sub : ref.subscripts) {
            if (subscriptContainsVar(sub, loopVar)) currentLoopVarInAnySubscript = true;
            if (subscriptContainsAnyNestedLoopVar(sub, nestedLoopVars)) nestedLoopVarInAnySubscript = true;
        }
        if (!currentLoopVarInAnySubscript && nestedLoopVarInAnySubscript) {
            result.hasLoopCarriedDependence = true;
            result.description = "write indexed by nested loop variable on " +
                                 ref.baseVariable->get_name().getString();
            result.source = ref.subscripts.empty() ? nullptr : ref.subscripts.front();
            return result;
        }
    }

    for (size_t i = 0; i < refs.size(); ++i) {
        for (size_t j = i + 1; j < refs.size(); ++j) {
            // Only consider pairs where at least one is a write.
            if (!refs[i].isWrite && !refs[j].isWrite) continue;

            // Only compare references to the same base variable.
            if (refs[i].baseVariable != refs[j].baseVariable) continue;
            if (!refs[i].baseVariable) continue;

            // All subscript lists must have the same rank for a precise check.
            // If ranks differ (e.g. A[i] vs A[i][j]), conservatively assume
            // dependence.
            if (refs[i].subscripts.size() != refs[j].subscripts.size()) {
                result.hasLoopCarriedDependence = true;
                result.description = "rank mismatch on " +
                                     refs[i].baseVariable->get_name().getString();
                return result;
            }

            for (size_t d = 0; d < refs[i].subscripts.size(); ++d) {
                if (hasLoopCarriedDependence(refs[i].subscripts[d],
                                             refs[j].subscripts[d],
                                             loopVar)) {
                    result.hasLoopCarriedDependence = true;
                    result.description = "loop-carried dependence detected on " +
                                         refs[i].baseVariable->get_name().getString();
                    result.source = refs[i].subscripts[d];
                    return result;
                }
            }
        }
    }

    result.hasLoopCarriedDependence = false;
    result.description = "no loop-carried dependence detected";
    return result;
}

// Collect subscript expressions from a possibly multi-dimensional array
// reference.  Returns indices from leftmost dimension to rightmost.
static std::vector<SgExpression*> collectSubscripts(SgPntrArrRefExp* arrRef) {
    std::vector<SgExpression*> subs;
    while (arrRef) {
        subs.push_back(arrRef->get_rhs_operand());
        SgExpression* lhs = arrRef->get_lhs_operand();
        arrRef = isSgPntrArrRefExp(lhs);
    }
    std::reverse(subs.begin(), subs.end());
    return subs;
}

std::vector<ArrayReference> LoopDependenceAnalysis::collectArrayReferences(
    SgForStatement* loop) {
    std::vector<ArrayReference> refs;
    if (!loop) return refs;

    SgStatement* body = loop->get_loop_body();
    if (!body) return refs;

    // Collect read and write references.
    std::vector<SgNode*> readRefs, writeRefs;
    SageInterface::collectReadWriteRefs(body, readRefs, writeRefs);

    auto processRef = [&](SgNode* node, bool isWrite) {
        SgExpression* expr = isSgExpression(node);
        if (!expr) return;

        // We are interested in array/pointer dereferences.
        SgPntrArrRefExp* arrRef = isSgPntrArrRefExp(expr);
        if (!arrRef) return;

        ArrayReference ref;
        ref.baseVariable = getBaseVariable(arrRef);
        ref.subscripts = collectSubscripts(arrRef);
        if (!ref.subscripts.empty()) {
            ref.subscriptExpr = ref.subscripts.front();
        }
        ref.isWrite = isWrite;
        refs.push_back(ref);
    };

    for (SgNode* node : readRefs) processRef(node, false);
    for (SgNode* node : writeRefs) processRef(node, true);

    return refs;
}

AffineSubscript LoopDependenceAnalysis::extractAffineSubscript(
    SgExpression* expr, SgInitializedName* loopVar) {
    AffineSubscript result;
    if (!expr || !loopVar) {
        result.note = "null expression or loop variable";
        return result;
    }

    expr = stripCastsAndParens(expr);

    // Case: loop variable itself.
    if (SgVarRefExp* varRef = isSgVarRefExp(expr)) {
        if (varRef->get_symbol()->get_declaration() == loopVar) {
            result.isAffine = true;
            result.coefficient = 1;
            result.constant = 0;
            return result;
        }
        // A variable other than the loop index is not a compile-time
        // constant. Treating it as coefficient-zero can incorrectly prove
        // indirect accesses such as a[indexSet[i]] independent. Unknown
        // subscripts must remain conservative until range/alias analysis can
        // prove disjointness.
        result.note = "unknown symbolic subscript";
        return result;
    }

    // Case: integer constant.
    if (SgIntVal* intVal = isSgIntVal(expr)) {
        result.isAffine = true;
        result.coefficient = 0;
        result.constant = intVal->get_value();
        return result;
    }
    if (SgLongIntVal* longVal = isSgLongIntVal(expr)) {
        result.isAffine = true;
        result.coefficient = 0;
        result.constant = longVal->get_value();
        return result;
    }
    if (SgLongLongIntVal* llVal = isSgLongLongIntVal(expr)) {
        result.isAffine = true;
        result.coefficient = 0;
        result.constant = llVal->get_value();
        return result;
    }

    // Case: binary operation.
    if (SgBinaryOp* binOp = isSgBinaryOp(expr)) {
        SgExpression* lhs = stripCastsAndParens(binOp->get_lhs_operand());
        SgExpression* rhs = stripCastsAndParens(binOp->get_rhs_operand());

        AffineSubscript lhsAffine = extractAffineSubscript(lhs, loopVar);
        AffineSubscript rhsAffine = extractAffineSubscript(rhs, loopVar);

        if (!lhsAffine.isAffine || !rhsAffine.isAffine) {
            result.note = "non-affine sub-expression";
            return result;
        }

        if (isSgAddOp(binOp)) {
            result.isAffine = true;
            result.coefficient = lhsAffine.coefficient + rhsAffine.coefficient;
            result.constant = lhsAffine.constant + rhsAffine.constant;
            return result;
        }
        if (isSgSubtractOp(binOp)) {
            result.isAffine = true;
            result.coefficient = lhsAffine.coefficient - rhsAffine.coefficient;
            result.constant = lhsAffine.constant - rhsAffine.constant;
            return result;
        }
        if (isSgMultiplyOp(binOp)) {
            // Affine only if one side is constant and the other is linear (or constant).
            bool lhsConst = (lhsAffine.coefficient == 0);
            bool rhsConst = (rhsAffine.coefficient == 0);
            if (lhsConst && rhsConst) {
                result.isAffine = true;
                result.coefficient = 0;
                result.constant = lhsAffine.constant * rhsAffine.constant;
                return result;
            }
            if (lhsConst && rhsAffine.coefficient == 1 && rhsAffine.constant == 0) {
                result.isAffine = true;
                result.coefficient = lhsAffine.constant;
                result.constant = 0;
                return result;
            }
            if (rhsConst && lhsAffine.coefficient == 1 && lhsAffine.constant == 0) {
                result.isAffine = true;
                result.coefficient = rhsAffine.constant;
                result.constant = 0;
                return result;
            }
            result.note = "non-linear multiplication";
            return result;
        }
    }

    // Case: unary minus.
    if (SgMinusOp* minusOp = isSgMinusOp(expr)) {
        AffineSubscript inner = extractAffineSubscript(
            stripCastsAndParens(minusOp->get_operand()), loopVar);
        if (inner.isAffine) {
            result.isAffine = true;
            result.coefficient = -inner.coefficient;
            result.constant = -inner.constant;
            return result;
        }
    }

    result.note = "unsupported expression form";
    return result;
}

bool LoopDependenceAnalysis::gcdTest(long long c1, long long c2,
                                     long long delta) const {
    long long g = gcd(c1, c2);
    if (g == 0) {
        // Both coefficients are zero. Dependence exists only if constants equal.
        return (delta == 0);
    }
    return (delta % g) == 0;
}

bool LoopDependenceAnalysis::hasLoopCarriedDependence(
    SgExpression* sub1, SgExpression* sub2,
    SgInitializedName* loopVar) {
    AffineSubscript s1 = extractAffineSubscript(sub1, loopVar);
    AffineSubscript s2 = extractAffineSubscript(sub2, loopVar);

    if (!s1.isAffine || !s2.isAffine) {
        // Conservative: if we cannot analyse the subscript, assume dependence.
        return true;
    }

    // Dependence equation: c1*i1 + k1 = c2*i2 + k2
    // => c1*i1 - c2*i2 = k2 - k1
    long long c1 = s1.coefficient;
    long long c2 = s2.coefficient;
    long long delta = s2.constant - s1.constant;

    if (!gcdTest(c1, c2, delta)) {
        return false; // No integer solution, no dependence.
    }

    // If coefficients and constants are identical, every solution has i1 = i2
    // (same iteration). That is loop-independent and safe for parallelization.
    if (c1 == c2 && s1.constant == s2.constant) {
        return false;
    }

    // A solution exists and the subscripts are not identical. Conservatively
    // treat this as a possible loop-carried dependence.
    return true;
}

} // namespace loomX
