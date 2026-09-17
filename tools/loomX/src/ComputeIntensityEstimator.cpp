#include "ComputeIntensityEstimator.h"
#include "LoopAnalysisUtil.h"
#include <cmath>
#include <set>

namespace loomX {

ComputeIntensityEstimator::ComputeIntensityEstimator()
    : iterationEstimator_(canonicalChecker_) {}

ComputeIntensityResult ComputeIntensityEstimator::analyze(SgForStatement* loop,
                                                           double targetFLOPsPerMemOp) {
    ComputeIntensityResult result;
    if (!loop) {
        result.note = "null loop";
        return result;
    }

    long long tripCount = iterationEstimator_.estimate(loop, false);
    if (tripCount < 0) tripCount = 1;

    SgStatement* body = loop->get_loop_body();
    countOperations(body, tripCount, result.flopCount, result.memoryOpCount,
                    result.integerOpCount, result.hasHeavyMath);

    if (result.memoryOpCount == 0) {
        // Pure compute kernel with no visible memory ops in the loop body.
        result.flopsPerMemoryOp = static_cast<double>(result.flopCount);
        result.classification = IntensityClass::COMPUTE_BOUND;
        result.note = "No memory accesses detected; compute-bound";
        return result;
    }

    result.flopsPerMemoryOp =
        static_cast<double>(result.flopCount) /
        static_cast<double>(result.memoryOpCount);

    result.accessPattern = classifyAccessPattern(loop);

    if (result.hasHeavyMath) {
        // Transcendentals/sqrt/log are expensive on most GPUs.
        result.classification = IntensityClass::COMPUTE_BOUND;
        result.note = "Heavy math functions present; compute-bound";
    } else if (result.flopsPerMemoryOp >= targetFLOPsPerMemOp) {
        result.classification = IntensityClass::COMPUTE_BOUND;
        result.note = "High FLOP/memory ratio; compute-bound";
    } else if (result.flopsPerMemoryOp >= targetFLOPsPerMemOp / 4.0) {
        result.classification = IntensityClass::BALANCED;
        result.note = "Balanced compute and memory";
    } else {
        result.classification = IntensityClass::MEMORY_BOUND;
        result.note = "Low FLOP/memory ratio; memory-bound";
    }

    return result;
}

static bool isInsideNestedLoop(SgNode* node, SgStatement* enclosingBody) {
    SgNode* current = node->get_parent();
    while (current && current != enclosingBody) {
        if (isSgForStatement(current)) return true;
        current = current->get_parent();
    }
    return false;
}

void ComputeIntensityEstimator::countOperations(SgStatement* body,
                                                 long long tripCount,
                                                 long long& flops,
                                                 long long& memOps,
                                                 long long& intOps,
                                                 bool& heavyMath) {
    if (!body) return;

    // Memory operations: array references and pointer dereferences.
    // Skip those that are inside a nested loop; those are handled recursively.
    Rose_STL_Container<SgNode*> arrRefs =
        NodeQuery::querySubTree(body, V_SgPntrArrRefExp);
    for (SgNode* node : arrRefs) {
        if (!isInsideNestedLoop(node, body)) {
            memOps += tripCount;
        }
    }

    Rose_STL_Container<SgNode*> derefRefs =
        NodeQuery::querySubTree(body, V_SgPointerDerefExp);
    for (SgNode* node : derefRefs) {
        if (!isInsideNestedLoop(node, body)) {
            memOps += tripCount;
        }
    }

    // Count binary operations.
    Rose_STL_Container<SgNode*> binOps =
        NodeQuery::querySubTree(body, V_SgBinaryOp);
    for (SgNode* node : binOps) {
        if (isInsideNestedLoop(node, body)) continue;

        SgBinaryOp* binOp = isSgBinaryOp(node);
        if (!binOp) continue;

        // Exclude assignments from FLOP counting (they are bookkeeping).
        if (isSgAssignOp(binOp) ||
            isSgPlusAssignOp(binOp) || isSgMinusAssignOp(binOp) ||
            isSgMultAssignOp(binOp) || isSgDivAssignOp(binOp) ||
            isSgModAssignOp(binOp) || isSgAndAssignOp(binOp) ||
            isSgIorAssignOp(binOp) || isSgXorAssignOp(binOp) ||
            isSgLshiftAssignOp(binOp) || isSgRshiftAssignOp(binOp)) {
            continue;
        }

        if (isFloatingPointOp(binOp)) {
            flops += tripCount;
        } else if (isSgAddOp(binOp) || isSgSubtractOp(binOp) ||
                   isSgMultiplyOp(binOp) || isSgDivideOp(binOp) ||
                   isSgModOp(binOp) || isSgIntegerDivideOp(binOp) ||
                   isSgBitAndOp(binOp) || isSgBitOrOp(binOp) ||
                   isSgBitXorOp(binOp) || isSgLshiftOp(binOp) ||
                   isSgRshiftOp(binOp)) {
            intOps += tripCount;
        }
    }

    // Count unary +/- on FP values.
    Rose_STL_Container<SgNode*> unaryOps =
        NodeQuery::querySubTree(body, V_SgUnaryOp);
    for (SgNode* node : unaryOps) {
        if (isInsideNestedLoop(node, body)) continue;

        SgUnaryOp* uop = isSgUnaryOp(node);
        if (!uop) continue;
        if (isSgMinusOp(uop) || isSgUnaryAddOp(uop)) {
            if (isFloatingPointType(uop->get_type())) {
                flops += tripCount;
            } else {
                intOps += tripCount;
            }
        }
    }

    // Detect heavy math function calls.
    Rose_STL_Container<SgNode*> calls =
        NodeQuery::querySubTree(body, V_SgFunctionCallExp);
    for (SgNode* node : calls) {
        if (isInsideNestedLoop(node, body)) continue;

        SgFunctionCallExp* call = isSgFunctionCallExp(node);
        if (!call) continue;
        SgFunctionDeclaration* decl = call->getAssociatedFunctionDeclaration();
        if (!decl) continue;
        if (isHeavyMathFunction(decl->get_name().getString())) {
            heavyMath = true;
            // Count each heavy math call as ~8 FLOPs equivalent.
            flops += 8 * tripCount;
        }
    }

    // Recurse into nested canonical loops, multiplying their body counts by
    // their own trip counts so that compute intensity reflects total work.
    Rose_STL_Container<SgNode*> nestedLoops =
        NodeQuery::querySubTree(body, V_SgForStatement);
    for (SgNode* node : nestedLoops) {
        SgForStatement* nested = isSgForStatement(node);
        if (!nested) continue;

        long nestedTrip = iterationEstimator_.estimate(nested, false);
        if (nestedTrip < 0) nestedTrip = 1;

        SgStatement* nestedBody = nested->get_loop_body();
        countOperations(nestedBody, nestedTrip * tripCount,
                        flops, memOps, intOps, heavyMath);
    }
}

bool ComputeIntensityEstimator::isFloatingPointOp(SgBinaryOp* op) {
    if (!op) return false;
    return isFloatingPointType(op->get_type());
}

bool ComputeIntensityEstimator::isFloatingPointType(SgType* type) {
    if (!type) return false;
    type = type->stripType(SgType::STRIP_TYPEDEF_TYPE);
    return isSgTypeFloat(type) || isSgTypeDouble(type) ||
           isSgTypeLongDouble(type);
}

bool ComputeIntensityEstimator::isHeavyMathFunction(const std::string& name) {
    static const std::set<std::string> heavy = {
        "sin", "cos", "tan", "asin", "acos", "atan", "atan2",
        "sinh", "cosh", "tanh", "asinh", "acosh", "atanh",
        "exp", "exp2", "expm1", "log", "log2", "log10", "log1p",
        "sqrt", "cbrt", "pow", "hypot", "erf", "erfc", "tgamma", "lgamma"
    };
    return heavy.count(name) > 0;
}

// Collect index expressions from a (possibly multi-dimensional) array reference.
// For A[i][k] this returns {i, k} in left-to-right order.
static void collectArrayIndices(SgPntrArrRefExp* arrRef,
                                std::vector<SgExpression*>& indices) {
    if (!arrRef) return;
    SgExpression* lhs = arrRef->get_lhs_operand();
    SgExpression* rhs = arrRef->get_rhs_operand();
    if (SgPntrArrRefExp* inner = isSgPntrArrRefExp(lhs)) {
        collectArrayIndices(inner, indices);
    }
    indices.push_back(rhs);
}

AccessPattern ComputeIntensityEstimator::classifyAccessPattern(SgForStatement* loop) {
    if (!loop) return AccessPattern::UNKNOWN;

    const CanonicalResult& canonical = canonicalChecker_.analyze(loop);
    SgInitializedName* loopVar = canonical.indexVar;
    if (!loopVar) return AccessPattern::UNKNOWN;

    AccessPattern worst = AccessPattern::UNKNOWN;

    SgStatement* body = loop->get_loop_body();
    Rose_STL_Container<SgNode*> arrRefs =
        NodeQuery::querySubTree(body, V_SgPntrArrRefExp);

    for (SgNode* node : arrRefs) {
        SgPntrArrRefExp* arrRef = isSgPntrArrRefExp(node);
        if (!arrRef) continue;

        std::vector<SgExpression*> indices;
        collectArrayIndices(arrRef, indices);
        if (indices.empty()) continue;

        // Determine which dimension the loop variable appears in.  In C the
        // rightmost dimension is contiguous in memory.
        bool loopVarFound = false;
        bool inRightmost = false;
        bool inOtherDimension = false;
        bool complexIndex = false;

        for (size_t d = 0; d < indices.size(); ++d) {
            AccessPattern pat = classifyIndexExpression(indices[d], loopVar);
            if (pat == AccessPattern::IRREGULAR) {
                complexIndex = true;
            } else if (pat == AccessPattern::UNIT_STRIDE ||
                       pat == AccessPattern::STRIDED) {
                loopVarFound = true;
                if (d == indices.size() - 1) {
                    inRightmost = true;
                } else {
                    inOtherDimension = true;
                }
            }
        }

        if (!loopVarFound) continue;

        if (complexIndex) {
            return AccessPattern::IRREGULAR;
        }
        if (inOtherDimension) {
            worst = AccessPattern::STRIDED;
        } else if (inRightmost) {
            if (worst == AccessPattern::UNKNOWN) {
                worst = AccessPattern::UNIT_STRIDE;
            }
        }
    }

    return worst;
}

AccessPattern ComputeIntensityEstimator::classifyIndexExpression(
    SgExpression* index, SgInitializedName* loopVar) const {
    if (!index || !loopVar) return AccessPattern::IRREGULAR;

    index = skipCasts(index);

    // Exactly the loop variable.
    if (SgVarRefExp* varRef = isSgVarRefExp(index)) {
        if (varRef->get_symbol()->get_declaration() == loopVar)
            return AccessPattern::UNIT_STRIDE;
        return AccessPattern::UNKNOWN;
    }

    // Binary expressions: look for loopVar * const, const * loopVar,
    // loopVar + const, loopVar - const.
    if (SgBinaryOp* binOp = isSgBinaryOp(index)) {
        SgExpression* lhs = skipCasts(binOp->get_lhs_operand());
        SgExpression* rhs = skipCasts(binOp->get_rhs_operand());

        bool lhsIsLoopVar = false;
        bool rhsIsLoopVar = false;
        if (SgVarRefExp* v = isSgVarRefExp(lhs)) {
            lhsIsLoopVar = (v->get_symbol()->get_declaration() == loopVar);
        }
        if (SgVarRefExp* v = isSgVarRefExp(rhs)) {
            rhsIsLoopVar = (v->get_symbol()->get_declaration() == loopVar);
        }

        bool lhsIsConst = (isSgIntVal(lhs) || isSgLongIntVal(lhs) ||
                           isSgLongLongIntVal(lhs));
        bool rhsIsConst = (isSgIntVal(rhs) || isSgLongIntVal(rhs) ||
                           isSgLongLongIntVal(rhs));

        if (isSgAddOp(index) || isSgSubtractOp(index)) {
            if ((lhsIsLoopVar && rhsIsConst) || (rhsIsLoopVar && lhsIsConst))
                return AccessPattern::UNIT_STRIDE;
        }
        if (isSgMultiplyOp(index)) {
            if ((lhsIsLoopVar && rhsIsConst) || (rhsIsLoopVar && lhsIsConst))
                return AccessPattern::STRIDED;
        }

        // Loop variable in a more complex expression => irregular.
        if (lhsIsLoopVar || rhsIsLoopVar) return AccessPattern::IRREGULAR;
        return AccessPattern::UNKNOWN;
    }

    return AccessPattern::UNKNOWN;
}

} // namespace loomX
