#include "ComputeIntensityEstimator.h"
#include "LoopAnalysisUtil.h"
#include <cmath>
#include <set>

namespace loomX {

ComputeIntensityResult ComputeIntensityEstimator::analyze(SgForStatement* loop,
                                                           double targetFLOPsPerMemOp) {
    ComputeIntensityResult result;
    if (!loop) {
        result.note = "null loop";
        return result;
    }

    countOperations(loop, result.flopCount, result.memoryOpCount,
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

void ComputeIntensityEstimator::countOperations(SgForStatement* loop,
                                                 int& flops,
                                                 int& memOps,
                                                 int& intOps,
                                                 bool& heavyMath) {
    flops = 0;
    memOps = 0;
    intOps = 0;
    heavyMath = false;

    if (!loop) return;
    SgStatement* body = loop->get_loop_body();
    if (!body) return;

    // Memory operations: array references and pointer dereferences.
    Rose_STL_Container<SgNode*> arrRefs =
        NodeQuery::querySubTree(body, V_SgPntrArrRefExp);
    memOps += static_cast<int>(arrRefs.size());

    Rose_STL_Container<SgNode*> derefRefs =
        NodeQuery::querySubTree(body, V_SgPointerDerefExp);
    memOps += static_cast<int>(derefRefs.size());

    // Count binary operations.
    Rose_STL_Container<SgNode*> binOps =
        NodeQuery::querySubTree(body, V_SgBinaryOp);
    for (SgNode* node : binOps) {
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
            flops++;
        } else if (isSgAddOp(binOp) || isSgSubtractOp(binOp) ||
                   isSgMultiplyOp(binOp) || isSgDivideOp(binOp) ||
                   isSgModOp(binOp) || isSgIntegerDivideOp(binOp) ||
                   isSgBitAndOp(binOp) || isSgBitOrOp(binOp) ||
                   isSgBitXorOp(binOp) || isSgLshiftOp(binOp) ||
                   isSgRshiftOp(binOp)) {
            intOps++;
        }
    }

    // Count unary +/- on FP values.
    Rose_STL_Container<SgNode*> unaryOps =
        NodeQuery::querySubTree(body, V_SgUnaryOp);
    for (SgNode* node : unaryOps) {
        SgUnaryOp* uop = isSgUnaryOp(node);
        if (!uop) continue;
        if (isSgMinusOp(uop) || isSgUnaryAddOp(uop)) {
            if (isFloatingPointType(uop->get_type())) {
                flops++;
            } else {
                intOps++;
            }
        }
    }

    // Detect heavy math function calls.
    Rose_STL_Container<SgNode*> calls =
        NodeQuery::querySubTree(body, V_SgFunctionCallExp);
    for (SgNode* node : calls) {
        SgFunctionCallExp* call = isSgFunctionCallExp(node);
        if (!call) continue;
        SgFunctionDeclaration* decl = call->getAssociatedFunctionDeclaration();
        if (!decl) continue;
        if (isHeavyMathFunction(decl->get_name().getString())) {
            heavyMath = true;
            // Count each heavy math call as ~8 FLOPs equivalent.
            flops += 8;
        }
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

} // namespace loomX
