#include "IterationCountEstimator.h"
#include "LoopAnalysisUtil.h"
#include <cmath>

namespace loomX {

IterationCountEstimator::IterationCountEstimator(LoopCanonicalChecker& canonicalChecker)
    : canonicalChecker_(canonicalChecker) {}

long IterationCountEstimator::estimate(SgForStatement* loop, bool exactLiteral) {
    const CanonicalResult& canonical = canonicalChecker_.analyze(loop);
    if (canonical.form != CanonicalForm::CANONICAL) {
        return -1;  // Cannot estimate non-canonical loops
    }

    long lower = 0;
    if (canonical.lowerBound) {
        lower = evaluateExpression(canonical.lowerBound);
    }

    long upper = -1;
    if (canonical.upperBound) {
        upper = evaluateExpression(canonical.upperBound);
    }

    long strideVal = 1;
    if (canonical.stride) {
        strideVal = evaluateExpression(canonical.stride);
        if (strideVal == 0) strideVal = 1;  // Guard against division by zero
    }
    if (strideVal < 0) strideVal = -strideVal;

    // If upper bound is symbolic, use a configurable default.
    if (upper < 0) {
        if (exactLiteral) return -1;
        return defaultForSymbolicBound();
    }

    // Adjust for inclusivity/exclusivity.
    long inclusiveUpper = canonical.isUpperExclusive ? (upper - 1) : upper;
    long range = inclusiveUpper - lower + 1;
    if (range <= 0) return 0;

    return (range + strideVal - 1) / strideVal;
}

long IterationCountEstimator::evaluateExpression(SgExpression* expr) {
    if (!expr) return -1;

    expr = skipCasts(expr);

    // Integer literal
    if (SgIntVal* intVal = isSgIntVal(expr)) {
        return intVal->get_value();
    }
    if (SgLongIntVal* longVal = isSgLongIntVal(expr)) {
        return longVal->get_value();
    }
    if (SgLongLongIntVal* llVal = isSgLongLongIntVal(expr)) {
        return static_cast<long>(llVal->get_value());
    }
    if (SgShortVal* shortVal = isSgShortVal(expr)) {
        return shortVal->get_value();
    }

    // Variable reference: try initializer
    if (SgVarRefExp* varRef = isSgVarRefExp(expr)) {
        SgInitializedName* var = varRef->get_symbol()->get_declaration();
        if (!var) return -1;

        SgInitializer* init = var->get_initializer();
        if (!init) return -1;

        if (SgIntVal* initInt = isSgIntVal(init)) return initInt->get_value();
        if (SgLongIntVal* initLong = isSgLongIntVal(init)) return initLong->get_value();

        if (SgAssignInitializer* assignInit = isSgAssignInitializer(init)) {
            return evaluateExpression(assignInit->get_operand());
        }
        return -1;
    }

    // Unary minus
    if (SgMinusOp* minus = isSgMinusOp(expr)) {
        long val = evaluateExpression(minus->get_operand());
        return (val >= 0) ? -val : -1;
    }

    // Binary operations
    if (SgBinaryOp* binOp = isSgBinaryOp(expr)) {
        long lhs = evaluateExpression(binOp->get_lhs_operand());
        long rhs = evaluateExpression(binOp->get_rhs_operand());

        // If either side is symbolic, we cannot compute a constant.
        if (lhs < 0 || rhs < 0) return -1;

        if (isSgAddOp(expr)) return lhs + rhs;
        if (isSgSubtractOp(expr)) return lhs - rhs;
        if (isSgMultiplyOp(expr)) return lhs * rhs;
        if (isSgDivideOp(expr)) {
            if (rhs == 0) return -1;
            return lhs / rhs;
        }
        if (isSgIntegerDivideOp(expr)) {
            if (rhs == 0) return -1;
            return lhs / rhs;
        }
        if (isSgModOp(expr)) {
            if (rhs == 0) return -1;
            return lhs % rhs;
        }
    }

    return -1;
}

long IterationCountEstimator::applyStride(long iterations, SgExpression* stride) {
    if (!stride) return iterations;
    long s = evaluateExpression(stride);
    if (s <= 0) return iterations;
    return (iterations + s - 1) / s;
}

long IterationCountEstimator::defaultForSymbolicBound() {
    // Heuristic: assume large when we cannot prove otherwise.
    // This is intentionally optimistic for GPU profitability analysis.
    return 100000;
}

} // namespace loomX
