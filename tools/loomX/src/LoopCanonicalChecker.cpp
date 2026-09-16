#include "LoopCanonicalChecker.h"
#include "LoopAnalysisUtil.h"
#include <sstream>

namespace loomX {

const CanonicalResult& LoopCanonicalChecker::analyze(SgForStatement* loop) {
    if (!loop) {
        static CanonicalResult empty;
        empty.form = CanonicalForm::NON_CANONICAL;
        empty.reason = NonCanonicalReason::NONE;
        empty.note = "null loop";
        return empty;
    }

    auto it = cache_.find(loop);
    if (it != cache_.end()) return it->second;

    CanonicalResult result;

    // Use ROSE's canonical loop test first; it fills in the main fields.
    bool roseCanonical = SageInterface::isCanonicalForLoop(
        loop,
        &result.indexVar,
        &result.lowerBound,
        &result.upperBound,
        &result.stride);

    if (!roseCanonical) {
        result.form = CanonicalForm::NON_CANONICAL;
        result.reason = NonCanonicalReason::NON_LITERAL_OR_PARAMETER_BOUND;
        result.note = "ROSE isCanonicalForLoop returned false";
        cache_[loop] = result;
        return cache_[loop];
    }

    // Validate that we have all pieces.
    if (!result.indexVar) {
        result.form = CanonicalForm::NON_CANONICAL;
        result.reason = NonCanonicalReason::NO_INIT;
        result.note = "No loop-index variable detected";
        cache_[loop] = result;
        return cache_[loop];
    }

    // Require an integer iterator.
    SgType* iterType = result.indexVar->get_type();
    if (!iterType || !(isSgTypeInt(iterType) || isSgTypeLong(iterType) ||
                       isSgTypeShort(iterType) || isSgTypeChar(iterType) ||
                       isSgTypeUnsignedInt(iterType) || isSgTypeUnsignedLong(iterType) ||
                       isSgTypeLongLong(iterType))) {
        result.form = CanonicalForm::NON_CANONICAL;
        result.reason = NonCanonicalReason::NON_INTEGER_ITERATOR;
        result.note = "Loop iterator is not an integral type";
        cache_[loop] = result;
        return cache_[loop];
    }

    // Verify the loop variable is not modified anywhere except by the increment.
    if (!iteratorModifiedOnlyByIncrement(loop, result.indexVar)) {
        result.form = CanonicalForm::NON_CANONICAL;
        result.reason = NonCanonicalReason::LOOP_VAR_MODIFIED_IN_BODY;
        result.note = "Loop iterator is modified inside the loop body";
        cache_[loop] = result;
        return cache_[loop];
    }

    // Determine inclusivity by inspecting the test operator.
    SgStatement* testStmt = loop->get_test();
    if (testStmt) {
        SgExpression* testExpr = nullptr;
        if (SgExprStatement* es = isSgExprStatement(testStmt)) {
            testExpr = es->get_expression();
        }
        if (testExpr) {
            if (isSgLessThanOp(testExpr) || isSgNotEqualOp(testExpr)) {
                result.isUpperExclusive = true;
            } else if (isSgLessOrEqualOp(testExpr)) {
                result.isUpperExclusive = false;
            }
        }
    }

    // Inspect the increment for stride extraction.
    SgExpression* increment = loop->get_increment();
    if (!increment) {
        result.form = CanonicalForm::NON_CANONICAL;
        result.reason = NonCanonicalReason::NO_INCREMENT;
        result.note = "No increment expression";
        cache_[loop] = result;
        return cache_[loop];
    }

    if (!incrementIsSimple(increment, result.indexVar, result.stride)) {
        result.form = CanonicalForm::NON_CANONICAL;
        result.reason = NonCanonicalReason::NON_IDIOMATIC_INCREMENT;
        result.note = "Increment is not a simple +/- of the loop iterator";
        cache_[loop] = result;
        return cache_[loop];
    }

    // Build a short summary.
    result.form = CanonicalForm::CANONICAL;
    std::ostringstream oss;
    oss << "i=";
    if (result.indexVar) oss << result.indexVar->get_name().getString();
    oss << ", lb=";
    if (result.lowerBound) oss << result.lowerBound->unparseToString();
    oss << ", ub=";
    if (result.upperBound) oss << result.upperBound->unparseToString();
    oss << ", stride=";
    if (result.stride) oss << result.stride->unparseToString();
    else oss << "1";
    result.note = oss.str();

    cache_[loop] = result;
    return cache_[loop];
}

bool LoopCanonicalChecker::isCanonical(SgForStatement* loop) {
    return analyze(loop).form == CanonicalForm::CANONICAL;
}

void LoopCanonicalChecker::clear() {
    cache_.clear();
}

bool LoopCanonicalChecker::hasSingleIterator(SgForStatement* loop, SgInitializedName*& iter) {
    iter = SageInterface::getLoopIndexVariable(loop);
    return (iter != nullptr);
}

bool LoopCanonicalChecker::incrementIsSimple(SgExpression* increment,
                                             SgInitializedName* iter,
                                             SgExpression*& stride) {
    if (!increment || !iter) return false;

    // Case 1: ++i / i++ / --i / i--
    if (isSgPlusPlusOp(increment) || isSgMinusMinusOp(increment)) {
        SgExpression* operand = isSgUnaryOp(increment)->get_operand();
        SgVarRefExp* varRef = isSgVarRefExp(skipCasts(operand));
        if (varRef && varRef->get_symbol()->get_declaration() == iter) {
            stride = SageBuilder::buildIntVal(1);
            return true;
        }
        return false;
    }

    // Case 2: i += expr / i -= expr
    SgCompoundAssignOp* compound = isSgCompoundAssignOp(increment);
    if (compound) {
        SgVarRefExp* lhsVar = isSgVarRefExp(skipCasts(compound->get_lhs_operand()));
        if (!lhsVar || lhsVar->get_symbol()->get_declaration() != iter) return false;

        if (isSgPlusAssignOp(compound)) {
            stride = compound->get_rhs_operand();
            return true;
        }
        if (isSgMinusAssignOp(compound)) {
            // Represent negative stride as unary minus of rhs
            stride = SageBuilder::buildMinusOp(compound->get_rhs_operand());
            return true;
        }
        return false;
    }

    // Case 3: i = i + expr / i = expr + i / i = i - expr
    SgAssignOp* assign = isSgAssignOp(increment);
    if (assign) {
        SgVarRefExp* lhsVar = isSgVarRefExp(skipCasts(assign->get_lhs_operand()));
        if (!lhsVar || lhsVar->get_symbol()->get_declaration() != iter) return false;

        SgExpression* rhs = assign->get_rhs_operand();
        // i + expr
        if (SgAddOp* add = isSgAddOp(rhs)) {
            SgVarRefExp* leftVar = isSgVarRefExp(skipCasts(add->get_lhs_operand()));
            if (leftVar && leftVar->get_symbol()->get_declaration() == iter) {
                stride = add->get_rhs_operand();
                return true;
            }
            SgVarRefExp* rightVar = isSgVarRefExp(skipCasts(add->get_rhs_operand()));
            if (rightVar && rightVar->get_symbol()->get_declaration() == iter) {
                stride = add->get_lhs_operand();
                return true;
            }
        }
        // i - expr
        if (SgSubtractOp* sub = isSgSubtractOp(rhs)) {
            SgVarRefExp* leftVar = isSgVarRefExp(skipCasts(sub->get_lhs_operand()));
            if (leftVar && leftVar->get_symbol()->get_declaration() == iter) {
                stride = SageBuilder::buildMinusOp(sub->get_rhs_operand());
                return true;
            }
        }
    }

    return false;
}

bool LoopCanonicalChecker::boundsUseSimpleExpressions(SgForStatement* loop) {
    SgStatement* init = loop->get_for_init_stmt();
    if (!init) return false;
    // ROSE's isCanonicalForLoop already checks this; additional checks can go here.
    return true;
}

bool LoopCanonicalChecker::iteratorModifiedOnlyByIncrement(SgForStatement* loop,
                                                           SgInitializedName* iter) {
    if (!loop || !iter) return false;

    SgStatement* body = loop->get_loop_body();
    if (!body) return true;

    Rose_STL_Container<SgNode*> varRefs =
        NodeQuery::querySubTree(body, V_SgVarRefExp);

    for (SgNode* node : varRefs) {
        SgVarRefExp* varRef = isSgVarRefExp(node);
        if (!varRef) continue;
        if (varRef->get_symbol()->get_declaration() != iter) continue;

        // Skip the increment expression itself (it is part of the for-stmt, not body).
        SgNode* current = varRef;
        while (current && current != body) {
            SgNode* parent = current->get_parent();
            if (parent == loop) break;  // reached the for-statement
            current = parent;
        }
        if (current == loop) continue;  // this ref is in the for-header

        // Check if this reference is on the LHS of any assignment.
        if (isLhsOfAssignment(varRef)) return false;

        // Check if it is operand of ++/--.
        SgNode* parent = varRef->get_parent();
        if (isSgPlusPlusOp(parent) || isSgMinusMinusOp(parent)) return false;
    }

    return true;
}

std::string LoopCanonicalChecker::reasonToString(NonCanonicalReason reason) {
    switch (reason) {
        case NonCanonicalReason::NONE: return "none";
        case NonCanonicalReason::NO_INIT: return "missing initializer";
        case NonCanonicalReason::NO_TEST: return "missing test";
        case NonCanonicalReason::NO_INCREMENT: return "missing increment";
        case NonCanonicalReason::NON_INTEGER_ITERATOR: return "non-integer iterator";
        case NonCanonicalReason::NON_LITERAL_OR_PARAMETER_BOUND: return "non-literal/parameter bound";
        case NonCanonicalReason::NON_UNIT_STRIDE: return "non-unit stride";
        case NonCanonicalReason::NON_IDIOMATIC_INCREMENT: return "non-idiomatic increment";
        case NonCanonicalReason::LOOP_VAR_MODIFIED_IN_BODY: return "iterator modified in body";
        case NonCanonicalReason::MULTIPLE_ITERATORS: return "multiple iterators";
    }
    return "unknown";
}

} // namespace loomX
