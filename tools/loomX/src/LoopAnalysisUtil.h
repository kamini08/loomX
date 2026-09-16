#pragma once
#include "rose.h"

namespace loomX {

// Skip casts and return the underlying expression.
inline SgExpression* skipCasts(SgExpression* expr) {
    while (SgCastExp* cast = isSgCastExp(expr)) {
        expr = cast->get_operand();
    }
    return expr;
}

inline SgNode* skipCasts(SgNode* node) {
    while (SgCastExp* cast = isSgCastExp(node)) {
        node = cast->get_operand();
    }
    return node;
}

// Return true if expr is the left-hand side of an assignment or compound assignment.
inline bool isLhsOfAssignment(SgExpression* expr) {
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
    if (isSgPlusPlusOp(parent) || isSgMinusMinusOp(parent)) return true;

    return false;
}

// For an array/pointer dereference expression, return the underlying variable
// that is being dereferenced. Handles chains like (*(arr + i)).
inline SgInitializedName* getBaseVariable(SgExpression* expr) {
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
        if (SgInitializedName* left = getBaseVariable(bin->get_lhs_operand())) return left;
        return getBaseVariable(bin->get_rhs_operand());
    }
    return nullptr;
}

// Walk up from a reference and look for an enclosing function definition.
inline SgFunctionDefinition* getEnclosingFunctionDefinition(SgNode* node) {
    while (node) {
        if (SgFunctionDefinition* def = isSgFunctionDefinition(node)) return def;
        node = node->get_parent();
    }
    return nullptr;
}

} // namespace loomX
