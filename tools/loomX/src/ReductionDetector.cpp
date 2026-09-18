#include "ReductionDetector.h"
#include "LoopAnalysisUtil.h"

namespace loomX {

namespace {

// If expr is a loop-invariant array element access (arr[idx] where none of
// the subscripts contain loopVar), return the base variable and the index
// expression of the outermost dimension.  Otherwise return nullptr.
std::pair<SgInitializedName*, SgExpression*>
isLoopInvariantArrayElement(SgExpression* expr, SgInitializedName* loopVar) {
    if (!expr || !loopVar) return {nullptr, nullptr};
    SgPntrArrRefExp* arrRef = isSgPntrArrRefExp(expr);
    if (!arrRef) return {nullptr, nullptr};

    // The loop variable must not appear anywhere in the full array subscript.
    Rose_STL_Container<SgNode*> allRefs =
        NodeQuery::querySubTree(arrRef, V_SgVarRefExp);
    for (SgNode* node : allRefs) {
        SgVarRefExp* ref = isSgVarRefExp(node);
        if (ref && ref->get_symbol()->get_declaration() == loopVar) {
            return {nullptr, nullptr};
        }
    }

    SgExpression* base = arrRef->get_lhs_operand();
    while (SgPntrArrRefExp* nested = isSgPntrArrRefExp(base)) {
        base = nested->get_lhs_operand();
    }
    SgVarRefExp* baseVar = isSgVarRefExp(base);
    if (!baseVar) return {nullptr, nullptr};
    return {baseVar->get_symbol()->get_declaration(), arrRef->get_rhs_operand()};
}

} // anonymous namespace

std::vector<ReductionInfo> ReductionDetector::analyze(SgForStatement* loop) {
    std::vector<ReductionInfo> results;
    if (!loop) return results;

    SgStatement* body = loop->get_loop_body();
    if (!body) return results;

    Rose_STL_Container<SgNode*> exprStmts =
        NodeQuery::querySubTree(body, V_SgExprStatement);

    for (SgNode* node : exprStmts) {
        SgExprStatement* stmt = isSgExprStatement(node);
        if (!stmt) continue;
        SgExpression* expr = stmt->get_expression();
        if (!expr) continue;

        ReductionInfo info;
        info.statement = stmt;

        // Case 1: compound assignment +=, -=, *=, /=, &=, |=, ^=
        if (SgCompoundAssignOp* compound = isSgCompoundAssignOp(expr)) {
            SgExpression* lhs = compound->get_lhs_operand();

            // Scalar reduction.
            if (SgVarRefExp* lhsVar = isSgVarRefExp(skipCasts(lhs))) {
                info.variable = lhsVar->get_symbol()->get_declaration();
                if (!isLoopLocalVariable(info.variable, loop)) {
                    info.op = detectCompoundAssignOp(compound);
                    if (info.op != ReductionOp::UNKNOWN) {
                        info.opString = opToString(info.op);
                        results.push_back(info);
                    }
                }
                continue;
            }

            // Loop-invariant array element reduction, e.g. tmp[i] += ...
            // inside a loop over j.
            if (SgInitializedName* loopVar = SageInterface::getLoopIndexVariable(loop)) {
                auto arrElem = isLoopInvariantArrayElement(lhs, loopVar);
                if (arrElem.first && !isLoopLocalVariable(arrElem.first, loop)) {
                    info.variable = arrElem.first;
                    info.isArrayElement = true;
                    info.arrayIndex = arrElem.second;
                    info.op = detectCompoundAssignOp(compound);
                    if (info.op != ReductionOp::UNKNOWN) {
                        info.opString = opToString(info.op);
                        results.push_back(info);
                    }
                }
            }
            continue;
        }

        // Case 2: simple assignment x = ...
        if (SgAssignOp* assign = isSgAssignOp(expr)) {
            SgInitializedName* var = nullptr;
            ReductionOp op = ReductionOp::UNKNOWN;
            if (isReductionAssignment(assign, var, op)) {
                if (!isLoopLocalVariable(var, loop)) {
                    info.variable = var;
                    info.op = op;
                    info.opString = opToString(op);
                    results.push_back(info);
                }
                continue;
            }

            // Loop-invariant array element assignment, e.g. tmp[i] = tmp[i] + ...
            if (SgInitializedName* loopVar = SageInterface::getLoopIndexVariable(loop)) {
                SgExpression* lhs = assign->get_lhs_operand();
                auto arrElem = isLoopInvariantArrayElement(lhs, loopVar);
                if (arrElem.first && !isLoopLocalVariable(arrElem.first, loop)) {
                    ReductionOp arrOp = detectArrayElementReductionOp(assign, arrElem.first);
                    if (arrOp != ReductionOp::UNKNOWN) {
                        info.variable = arrElem.first;
                        info.isArrayElement = true;
                        info.arrayIndex = arrElem.second;
                        info.op = arrOp;
                        info.opString = opToString(arrOp);
                        results.push_back(info);
                    }
                }
            }
            continue;
        }

        // Case 3: pre/post increment/decrement of scalar (treat as ADD reduction)
        if (isSgPlusPlusOp(expr) || isSgMinusMinusOp(expr)) {
            SgUnaryOp* uop = isSgUnaryOp(expr);
            SgExpression* operand = uop->get_operand();
            SgVarRefExp* varRef = isSgVarRefExp(skipCasts(operand));
            if (varRef) {
                SgInitializedName* var = varRef->get_symbol()->get_declaration();
                if (isLoopLocalVariable(var, loop)) continue;
                info.variable = var;
                info.op = isSgPlusPlusOp(expr) ? ReductionOp::ADD : ReductionOp::SUB;
                info.opString = opToString(info.op);
                results.push_back(info);
            }
        }
    }

    return results;
}

std::set<SgInitializedName*> ReductionDetector::getReductionVariables(SgForStatement* loop) {
    std::set<SgInitializedName*> vars;
    for (const ReductionInfo& info : analyze(loop)) {
        if (info.variable) vars.insert(info.variable);
    }
    return vars;
}

ReductionOp ReductionDetector::detectCompoundAssignOp(SgCompoundAssignOp* op) {
    if (!op) return ReductionOp::UNKNOWN;
    if (isSgPlusAssignOp(op)) return ReductionOp::ADD;
    if (isSgMinusAssignOp(op)) return ReductionOp::SUB;
    if (isSgMultAssignOp(op)) return ReductionOp::MUL;
    if (isSgDivAssignOp(op)) return ReductionOp::UNKNOWN;  // division is not associative
    if (isSgAndAssignOp(op)) return ReductionOp::BIT_AND;
    if (isSgIorAssignOp(op)) return ReductionOp::BIT_OR;
    if (isSgXorAssignOp(op)) return ReductionOp::BIT_XOR;
    return ReductionOp::UNKNOWN;
}

ReductionOp ReductionDetector::detectBinaryReductionOp(SgInitializedName* var,
                                                        SgExpression* rhs) {
    if (!var || !rhs) return ReductionOp::UNKNOWN;

    // x = x + y  or  x = y + x
    if (SgAddOp* add = isSgAddOp(rhs)) {
        if (exprContainsVar(add->get_lhs_operand(), var) ||
            exprContainsVar(add->get_rhs_operand(), var)) {
            return ReductionOp::ADD;
        }
    }

    // x = x - y  -> subtractive reduction
    if (SgSubtractOp* sub = isSgSubtractOp(rhs)) {
        if (exprContainsVar(sub->get_lhs_operand(), var)) {
            return ReductionOp::SUB;
        }
    }

    // x = x * y  or  x = y * x
    if (SgMultiplyOp* mul = isSgMultiplyOp(rhs)) {
        if (exprContainsVar(mul->get_lhs_operand(), var) ||
            exprContainsVar(mul->get_rhs_operand(), var)) {
            return ReductionOp::MUL;
        }
    }

    // x = x & y  or  x = y & x
    if (SgBitAndOp* band = isSgBitAndOp(rhs)) {
        if (exprContainsVar(band->get_lhs_operand(), var) ||
            exprContainsVar(band->get_rhs_operand(), var)) {
            return ReductionOp::BIT_AND;
        }
    }

    // x = x | y  or  x = y | x
    if (SgBitOrOp* bor = isSgBitOrOp(rhs)) {
        if (exprContainsVar(bor->get_lhs_operand(), var) ||
            exprContainsVar(bor->get_rhs_operand(), var)) {
            return ReductionOp::BIT_OR;
        }
    }

    // x = x ^ y  or  x = y ^ x
    if (SgBitXorOp* bxor = isSgBitXorOp(rhs)) {
        if (exprContainsVar(bxor->get_lhs_operand(), var) ||
            exprContainsVar(bxor->get_rhs_operand(), var)) {
            return ReductionOp::BIT_XOR;
        }
    }

    return ReductionOp::UNKNOWN;
}

// Check whether assign is of the form arr[idx] = arr[idx] op rhs where
// arr[idx] is the same array element on both sides.  Only associative ops
// are treated as reductions.
ReductionOp ReductionDetector::detectArrayElementReductionOp(
    SgAssignOp* assign, SgInitializedName* baseVar) {
    if (!assign || !baseVar) return ReductionOp::UNKNOWN;

    SgExpression* lhs = assign->get_lhs_operand();
    SgExpression* rhs = assign->get_rhs_operand();

    // The RHS must contain the same array element.
    Rose_STL_Container<SgNode*> refs = NodeQuery::querySubTree(rhs, V_SgPntrArrRefExp);
    bool foundMatch = false;
    for (SgNode* node : refs) {
        SgPntrArrRefExp* arrRef = isSgPntrArrRefExp(node);
        if (!arrRef) continue;
        SgExpression* base = arrRef->get_lhs_operand();
        while (SgPntrArrRefExp* nested = isSgPntrArrRefExp(base)) {
            base = nested->get_lhs_operand();
        }
        SgVarRefExp* baseVarRef = isSgVarRefExp(base);
        if (baseVarRef && baseVarRef->get_symbol()->get_declaration() == baseVar) {
        // Simple structural equality of the full array reference is
        // sufficient for the common case (tmp[i] on both sides).
        if (arrRef->unparseToString() == lhs->unparseToString()) {
            foundMatch = true;
            break;
        }
        }
    }
    if (!foundMatch) return ReductionOp::UNKNOWN;

    // Determine the operator from the RHS shape.
    if (SgAddOp* add = isSgAddOp(rhs)) {
        if (exprContainsVar(add->get_lhs_operand(), baseVar) ||
            exprContainsVar(add->get_rhs_operand(), baseVar)) {
            return ReductionOp::ADD;
        }
    }
    if (SgSubtractOp* sub = isSgSubtractOp(rhs)) {
        if (exprContainsVar(sub->get_lhs_operand(), baseVar)) {
            return ReductionOp::SUB;
        }
    }
    if (SgMultiplyOp* mul = isSgMultiplyOp(rhs)) {
        if (exprContainsVar(mul->get_lhs_operand(), baseVar) ||
            exprContainsVar(mul->get_rhs_operand(), baseVar)) {
            return ReductionOp::MUL;
        }
    }
    if (SgBitAndOp* band = isSgBitAndOp(rhs)) {
        if (exprContainsVar(band->get_lhs_operand(), baseVar) ||
            exprContainsVar(band->get_rhs_operand(), baseVar)) {
            return ReductionOp::BIT_AND;
        }
    }
    if (SgBitOrOp* bor = isSgBitOrOp(rhs)) {
        if (exprContainsVar(bor->get_lhs_operand(), baseVar) ||
            exprContainsVar(bor->get_rhs_operand(), baseVar)) {
            return ReductionOp::BIT_OR;
        }
    }
    if (SgBitXorOp* bxor = isSgBitXorOp(rhs)) {
        if (exprContainsVar(bxor->get_lhs_operand(), baseVar) ||
            exprContainsVar(bxor->get_rhs_operand(), baseVar)) {
            return ReductionOp::BIT_XOR;
        }
    }

    return ReductionOp::UNKNOWN;
}

ReductionOp ReductionDetector::detectMinMaxOp(SgInitializedName* var,
                                               SgExpression* rhs,
                                               bool isAssignment) {
    if (!var || !rhs) return ReductionOp::UNKNOWN;

    // x = (x < y) ? x : y   -> min reduction
    // x = (x > y) ? x : y   -> max reduction
    if (SgConditionalExp* cond = isSgConditionalExp(rhs)) {
        SgExpression* condExpr = cond->get_conditional_exp();
        SgExpression* trueExpr = cond->get_true_exp();
        SgExpression* falseExpr = cond->get_false_exp();

        if (exprContainsVar(trueExpr, var) && exprContainsVar(falseExpr, var)) {
            if (SgLessThanOp* lt = isSgLessThanOp(condExpr)) {
                if (exprContainsVar(lt->get_lhs_operand(), var)) return ReductionOp::MIN;
            }
            if (SgGreaterThanOp* gt = isSgGreaterThanOp(condExpr)) {
                if (exprContainsVar(gt->get_lhs_operand(), var)) return ReductionOp::MAX;
            }
        }
    }

    return ReductionOp::UNKNOWN;
}

std::string ReductionDetector::opToString(ReductionOp op) {
    switch (op) {
        case ReductionOp::ADD: return "+";
        case ReductionOp::SUB: return "-";
        case ReductionOp::MUL: return "*";
        case ReductionOp::MIN: return "min";
        case ReductionOp::MAX: return "max";
        case ReductionOp::BIT_AND: return "&";
        case ReductionOp::BIT_OR: return "|";
        case ReductionOp::BIT_XOR: return "^";
        case ReductionOp::LOGICAL_AND: return "&&";
        case ReductionOp::LOGICAL_OR: return "||";
        default: return "";
    }
}

bool ReductionDetector::exprContainsVar(SgExpression* expr, SgInitializedName* var) {
    if (!expr || !var) return false;
    Rose_STL_Container<SgNode*> refs = NodeQuery::querySubTree(expr, V_SgVarRefExp);
    for (SgNode* node : refs) {
        SgVarRefExp* ref = isSgVarRefExp(node);
        if (ref && ref->get_symbol()->get_declaration() == var) return true;
    }
    return false;
}

bool ReductionDetector::isReductionAssignment(SgAssignOp* assign,
                                               SgInitializedName*& outVar,
                                               ReductionOp& outOp) {
    outVar = nullptr;
    outOp = ReductionOp::UNKNOWN;

    if (!assign) return false;

    SgExpression* lhs = assign->get_lhs_operand();
    SgExpression* rhs = assign->get_rhs_operand();

    SgVarRefExp* lhsVar = isSgVarRefExp(skipCasts(lhs));
    if (!lhsVar) return false;

    SgInitializedName* var = lhsVar->get_symbol()->get_declaration();
    if (!var) return false;

    // Try min/max first.
    ReductionOp minmax = detectMinMaxOp(var, rhs, true);
    if (minmax != ReductionOp::UNKNOWN) {
        outVar = var;
        outOp = minmax;
        return true;
    }

    // Try binary reduction patterns.
    ReductionOp binOp = detectBinaryReductionOp(var, rhs);
    if (binOp != ReductionOp::UNKNOWN) {
        outVar = var;
        outOp = binOp;
        return true;
    }

    return false;
}

bool ReductionDetector::isLoopLocalVariable(SgInitializedName* var,
                                            SgForStatement* loop) const {
    if (!var || !loop) return false;

    // Variables declared in the loop initializer are private to each iteration.
    Rose_STL_Container<SgNode*> initStmts =
        NodeQuery::querySubTree(loop->get_for_init_stmt(), V_SgInitializedName);
    for (SgNode* node : initStmts) {
        if (isSgInitializedName(node) == var) return true;
    }

    // Variables declared in the loop body (or any nested scope inside it)
    // are private to the iteration.
    SgStatement* body = loop->get_loop_body();
    SgScopeStatement* varScope = var->get_scope();
    if (!varScope || !body) return false;

    SgNode* current = varScope;
    while (current) {
        if (current == body) return true;
        if (isSgFunctionDefinition(current)) break;
        current = current->get_parent();
    }

    return false;
}

} // namespace loomX
