#include "GpuProfitability.h"
#include <iostream>
#include <cmath>

ParallelTarget GpuProfitability::classifyLoop(SgForStatement* loop) {
    long iterations = estimateIterationCount(loop);
    bool regular = hasRegularAccessPattern(loop);
    bool divergent = hasDivergentControlFlow(loop);
    bool computeHeavy = isComputeIntensive(loop);

    std::cout << "[GpuProfitability] Loop at line "
              << loop->get_file_info()->get_line()
              << " iterations=" << iterations
              << " regular=" << regular
              << " divergent=" << divergent
              << " computeHeavy=" << computeHeavy
              << "\n";

    // Decision tree (calibrated for demo benchmarks)
    if (iterations < 100) {
        // Too small even for CPU thread launch
        return ParallelTarget::SEQUENTIAL;
    }

    if (!regular || divergent) {
        // Irregular: CPU only if large enough
        if (iterations >= 1000) {
            return ParallelTarget::CPU_OPENMP;
        }
        return ParallelTarget::SEQUENTIAL;
    }

    if (iterations >= 100000 && computeHeavy) {
        // Good GPU candidate: large iteration count, regular access, compute-heavy
        return ParallelTarget::GPU_OFFLOAD;
    }

    if (iterations >= 100) {
        // Moderate candidate: CPU parallelization
        return ParallelTarget::CPU_OPENMP;
    }

    return ParallelTarget::SEQUENTIAL;
}

// Try to evaluate a constant expression to an integer value
// Returns -1 if the expression cannot be evaluated statically
long GpuProfitability::evaluateExpression(SgExpression* expr) {
    if (!expr) return -1;

    // Direct integer literal
    SgIntVal* intVal = isSgIntVal(expr);
    if (intVal) {
        return intVal->get_value();
    }

    // Variable reference: try to look up its initializer
    SgVarRefExp* varRef = isSgVarRefExp(expr);
    if (varRef) {
        SgInitializedName* var = varRef->get_symbol()->get_declaration();
        if (var) {
            SgInitializer* init = var->get_initializer();
            if (init) {
                SgIntVal* initInt = isSgIntVal(init);
                if (initInt) return initInt->get_value();

                SgAssignInitializer* assignInit = isSgAssignInitializer(init);
                if (assignInit) {
                    return evaluateExpression(assignInit->get_operand());
                }
            }
        }
        return -1;
    }

    // Binary operations: +, -, *, / on constant operands
    SgBinaryOp* binOp = isSgBinaryOp(expr);
    if (binOp) {
        long lhs = evaluateExpression(binOp->get_lhs_operand());
        long rhs = evaluateExpression(binOp->get_rhs_operand());
        if (lhs < 0 || rhs < 0) return -1;

        if (isSgAddOp(expr)) return lhs + rhs;
        if (isSgSubtractOp(expr)) return lhs - rhs;
        if (isSgMultiplyOp(expr)) return lhs * rhs;
        if (isSgDivideOp(expr)) {
            if (rhs == 0) return -1;
            return lhs / rhs;
        }
    }

    return -1;
}

long GpuProfitability::estimateIterationCount(SgForStatement* loop) {
    // Try to extract loop bounds from canonical form
    SgInitializedName* ivar = nullptr;
    SgExpression* lowerBound = nullptr;
    SgExpression* upperBound = nullptr;
    SgExpression* stride = nullptr;

    bool isCanonical = SageInterface::isCanonicalForLoop(loop, &ivar, &lowerBound, &upperBound, &stride);
    if (!isCanonical) return -1;  // Unknown

    long boundVal = evaluateExpression(upperBound);
    if (boundVal >= 0) {
        return boundVal;
    }

    // Default: assume large symbolic bounds
    return 100000;
}

bool GpuProfitability::hasRegularAccessPattern(SgForStatement* loop) {
    // Check all array references in the loop
    Rose_STL_Container<SgNode*> arrRefs =
        NodeQuery::querySubTree(loop, V_SgPntrArrRefExp);

    for (SgNode* node : arrRefs) {
        SgPntrArrRefExp* arrRef = isSgPntrArrRefExp(node);
        if (!arrRef) continue;

        // Get the index expression
        SgExpression* index = arrRef->get_rhs_operand();
        if (!index) continue;

        // Check if index is a simple variable (loop index) or linear expression
        // For now, accept any non-function-call index as "regular enough"
        Rose_STL_Container<SgNode*> calls =
            NodeQuery::querySubTree(index, V_SgFunctionCallExp);
        if (!calls.empty()) {
            return false;  // Function call in array index = irregular
        }
    }

    return true;
}

bool GpuProfitability::hasDivergentControlFlow(SgForStatement* loop) {
    // Count if/else statements with data-dependent conditions
    Rose_STL_Container<SgNode*> ifStmts =
        NodeQuery::querySubTree(loop, V_SgIfStmt);

    for (SgNode* node : ifStmts) {
        SgIfStmt* ifStmt = isSgIfStmt(node);
        if (!ifStmt) continue;

        SgStatement* condition = ifStmt->get_conditional();
        if (!condition) continue;

        // Check if condition depends on array elements (data-dependent)
        Rose_STL_Container<SgNode*> arrRefs =
            NodeQuery::querySubTree(condition, V_SgPntrArrRefExp);
        if (!arrRefs.empty()) {
            return true;  // Data-dependent branching
        }
    }

    return false;
}

bool GpuProfitability::isComputeIntensive(SgForStatement* loop) {
    // Simple heuristic: count arithmetic operations vs. memory accesses
    Rose_STL_Container<SgNode*> binOps =
        NodeQuery::querySubTree(loop, V_SgBinaryOp);
    Rose_STL_Container<SgNode*> arrRefs =
        NodeQuery::querySubTree(loop, V_SgPntrArrRefExp);

    int ops = binOps.size();
    int mem = arrRefs.size();

    // Compute-intensive if more ops than memory accesses
    return ops > mem;
}
