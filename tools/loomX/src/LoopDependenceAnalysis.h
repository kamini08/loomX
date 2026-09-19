#pragma once
#include "rose.h"
#include <set>
#include <string>
#include <vector>

namespace loomX {

// Affine subscript representation: coeff * loopIndex + constant.
// If isAffine is false, the subscript could not be expressed linearly.
struct AffineSubscript {
    bool isAffine = false;
    long long coefficient = 0;   // Coefficient of loop index
    long long constant = 0;      // Constant offset
    std::string note;            // Human-readable detail
};

// A memory reference inside a loop: array base + subscript expressions.
// Multi-dimensional references store one expression per dimension.
struct ArrayReference {
    SgInitializedName* baseVariable = nullptr;            // Array/pointer variable
    std::vector<SgExpression*> subscripts;                // Index expressions (outermost first)
    SgExpression* subscriptExpr = nullptr;                // Legacy: outermost subscript
    bool isWrite = false;                                 // Read or write access
    std::string note;
};

// Result of dependence analysis for a single loop.
struct DependenceResult {
    bool hasLoopCarriedDependence = false;
    std::string description;
    SgNode* source = nullptr;  // Reference involved in the dependence
};

// Basic loop-carried dependence analysis for single canonical loops.
//
// The analysis currently handles one-dimensional array/pointer subscripts that
// are affine in the loop index (e.g., A[i], A[i-1], A[2*i+3]). For each pair
// of references to the same array, it applies the GCD test to decide whether
// two different iterations could access the same element. If so, the loop is
// marked as having a loop-carried dependence and should not be parallelized.
class LoopDependenceAnalysis {
public:
    // Analyse a single for-loop. Returns true if the loop is safe to
    // parallelize (no detected loop-carried dependences).
    DependenceResult analyze(SgForStatement* loop);

private:
    // Collect array/pointer references in the loop body.
    std::vector<ArrayReference> collectArrayReferences(SgForStatement* loop);

    // Try to express an index expression as coeff*loopIndex + constant.
    AffineSubscript extractAffineSubscript(SgExpression* expr,
                                           SgInitializedName* loopVar);

    // GCD test for two affine subscripts.
    bool gcdTest(long long c1, long long c2, long long delta) const;

    // Check whether two subscript expressions to the same array dimension
    // could be loop-carried.
    bool hasLoopCarriedDependence(SgExpression* sub1,
                                  SgExpression* sub2,
                                  SgInitializedName* loopVar);

    // Induction variables of the loop nest being analysed.  These are treated
    // as symbolic constants in subscript expressions.
    std::set<SgInitializedName*> knownLoopVars_;
};

} // namespace loomX
