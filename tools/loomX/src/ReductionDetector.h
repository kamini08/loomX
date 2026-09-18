#pragma once
#include "LoopAnalysisTypes.h"

namespace loomX {

// Detects reduction variables and their operators inside a loop.
// Recognizes compound assignments (+=, *=, etc.), binary assignments
// (x = x + y), and min/max reduction idioms.
class ReductionDetector {
public:
    ReductionDetector() = default;

    // Analyze a loop and return all detected reductions.
    std::vector<ReductionInfo> analyze(SgForStatement* loop);

    // Convenience: return only the variables participating in reductions.
    std::set<SgInitializedName*> getReductionVariables(SgForStatement* loop);

private:
    ReductionOp detectCompoundAssignOp(SgCompoundAssignOp* op);
    ReductionOp detectBinaryReductionOp(SgInitializedName* var, SgExpression* rhs);
    ReductionOp detectArrayElementReductionOp(SgAssignOp* assign, SgInitializedName* baseVar);
    ReductionOp detectMinMaxOp(SgInitializedName* var, SgExpression* rhs, bool isAssignment);
    std::string opToString(ReductionOp op);
    bool exprContainsVar(SgExpression* expr, SgInitializedName* var);
    bool isReductionAssignment(SgAssignOp* assign, SgInitializedName*& outVar, ReductionOp& outOp);

    // True if the variable is declared inside the given loop (body or
    // initializer). Loop-local variables are private, not reductions.
    bool isLoopLocalVariable(SgInitializedName* var, SgForStatement* loop) const;
};

} // namespace loomX
