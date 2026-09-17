#pragma once

#include "rose.h"
#include <string>
#include <vector>

namespace loomX {

enum class PragmaEffect {
    NONE,
    ANALYSIS_ONLY,
    EXISTING_PARALLEL,
    DATA_MAPPING,
    SYNCHRONIZATION,
    DEVICE_REGION,
    UNKNOWN
};

struct PragmaInfo {
    PragmaEffect effect = PragmaEffect::NONE;
    std::string text;
    SgPragmaDeclaration* declaration = nullptr;
};

struct PragmaAnalysisResult {
    std::vector<PragmaInfo> pragmas;
    bool blocksTransformation = false;
    bool alreadyParallel = false;
    bool hasSynchronization = false;
    std::string reason;
};

class PragmaAnalysis {
public:
    PragmaAnalysisResult analyze(SgForStatement* loop) const;

private:
    static PragmaEffect classify(const std::string& text);
    static bool isRelevantToLoop(SgPragmaDeclaration* pragma,
                                 SgForStatement* loop);
};

} // namespace loomX