#include "PragmaAnalysis.h"
#include <algorithm>
#include <cctype>

namespace loomX {

namespace {

std::string lower(std::string text) {
    std::transform(text.begin(), text.end(), text.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return text;
}

bool contains(const std::string& text, const char* token) {
    return text.find(token) != std::string::npos;
}

void parseClauses(PragmaInfo& info) {
    const std::string text = lower(info.text);
    info.hasPrivate = contains(text, "private(");
    info.hasFirstprivate = contains(text, "firstprivate(");
    info.hasLastprivate = contains(text, "lastprivate(");
    info.hasReduction = contains(text, "reduction(");
    info.hasNowait = contains(text, "nowait");
    info.hasOrdered = contains(text, "ordered");
    info.hasDepend = contains(text, "depend(");
    info.hasThreadprivate = contains(text, "threadprivate");
}

} // namespace

PragmaEffect PragmaAnalysis::classify(const std::string& sourceText) {
    std::string text = lower(sourceText);

    if (contains(text, "loomx metadata")) {
        return PragmaEffect::ANALYSIS_ONLY;
    }

    if (contains(text, "loomx semantic synchronization")) {
        return PragmaEffect::SYNCHRONIZATION;
    }

    if (contains(text, "scop") && !contains(text, "omp")) {
        return PragmaEffect::ANALYSIS_ONLY;
    }

    if (contains(text, "omp atomic") || contains(text, "omp critical") ||
        contains(text, "omp barrier") || contains(text, "omp flush") ||
        contains(text, "omp ordered") || contains(text, "omp taskwait") ||
        contains(text, "omp taskgroup") || contains(text, "omp cancel")) {
        return PragmaEffect::SYNCHRONIZATION;
    }

    if (contains(text, "omp target") || contains(text, "omp teams") ||
        contains(text, "omp declare target") || contains(text, "omp end target") ||
        contains(text, "omp target data") || contains(text, "omp target enter data") ||
        contains(text, "omp target exit data") || contains(text, "omp target update")) {
        return PragmaEffect::DEVICE_REGION;
    }

    if (contains(text, "omp map") || contains(text, "omp declare mapper")) {
        return PragmaEffect::DATA_MAPPING;
    }

    if (contains(text, "omp parallel") || contains(text, "omp for") ||
        contains(text, "omp distribute") || contains(text, "omp simd") ||
        contains(text, "omp task") || contains(text, "omp sections")) {
        return PragmaEffect::EXISTING_PARALLEL;
    }

    return PragmaEffect::UNKNOWN;
}

bool PragmaAnalysis::isRelevantToLoop(SgPragmaDeclaration* pragma,
                                      SgForStatement* loop) {
    if (!pragma || !loop) return false;

    SgNode* parent = loop->get_parent();
    if (SgBasicBlock* block = isSgBasicBlock(parent)) {
        const SgStatementPtrList& statements = block->get_statements();
        for (size_t i = 0; i < statements.size(); ++i) {
            if (statements[i] != loop) continue;
            return i > 0 && statements[i - 1] == pragma;
        }
    }

    SgNode* current = pragma->get_parent();
    while (current) {
        if (current == loop->get_loop_body()) return true;
        if (current == loop) return false;
        current = current->get_parent();
    }
    return false;
}

PragmaAnalysisResult PragmaAnalysis::analyze(SgForStatement* loop) const {
    PragmaAnalysisResult result;
    if (!loop) return result;

    Rose_STL_Container<SgNode*> nodes =
        NodeQuery::querySubTree(loop, V_SgPragmaDeclaration);
    if (SgNode* parent = loop->get_parent()) {
        if (SgBasicBlock* block = isSgBasicBlock(parent)) {
            const SgStatementPtrList& statements = block->get_statements();
            for (size_t i = 0; i < statements.size(); ++i) {
                if (statements[i] == loop && i > 0) {
                    if (SgPragmaDeclaration* pragma =
                            isSgPragmaDeclaration(statements[i - 1])) {
                        nodes.push_back(pragma);
                    }
                    break;
                }
            }
        }
    }

    for (SgNode* node : nodes) {
        SgPragmaDeclaration* pragma = isSgPragmaDeclaration(node);
        if (!pragma || !isRelevantToLoop(pragma, loop)) continue;

        PragmaInfo info;
        info.declaration = pragma;
        info.text = pragma->unparseToString();
        info.effect = classify(info.text);
        parseClauses(info);
        result.pragmas.push_back(info);

        result.hasDataSharingClauses =
            result.hasDataSharingClauses || info.hasPrivate ||
            info.hasFirstprivate || info.hasLastprivate;
        result.hasLastprivate = result.hasLastprivate || info.hasLastprivate;
        result.hasReduction = result.hasReduction || info.hasReduction;
        result.hasTaskDependencies = result.hasTaskDependencies || info.hasDepend;
        result.hasOrderedRegion = result.hasOrderedRegion || info.hasOrdered;
        result.hasNowait = result.hasNowait || info.hasNowait;
        result.hasThreadprivate = result.hasThreadprivate || info.hasThreadprivate;

        if (info.effect == PragmaEffect::SYNCHRONIZATION) {
            result.blocksTransformation = true;
            result.hasSynchronization = true;
            result.reason = "existing synchronization/ordering pragma: " + info.text;
        } else if (info.effect == PragmaEffect::EXISTING_PARALLEL ||
                   info.effect == PragmaEffect::DEVICE_REGION) {
            result.blocksTransformation = true;
            result.alreadyParallel = true;
            result.reason = "existing execution pragma: " + info.text;
        } else if (info.effect == PragmaEffect::UNKNOWN) {
            result.blocksTransformation = true;
            result.reason = "unknown pragma semantics: " + info.text;
        }
    }

    return result;
}

} // namespace loomX