#include "OpenMPCodeGen.h"
#include "LoopAnalysisUtil.h"
#include <set>
#include <sstream>

using namespace loomX;

void OpenMPCodeGen::generatePragmas(const loomX::LoopSummary& summary) {
    switch (summary.target) {
        case ParallelTarget::CPU_OPENMP:
            insertCPUPragma(summary);
            break;
        case ParallelTarget::GPU_OFFLOAD:
            insertDeclareTargetPragmas(summary);
            insertGPUPragma(summary);
            break;
        case ParallelTarget::SEQUENTIAL:
            // No pragma inserted.
            break;
    }
}

void OpenMPCodeGen::insertCPUPragma(const loomX::LoopSummary& summary) {
    SgForStatement* loop = summary.loop;
    if (!loop) return;

    std::ostringstream pragmaText;
    pragmaText << "omp parallel for";

    if (!summary.privateVars.empty()) {
        pragmaText << " private(" << buildVarList(summary.privateVars) << ")";
    }

    if (!summary.reductions.empty()) {
        pragmaText << " " << buildReductionClause(summary.reductions);
    }

    SgPragmaDeclaration* pragmaDecl =
        SageBuilder::buildPragmaDeclaration(pragmaText.str(),
                                            SageInterface::getScope(loop));
    SageInterface::insertStatementBefore(loop, pragmaDecl);
}

void OpenMPCodeGen::insertGPUPragma(const loomX::LoopSummary& summary) {
    SgForStatement* loop = summary.loop;
    if (!loop) return;

    std::ostringstream pragmaText;
    pragmaText << "omp target teams distribute parallel for";

    if (!summary.privateVars.empty()) {
        pragmaText << " private(" << buildVarList(summary.privateVars) << ")";
    }

    if (!summary.reductions.empty()) {
        pragmaText << " " << buildReductionClause(summary.reductions);
    }

    if (!summary.mapClauses.empty()) {
        pragmaText << " " << buildMapClause(summary.mapClauses);
    }

    SgPragmaDeclaration* pragmaDecl =
        SageBuilder::buildPragmaDeclaration(pragmaText.str(),
                                            SageInterface::getScope(loop));
    SageInterface::insertStatementBefore(loop, pragmaDecl);
}

void OpenMPCodeGen::insertDeclareTargetPragmas(const loomX::LoopSummary& summary) {
    SgForStatement* loop = summary.loop;
    if (!loop) return;

    // Find every function called inside the loop and mark its definition
    // with #pragma omp declare target so it can be used on the GPU.
    Rose_STL_Container<SgNode*> calls =
        NodeQuery::querySubTree(loop, V_SgFunctionCallExp);

    for (SgNode* node : calls) {
        SgFunctionCallExp* call = isSgFunctionCallExp(node);
        if (!call) continue;

        SgFunctionDeclaration* callee = call->getAssociatedFunctionDeclaration();
        if (!callee) continue;

        // The declaration associated with a call may be non-defining.
        // Look up the defining declaration to find the function body.
        SgDeclarationStatement* definingDecl = callee->get_definingDeclaration();
        SgFunctionDeclaration* definingFunc = isSgFunctionDeclaration(definingDecl);
        if (!definingFunc) continue;

        SgFunctionDefinition* def = definingFunc->get_definition();
        if (!def) continue;

        SgFunctionDeclaration* decl = def->get_declaration();
        if (!decl) continue;

        // Avoid inserting duplicate declare-target pragmas.
        bool alreadyDeclared = false;
        SgStatement* prev = SageInterface::getPreviousStatement(decl);
        if (prev) {
            SgPragmaDeclaration* prevPragma = isSgPragmaDeclaration(prev);
            if (prevPragma) {
                std::string text = prevPragma->unparseToString();
                if (text.find("omp declare target") != std::string::npos) {
                    alreadyDeclared = true;
                }
            }
        }

        if (alreadyDeclared) continue;

        SgPragmaDeclaration* declareTarget =
            SageBuilder::buildPragmaDeclaration("omp declare target",
                                                decl->get_scope());
        SageInterface::insertStatementBefore(decl, declareTarget);

        SgPragmaDeclaration* endDeclareTarget =
            SageBuilder::buildPragmaDeclaration("omp end declare target",
                                                decl->get_scope());
        SageInterface::insertStatementAfter(decl, endDeclareTarget);
    }
}

// Build a mapped variable reference. For arrays with a known constant size,
// emit an array section (e.g., "A[0:1024]"). For pointers or arrays whose
// size cannot be determined, fall back to the bare variable name.
static std::string buildMappedVarName(SgInitializedName* var) {
    if (!var) return "";

    std::string name = var->get_name().getString();
    SgType* type = var->get_type();
    if (!type) return name;

    type = type->stripType(SgType::STRIP_MODIFIER_TYPE | SgType::STRIP_TYPEDEF_TYPE);
    SgArrayType* arrType = isSgArrayType(type);
    if (!arrType) return name;

    SgExpression* index = arrType->get_index();
    if (!index) return name;

    std::string sizeStr = index->unparseToString();
    if (sizeStr.empty()) return name;

    return name + "[0:" + sizeStr + "]";
}

std::string OpenMPCodeGen::buildMapClause(
    const std::set<std::pair<SgInitializedName*, std::string>>& mapClauses) {
    // Group variables by direction.
    std::map<std::string, std::vector<std::string>> directionGroups;
    for (const auto& [var, direction] : mapClauses) {
        directionGroups[direction].push_back(buildMappedVarName(var));
    }

    std::ostringstream oss;
    bool firstClause = true;
    for (const auto& [direction, vars] : directionGroups) {
        if (!firstClause) oss << " ";
        firstClause = false;
        oss << "map(" << direction << ":";
        bool firstVar = true;
        for (const std::string& varName : vars) {
            if (!firstVar) oss << ", ";
            firstVar = false;
            oss << varName;
        }
        oss << ")";
    }
    return oss.str();
}

std::string OpenMPCodeGen::buildVarList(const std::set<SgInitializedName*>& vars) {
    std::ostringstream oss;
    bool first = true;
    for (SgInitializedName* var : vars) {
        if (!first) oss << ", ";
        first = false;
        oss << var->get_name().getString();
    }
    return oss.str();
}

std::string OpenMPCodeGen::buildReductionClause(
    const std::vector<loomX::ReductionInfo>& reductionDetails) {
    // Group variables by operator string.
    std::map<std::string, std::vector<std::string>> groups;
    for (const loomX::ReductionInfo& info : reductionDetails) {
        if (!info.variable) continue;
        std::string op = info.opString.empty() ? "+" : info.opString;
        groups[op].push_back(info.variable->get_name().getString());
    }

    std::ostringstream oss;
    bool firstClause = true;
    for (const auto& [op, vars] : groups) {
        if (!firstClause) oss << " ";
        firstClause = false;
        oss << "reduction(" << op << ":";
        bool firstVar = true;
        for (const std::string& varName : vars) {
            if (!firstVar) oss << ", ";
            firstVar = false;
            oss << varName;
        }
        oss << ")";
    }
    return oss.str();
}

// ---------------------------------------------------------------------------
// Target-data region hoisting (text-based)
// ---------------------------------------------------------------------------

// A GPU target-loop pair in the unparsed source is a pragma line followed by
// a for-loop. We locate consecutive pairs and wrap them with a single
// #pragma omp target data region, rewriting inner pragmas to
// "omp teams distribute parallel for" without map clauses.

static bool lineIsGPUTargetPragma(const std::string& line) {
    return line.find("#pragma omp target teams distribute parallel for") != std::string::npos;
}

static bool lineStartsForLoop(const std::string& line) {
    // Handle whitespace before "for".
    size_t i = 0;
    while (i < line.size() && std::isspace(static_cast<unsigned char>(line[i]))) ++i;
    return line.compare(i, 3, "for") == 0;
}

// Given the start of a pragma line, find the line that ends the pragma
// (handles backslash continuation, though ROSE usually emits single-line
// pragmas).
static size_t findPragmaEnd(const std::string& source, size_t pragmaStart) {
    size_t pos = pragmaStart;
    while (true) {
        size_t nl = source.find('\n', pos);
        if (nl == std::string::npos) return source.size();
        if (nl == 0 || source[nl - 1] != '\\') return nl;
        pos = nl + 1;
    }
}

// Find the end of the for-statement that starts at forStart (skip its body).
static size_t findForStatementEnd(const std::string& source, size_t forStart) {
    size_t pos = source.find('{', forStart);
    if (pos == std::string::npos) return source.size();

    int depth = 0;
    for (size_t i = pos; i < source.size(); ++i) {
        if (source[i] == '{') ++depth;
        else if (source[i] == '}') {
            --depth;
            if (depth == 0) return i + 1;
        }
    }
    return source.size();
}

// Skip whitespace, blank lines, and // comments. Return source.size() if none
// found.
static size_t skipBlankAndComments(const std::string& source, size_t pos) {
    while (pos < source.size()) {
        size_t next = source.find_first_not_of(" \t\n", pos);
        if (next == std::string::npos) return source.size();
        if (source.compare(next, 2, "//") == 0) {
            size_t eol = source.find('\n', next);
            pos = (eol == std::string::npos) ? source.size() : eol + 1;
            continue;
        }
        return next;
    }
    return source.size();
}

void OpenMPCodeGen::hoistTargetDataRegions(std::string& source) {
    const std::string targetPrefix = "#pragma omp target teams distribute parallel for";

    std::string result;
    size_t pos = 0;
    while (pos < source.size()) {
        size_t lineStart = source.find_first_not_of(" \t\n", pos);
        if (lineStart == std::string::npos) break;

        // Look for a GPU target pragma at lineStart.
        if (source.compare(lineStart, targetPrefix.size(), targetPrefix) != 0) {
            // Copy until next non-empty line start.
            size_t nextLine = source.find('\n', lineStart);
            if (nextLine == std::string::npos) nextLine = source.size();
            result.append(source, pos, nextLine - pos + 1);
            pos = nextLine + 1;
            continue;
        }

        // Found a GPU target pragma. Try to gather a run of consecutive
        // target-loop pairs, ignoring blank lines and // comments between them.
        std::vector<std::pair<size_t, size_t>> pairs;
        size_t scan = lineStart;
        while (scan < source.size()) {
            size_t pragmaStart = skipBlankAndComments(source, scan);
            if (pragmaStart == source.size()) break;
            if (source.compare(pragmaStart, targetPrefix.size(), targetPrefix) != 0) break;

            size_t pragmaEnd = findPragmaEnd(source, pragmaStart);
            size_t afterPragma = skipBlankAndComments(source, pragmaEnd + 1);
            if (afterPragma == source.size()) break;
            if (!lineStartsForLoop(source.substr(afterPragma, 10))) break;

            size_t forEnd = findForStatementEnd(source, afterPragma);
            pairs.push_back({pragmaStart, forEnd});
            scan = forEnd;
        }

        if (pairs.size() < 2) {
            // Single GPU loop: leave as-is.
            size_t end = pairs.empty() ? lineStart + targetPrefix.size()
                                       : pairs.back().second;
            result.append(source, pos, end - pos);
            pos = end;
            continue;
        }

        // Collect mapped variables from all pragmas in the run.
        std::set<std::string> mappedVars;
        for (const auto& p : pairs) {
            size_t pragmaEnd = findPragmaEnd(source, p.first);
            std::string pragmaText = source.substr(p.first, pragmaEnd - p.first);
            std::set<std::string> vars = collectMappedVars(pragmaText);
            mappedVars.insert(vars.begin(), vars.end());
        }

        // Emit target data pragma.
        result += "#pragma omp target data " + buildTargetDataMapClause(mappedVars) + "\n{\n";

        // Emit rewritten inner loops.
        for (const auto& p : pairs) {
            size_t pragmaEnd = findPragmaEnd(source, p.first);
            std::string pragmaText = source.substr(p.first, pragmaEnd - p.first);
            std::string bodyText = source.substr(pragmaEnd + 1, p.second - (pragmaEnd + 1));
            result += stripTargetAndMap(pragmaText) + "\n";
            result += bodyText;
            // Ensure a newline between loops.
            if (!bodyText.empty() && bodyText.back() != '\n') result += "\n";
        }
        result += "}\n";

        pos = pairs.back().second;
    }

    source = result;
}

std::set<std::string> OpenMPCodeGen::collectMappedVars(const std::string& pragmaText) {
    std::set<std::string> vars;
    size_t pos = 0;
    while ((pos = pragmaText.find("map(", pos)) != std::string::npos) {
        size_t open = pragmaText.find('(', pos);
        size_t close = pragmaText.find(')', open);
        if (open == std::string::npos || close == std::string::npos) break;

        std::string inside = pragmaText.substr(open + 1, close - open - 1);
        size_t colon = inside.find(':');
        if (colon != std::string::npos) {
            std::string varList = inside.substr(colon + 1);
            std::istringstream iss(varList);
            std::string var;
            while (std::getline(iss, var, ',')) {
                size_t b = var.find_first_not_of(" \t");
                size_t e = var.find_last_not_of(" \t");
                if (b != std::string::npos && e != std::string::npos) {
                    vars.insert(var.substr(b, e - b + 1));
                }
            }
        }
        pos = close + 1;
    }
    return vars;
}

std::string OpenMPCodeGen::buildTargetDataMapClause(
    const std::set<std::string>& mappedVars) {
    std::ostringstream oss;
    oss << "map(tofrom:";
    bool first = true;
    for (const std::string& var : mappedVars) {
        if (!first) oss << ", ";
        first = false;
        oss << var;
    }
    oss << ")";
    return oss.str();
}

std::string OpenMPCodeGen::stripTargetAndMap(const std::string& pragmaText) {
    std::string result = pragmaText;

    // Keep the 'target teams distribute parallel for' combined construct;
    // it is valid inside a target data region and inherits the mapped data.
    // Only remove the per-loop map clauses.
    size_t pos = 0;
    while ((pos = result.find(" map(", pos)) != std::string::npos) {
        size_t close = result.find(')', pos);
        if (close == std::string::npos) break;
        result.erase(pos, close - pos + 1);
    }

    return result;
}
