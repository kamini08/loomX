#include "OpenMPCodeGen.h"
#include "LoopAnalysisUtil.h"
#include <map>
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

    // Flatten a perfectly nested 2D/3D fan for better GPU occupancy.  The
    // collapse depth is precomputed by GpuProfitability::collapseDepthFor and
    // only ever >= 2 when every collapsed inner loop is canonical and carries
    // no loop-carried dependence.
    if (summary.collapseDepth > 1) {
        pragmaText << " collapse(" << summary.collapseDepth << ")";
    }

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

// Build a mapped variable reference.
//
// - File-scope / static / local arrays: the bare variable name is enough;
//   the runtime sizes the transfer from the compile-time extent.
// - Function-parameter arrays: a parameter of type "double A[N][N]" is
//   really a pointer at the ABI level, so map(A) would transfer only the
//   pointer (8 bytes) and the device kernel would fault.  These need an
//   explicit array section.
//
// Multi-dimensional arrays must use a per-dimension section
// "A[0:N][0:M]": clang-15 interprets the length of "A[0:K]" as the number of
// ROWS, not the flat element count, so a flat section "A[0:(N)*(M)]"
// allocates N*M*M bytes (cuMemAlloc OOM), and flat sections on static arrays
// are mis-sized past their object (libomptarget abort "explicit extension not
// allowed").  Per-dimension sections have an exact size in both cases.
static std::string buildMappedVarName(SgInitializedName* var) {
    if (!var) return "";

    std::string name = var->get_name().getString();
    SgType* type = var->get_type();
    if (!type) return name;

    type = type->stripType(SgType::STRIP_MODIFIER_TYPE | SgType::STRIP_TYPEDEF_TYPE);
    SgArrayType* arrType = isSgArrayType(type);
    if (!arrType) return name;

    bool isParameter = isSgFunctionParameterList(var->get_parent()) != nullptr;
    if (!isParameter) return name;

    // Emit one [0:<dim>] per array dimension.
    std::string section;
    while (arrType) {
        SgExpression* index = arrType->get_index();
        if (!index) return name;
        std::string dim = index->unparseToString();
        if (dim.empty()) return name;
        section += "[0:" + dim + "]";
        SgType* baseType = arrType->get_base_type();
        if (!baseType) break;
        baseType = baseType->stripType(SgType::STRIP_MODIFIER_TYPE | SgType::STRIP_TYPEDEF_TYPE);
        arrType = isSgArrayType(baseType);
    }

    if (section.empty()) return name;
    return name + section;
}

std::string OpenMPCodeGen::buildMapClause(
    const std::set<std::pair<SgInitializedName*, std::string>>& mapClauses) {
    // Deduplicate by variable name, keeping the most conservative direction
    // (tofrom > from > to) in case the same name was collected from multiple
    // variable references (e.g. outer-loop scalars also referenced in inner
    // loops).
    std::map<std::string, std::string> directionByName;
    auto directionRank = [](const std::string& d) {
        if (d == "tofrom") return 2;
        if (d == "from") return 1;
        return 0;
    };
    for (const auto& [var, direction] : mapClauses) {
        std::string name = buildMappedVarName(var);
        auto it = directionByName.find(name);
        if (it == directionByName.end() || directionRank(direction) > directionRank(it->second)) {
            directionByName[name] = direction;
        }
    }

    // Group variables by direction.
    std::map<std::string, std::vector<std::string>> directionGroups;
    for (const auto& [name, direction] : directionByName) {
        directionGroups[direction].push_back(name);
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
        std::string varName = info.variable->get_name().getString();
        if (info.isArrayElement && info.arrayIndex) {
            varName += "[" + info.arrayIndex->unparseToString() + "]";
        }
        groups[op].push_back(varName);
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

// Find the index of the closing delimiter that matches the opening delimiter
// at openPos. Returns std::string::npos if no matching close is found. This
// correctly skips nested delimiters, which matters for OpenMP map clauses like
// map(tofrom:A[0:(N) * (N)]) and for braced loop bodies.
static size_t findMatchingClose(const std::string& source, size_t openPos,
                                char openChar, char closeChar) {
    if (openPos >= source.size() || source[openPos] != openChar) {
        return std::string::npos;
    }
    int depth = 0;
    for (size_t i = openPos; i < source.size(); ++i) {
        if (source[i] == openChar) {
            ++depth;
        } else if (source[i] == closeChar) {
            --depth;
            if (depth == 0) return i;
        }
    }
    return std::string::npos;
}

static size_t findMatchingCloseParen(const std::string& source, size_t openPos) {
    return findMatchingClose(source, openPos, '(', ')');
}

static size_t findMatchingCloseBrace(const std::string& source, size_t openPos) {
    return findMatchingClose(source, openPos, '{', '}');
}

// Find the end of the for-statement that starts at forStart. Handles both
// braced bodies (for (...) { ... }) and single-statement bodies
// (for (...) stmt;). Returns source.size() if the structure cannot be parsed.
static size_t findForStatementEnd(const std::string& source, size_t forStart) {
    // Find the '(' that begins the for-header.
    size_t headerOpen = source.find('(', forStart);
    if (headerOpen == std::string::npos) return source.size();

    // Find the matching ')' that ends the for-header.
    size_t headerClose = findMatchingCloseParen(source, headerOpen);
    if (headerClose == std::string::npos) return source.size();

    // Skip whitespace after the header to find the body.
    size_t bodyStart = source.find_first_not_of(" \t\n", headerClose + 1);
    if (bodyStart == std::string::npos) return source.size();

    // Braced body: find matching '}'.
    if (source[bodyStart] == '{') {
        size_t bodyClose = findMatchingCloseBrace(source, bodyStart);
        if (bodyClose == std::string::npos) return source.size();
        return bodyClose + 1;
    }

    // Single-statement body: find the terminating ';' that is not inside
    // nested braces or parentheses. We must skip semicolons inside nested
    // for/if/while headers (e.g. for(j=0; j<n; j++) ...).
    int braceDepth = 0;
    int parenDepth = 0;
    for (size_t i = bodyStart; i < source.size(); ++i) {
        char c = source[i];
        if (c == '{') {
            ++braceDepth;
        } else if (c == '}') {
            --braceDepth;
            if (braceDepth < 0) braceDepth = 0;
        } else if (c == '(') {
            ++parenDepth;
        } else if (c == ')') {
            --parenDepth;
            if (parenDepth < 0) parenDepth = 0;
        } else if (c == ';' && braceDepth == 0 && parenDepth == 0) {
            return i + 1;
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

        // Collect mapped variables and their directions from all pragmas in
        // the run. Directions are merged conservatively.
        std::map<std::string, std::string> mappedVars;
        for (const auto& p : pairs) {
            size_t pragmaEnd = findPragmaEnd(source, p.first);
            std::string pragmaText = source.substr(p.first, pragmaEnd - p.first);
            std::map<std::string, std::string> vars = collectMappedVars(pragmaText);
            for (const auto& kv : vars) {
                const std::string& name = kv.first;
                const std::string& direction = kv.second;
                auto it = mappedVars.find(name);
                if (it == mappedVars.end()) {
                    mappedVars[name] = direction;
                } else {
                    auto rank = [](const std::string& d) {
                        if (d == "tofrom") return 2;
                        if (d == "from") return 1;
                        return 0;
                    };
                    if (rank(direction) > rank(it->second)) {
                        it->second = direction;
                    }
                }
            }
        }

        // If no variables are mapped, hoisting a target data region is pointless
        // and produces an invalid empty map clause. Leave the loops as-is.
        if (mappedVars.empty()) {
            size_t end = pairs.back().second;
            result.append(source, pos, end - pos);
            pos = end;
            continue;
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

// Split a comma-separated list respecting nested parentheses and square
// brackets. Used to parse OpenMP variable lists inside map clauses.
static std::vector<std::string> splitTopLevelCommas(const std::string& text) {
    std::vector<std::string> parts;
    std::string current;
    int parenDepth = 0;
    int bracketDepth = 0;
    for (char c : text) {
        if (c == '(') {
            ++parenDepth;
            current += c;
        } else if (c == ')') {
            --parenDepth;
            current += c;
        } else if (c == '[') {
            ++bracketDepth;
            current += c;
        } else if (c == ']') {
            --bracketDepth;
            current += c;
        } else if (c == ',' && parenDepth == 0 && bracketDepth == 0) {
            parts.push_back(current);
            current.clear();
        } else {
            current += c;
        }
    }
    if (!current.empty() || !parts.empty()) {
        parts.push_back(current);
    }
    return parts;
}

std::map<std::string, std::string> OpenMPCodeGen::collectMappedVars(
    const std::string& pragmaText) {
    std::map<std::string, std::string> vars;
    size_t pos = 0;
    while ((pos = pragmaText.find("map(", pos)) != std::string::npos) {
        size_t open = pragmaText.find('(', pos);
        if (open == std::string::npos) break;
        size_t close = findMatchingCloseParen(pragmaText, open);
        if (close == std::string::npos) break;

        std::string inside = pragmaText.substr(open + 1, close - open - 1);
        size_t colon = inside.find(':');
        if (colon != std::string::npos) {
            std::string direction = inside.substr(0, colon);
            // Strip whitespace from the direction.
            size_t db = direction.find_first_not_of(" \t");
            size_t de = direction.find_last_not_of(" \t");
            if (db != std::string::npos && de != std::string::npos) {
                direction = direction.substr(db, de - db + 1);
            } else {
                direction = "tofrom";
            }
            std::string varList = inside.substr(colon + 1);
            for (std::string var : splitTopLevelCommas(varList)) {
                size_t b = var.find_first_not_of(" \t");
                size_t e = var.find_last_not_of(" \t");
                if (b != std::string::npos && e != std::string::npos) {
                    std::string name = var.substr(b, e - b + 1);
                    // Merge directions conservatively: tofrom > from > to.
                    auto it = vars.find(name);
                    if (it == vars.end()) {
                        vars[name] = direction;
                    } else {
                        auto rank = [](const std::string& d) {
                            if (d == "tofrom") return 2;
                            if (d == "from") return 1;
                            return 0;
                        };
                        if (rank(direction) > rank(it->second)) {
                            it->second = direction;
                        }
                    }
                }
            }
        }
        pos = close + 1;
    }
    return vars;
}

std::string OpenMPCodeGen::buildTargetDataMapClause(
    const std::map<std::string, std::string>& mappedVars) {
    if (mappedVars.empty()) return "";

    // Group variables by direction.
    std::map<std::string, std::vector<std::string>> directionGroups;
    for (const auto& [name, direction] : mappedVars) {
        directionGroups[direction].push_back(name);
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

std::string OpenMPCodeGen::stripTargetAndMap(const std::string& pragmaText) {
    std::string result = pragmaText;

    // Keep the 'target teams distribute parallel for' combined construct;
    // it is valid inside a target data region and inherits the mapped data.
    // Only remove the per-loop map clauses.
    size_t pos = 0;
    while ((pos = result.find(" map(", pos)) != std::string::npos) {
        size_t open = pos + std::string(" map(").size() - 1;
        size_t close = findMatchingCloseParen(result, open);
        if (close == std::string::npos) break;
        result.erase(pos, close - pos + 1);
    }

    return result;
}
