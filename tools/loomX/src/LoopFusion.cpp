#include "LoopFusion.h"

#include <cctype>
#include <cstdlib>
#include <iostream>
#include <regex>
#include <set>
#include <string>
#include <vector>

namespace loomX {
namespace {

const char* kGpuPragma =
    "#pragma omp target teams distribute parallel for";

bool isIdentChar(char c) {
    return std::isalnum(static_cast<unsigned char>(c)) || c == '_';
}

// Collapse runs of whitespace so two headers can be compared textually.
std::string normalize(const std::string& s) {
    std::string out;
    bool pending = false;
    for (char c : s) {
        if (std::isspace(static_cast<unsigned char>(c))) {
            pending = !out.empty();
            continue;
        }
        if (pending) { out += ' '; pending = false; }
        out += c;
    }
    return out;
}

size_t matchDelim(const std::string& s, size_t open, char oc, char cc) {
    if (open >= s.size() || s[open] != oc) return std::string::npos;
    int depth = 0;
    for (size_t i = open; i < s.size(); ++i) {
        if (s[i] == oc) ++depth;
        else if (s[i] == cc) { if (--depth == 0) return i; }
    }
    return std::string::npos;
}

bool onlyWhitespace(const std::string& s, size_t from, size_t to) {
    for (size_t i = from; i < to; ++i) {
        if (!std::isspace(static_cast<unsigned char>(s[i]))) return false;
    }
    return true;
}

// True if line looks like a C variable declaration, e.g. "double sum = 0.0;".
// Conservative: requires a type-name sequence followed by an identifier and ';'.
bool isDeclarationLine(const std::string& line) {
    static const std::regex declRe(
        R"(^\s*(?:const\s+|volatile\s+)*(?:\w+\s+)+(?:\*\s*)*\w+(?:\s*=\s*[^;]+)?\s*;\s*$)");
    return std::regex_match(line, declRe);
}

// True if line is a comment or empty.
bool isCommentOrBlank(const std::string& line) {
    static const std::regex blankRe(R"(^\s*$)");
    static const std::regex cppCommentRe(R"(^\s*//.*$)");
    if (std::regex_match(line, blankRe)) return true;
    if (std::regex_match(line, cppCommentRe)) return true;
    // C-style comments spanning a whole line are rare here; skip for simplicity.
    return false;
}

// If the text between [from, to) consists only of whitespace, comments, and
// variable declarations, return the declaration text (preserving newlines).
// Otherwise return the empty string.  The declarations may be moved in front of
// the fused loop without changing semantics, provided their initializers do not
// depend on the producer's writes.
std::string extractMovableDeclarations(const std::string& s, size_t from, size_t to) {
    if (from >= to || to > s.size()) return "";
    std::string gap = s.substr(from, to - from);
    std::string out;
    size_t lineStart = 0;
    while (lineStart < gap.size()) {
        size_t nl = gap.find('\n', lineStart);
        std::string line = (nl == std::string::npos)
                               ? gap.substr(lineStart)
                               : gap.substr(lineStart, nl - lineStart + 1);
        if (!isCommentOrBlank(line) && !isDeclarationLine(line)) {
            return "";
        }
        if (isDeclarationLine(line)) out += line;
        lineStart = (nl == std::string::npos) ? gap.size() : nl + 1;
    }
    return out;
}

bool containsWord(const std::string& s, const std::string& w) {
    size_t pos = 0;
    while ((pos = s.find(w, pos)) != std::string::npos) {
        bool leftOk = pos == 0 || !isIdentChar(s[pos - 1]);
        size_t after = pos + w.size();
        bool rightOk = after >= s.size() || !isIdentChar(s[after]);
        if (leftOk && rightOk) return true;
        pos = after;
    }
    return false;
}

// One GPU pragma plus the for-loop it governs.
struct LoopRec {
    size_t pragmaStart = 0, pragmaEnd = 0;  // pragma text, [start, end)
    size_t forStart = 0, forEnd = 0;        // for statement text, [start, end)
    size_t bodyStart = 0;                   // first byte of the body
    std::string header;                     // "for (...)", normalized
    std::string inner;                      // body text, braces stripped
    std::string iv;                         // induction variable
    std::string clauses;                    // text after the base construct name
    std::set<std::string> writes;           // array bases assigned by the body
};

// Parse the for-loop that starts at `from`. Returns false on malformed input.
bool parseLoop(const std::string& s, size_t from, size_t* forEnd,
               size_t* bodyStart, std::string* header, std::string* inner) {
    size_t open = s.find('(', from);
    if (open == std::string::npos) return false;
    size_t close = matchDelim(s, open, '(', ')');
    if (close == std::string::npos) return false;

    size_t bodyAt = s.find_first_not_of(" \t\n\r", close + 1);
    if (bodyAt == std::string::npos) return false;

    std::string body;
    size_t end;
    if (s[bodyAt] == '{') {
        size_t bodyClose = matchDelim(s, bodyAt, '{', '}');
        if (bodyClose == std::string::npos) return false;
        body = s.substr(bodyAt + 1, bodyClose - bodyAt - 1);
        end = bodyClose + 1;
    } else {
        // Single-statement body: run to the first ';' at depth 0.
        int depth = 0;
        end = s.size();
        for (size_t i = bodyAt; i < s.size(); ++i) {
            char c = s[i];
            if (c == '(' || c == '{') ++depth;
            else if (c == ')' || c == '}') --depth;
            else if (c == ';' && depth == 0) { end = i + 1; break; }
        }
        body = s.substr(bodyAt, end - bodyAt);
    }

    *forEnd = end;
    *bodyStart = bodyAt;
    *header = normalize(s.substr(from, close - from + 1));
    *inner = body;
    return true;
}

// The induction variable is the identifier just left of the first '=' in the
// header, e.g. "for (int i = 0; ...)" -> "i".
std::string headerIV(const std::string& header) {
    size_t eq = header.find('=');
    if (eq == std::string::npos) return "";
    size_t i = eq;
    while (i > 0 && std::isspace(static_cast<unsigned char>(header[i - 1]))) --i;
    size_t end = i;
    while (i > 0 && isIdentChar(header[i - 1])) --i;
    if (end == i) return "";
    return header.substr(i, end - i);
}

// Array bases that this body assigns to, e.g. "a[i] = ..." -> {"a"}.
std::set<std::string> assignedBases(const std::string& inner) {
    std::set<std::string> out;
    for (size_t i = 0; i < inner.size(); ++i) {
        if (inner[i] != '=') continue;
        if (i + 1 < inner.size() && inner[i + 1] == '=') { ++i; continue; }
        if (i > 0) {
            char p = inner[i - 1];
            if (p == '=' || p == '!' || p == '<' || p == '>' || p == '+' ||
                p == '-' || p == '*' || p == '/' || p == '%' || p == '&' ||
                p == '|' || p == '^') {
                continue;
            }
        }
        // Walk left over the subscript, then read the base identifier.
        size_t j = i;
        while (j > 0 && std::isspace(static_cast<unsigned char>(inner[j - 1]))) --j;
        if (j == 0 || inner[j - 1] != ']') continue;
        size_t close = j - 1;
        int depth = 0;
        size_t k = close;
        for (;;) {
            if (inner[k] == ']') ++depth;
            else if (inner[k] == '[') { if (--depth == 0) break; }
            if (k == 0) break;
            --k;
        }
        size_t baseEnd = k;
        while (baseEnd > 0 && std::isspace(static_cast<unsigned char>(inner[baseEnd - 1]))) {
            --baseEnd;
        }
        size_t baseStart = baseEnd;
        while (baseStart > 0 && isIdentChar(inner[baseStart - 1])) --baseStart;
        if (baseStart == baseEnd) continue;
        out.insert(inner.substr(baseStart, baseEnd - baseStart));
    }
    return out;
}

// True when the identifier ending at `after` is used as a subscript, i.e. the
// occurrence is an element access rather than a declaration or a bare pointer.
bool isSubscriptedUse(const std::string& s, size_t after, size_t limit) {
    size_t p = after;
    while (p < limit && std::isspace(static_cast<unsigned char>(s[p]))) ++p;
    return p < limit && s[p] == '[';
}

// Every occurrence of `name` in [from, to) that is a whole identifier.
std::vector<size_t> identOccurrences(const std::string& s, size_t from, size_t to,
                                     const std::string& name) {
    std::vector<size_t> out;
    size_t pos = from;
    while (pos < to && pos < s.size()) {
        size_t hit = s.find(name, pos);
        if (hit == std::string::npos || hit >= to) break;
        bool leftOk = hit == 0 || !isIdentChar(s[hit - 1]);
        size_t after = hit + name.size();
        bool rightOk = after >= s.size() || !isIdentChar(s[after]);
        if (leftOk && rightOk) out.push_back(hit);
        pos = hit + 1;
    }
    return out;
}

// The consumer may touch a produced array only as `name[iv]`: same index, so
// the fill and the use stay in the same iteration and cannot race.
bool consumerUsesOnlySameIndex(const std::string& s, const LoopRec& prod,
                               const LoopRec& cons) {
    for (const std::string& arr : prod.writes) {
        for (size_t hit : identOccurrences(s, cons.forStart, cons.forEnd, arr)) {
            size_t p = hit + arr.size();
            while (p < s.size() && std::isspace(static_cast<unsigned char>(s[p]))) ++p;
            if (p >= cons.forEnd || s[p] != '[') {
                // A bare name, including one passed to a callee, would leave
                // the index unconstrained. Refuse.
                return false;
            }
            size_t close = matchDelim(s, p, '[', ']');
            if (close == std::string::npos) return false;
            if (normalize(s.substr(p + 1, close - p - 1)) != prod.iv) return false;
        }
    }
    return true;
}

// A produced array is written at the loop index, so it may be consumed at that
// same index and nowhere else. This is the property that actually makes the
// merge safe.
//
// It is *not* required that no later loop read the array: successive
// "teams distribute parallel for" constructs each end in an implicit barrier, so
// a loop in a later target region already observes completed writes. Rejecting
// on a later reader would block exactly the kernels fusion exists to speed up.
bool producerWritesAtIndexOnly(const std::string& s, const LoopRec& prod) {
    for (const std::string& arr : prod.writes) {
        for (size_t hit : identOccurrences(s, prod.forStart, prod.forEnd, arr)) {
            size_t p = hit + arr.size();
            while (p < prod.forEnd && std::isspace(static_cast<unsigned char>(s[p]))) ++p;
            if (p >= prod.forEnd || s[p] != '[') continue;  // a bare pointer use
            size_t close = matchDelim(s, p, '[', ']');
            if (close == std::string::npos) return false;
            if (normalize(s.substr(p + 1, close - p - 1)) != prod.iv) return false;
        }
    }
    return true;
}

// True if expr contains an assignment, increment/decrement, or other
// construct that is unsafe to duplicate.  Function calls are allowed: the
// producer expression is already evaluated once per iteration, and duplicating
// a pure helper call into the consumer is exactly what enables call_chain and
// friends to collapse to the hand-written ideal.
bool hasSideEffects(const std::string& expr) {
    for (size_t i = 0; i < expr.size(); ++i) {
        if (i + 1 < expr.size() && expr[i] == '+' && expr[i + 1] == '+') return true;
        if (i + 1 < expr.size() && expr[i] == '-' && expr[i + 1] == '-') return true;
        if (expr[i] == '=') {
            // == is comparison; everything else containing '=' is an assignment
            // or compound assignment and is unsafe to duplicate.
            bool comparison = (i > 0 && expr[i - 1] == '=') ||
                              (i + 1 < expr.size() && expr[i + 1] == '=');
            if (!comparison) return true;
        }
    }
    return false;
}

// Find the last simple assignment "arr[iv] = expr;" in producerInner and
// return its RHS expression.  Returns empty if not found.  Also appends the
// positions of all such assignments to assignmentsOut so they can be removed.
std::string findLastIndexAssignmentRhs(const std::string& producerInner,
                                       const std::string& arr,
                                       const std::string& iv,
                                       std::vector<size_t>& assignmentsOut) {
    std::string needle = arr + "[" + iv + "]";
    std::string result;
    size_t pos = 0;
    while ((pos = producerInner.find(needle, pos)) != std::string::npos) {
        size_t after = pos + needle.size();
        while (after < producerInner.size() &&
               std::isspace(static_cast<unsigned char>(producerInner[after]))) ++after;
        if (after >= producerInner.size() || producerInner[after] != '=') { ++pos; continue; }
        if (after + 1 < producerInner.size() && producerInner[after + 1] == '=') { ++pos; continue; }
        size_t stmtEnd = producerInner.find(';', after);
        if (stmtEnd == std::string::npos) { ++pos; continue; }
        std::string expr = producerInner.substr(after + 1, stmtEnd - (after + 1));
        size_t e = 0;
        while (e < expr.size() && std::isspace(static_cast<unsigned char>(expr[e]))) ++e;
        size_t r = expr.size();
        while (r > e && std::isspace(static_cast<unsigned char>(expr[r - 1]))) --r;
        expr = expr.substr(e, r - e);
        assignmentsOut.push_back(pos);
        result = expr;
        pos = stmtEnd + 1;
    }
    return result;
}

// True if consumerInner writes arr[iv] before reading it.  Conservative: any
// write at the same index counts.
bool consumerOverwritesArray(const std::string& consumerInner,
                             const std::string& arr,
                             const std::string& iv) {
    std::string needle = arr + "[" + iv + "]";
    size_t pos = 0;
    while ((pos = consumerInner.find(needle, pos)) != std::string::npos) {
        size_t after = pos + needle.size();
        while (after < consumerInner.size() &&
               std::isspace(static_cast<unsigned char>(consumerInner[after]))) ++after;
        if (after < consumerInner.size() && consumerInner[after] == '=' &&
            (after + 1 >= consumerInner.size() || consumerInner[after + 1] != '=')) {
            return true;
        }
        ++pos;
    }
    return false;
}

// Replace every occurrence of arr[iv] in text with "(expr)".
std::string substituteArrayRead(const std::string& text,
                                const std::string& arr,
                                const std::string& iv,
                                const std::string& expr) {
    std::string needle = arr + "[" + iv + "]";
    std::string replacement = "(" + expr + ")";
    std::string out = text;
    size_t pos = 0;
    while ((pos = out.find(needle, pos)) != std::string::npos) {
        out.replace(pos, needle.size(), replacement);
        pos += replacement.size();
    }
    return out;
}

// Try to scalar-replace produced arrays that are only read at the same index.
// For each arr[iv] = expr in the producer, if expr is side-effect-free and the
// consumer only reads arr[iv] (does not overwrite it), replace the reads in the
// consumer with (expr) and drop the producer assignment.
// Returns true if any contraction happened.
bool contractProducerArrays(const std::string& producerInner,
                            const std::string& consumerInner,
                            const std::string& iv,
                            const std::set<std::string>& producedArrays,
                            std::string& newProducerInner,
                            std::string& newConsumerInner) {
    newProducerInner = producerInner;
    newConsumerInner = consumerInner;
    bool changed = false;

    for (const std::string& arr : producedArrays) {
        std::vector<size_t> assigns;
        std::string expr = findLastIndexAssignmentRhs(newProducerInner, arr, iv, assigns);
        if (expr.empty() || hasSideEffects(expr)) continue;
        if (consumerOverwritesArray(newConsumerInner, arr, iv)) continue;

        // If the assignments to this array are guarded by differing control
        // flow (e.g. one in an if branch and one in an else branch), the final
        // value is not a single expression and contraction is unsafe.
        if (assigns.size() > 1) {
            size_t first = assigns.front();
            size_t last = assigns.back();
            if (last > first) {
                std::string between = newProducerInner.substr(first, last - first);
                if (between.find("if") != std::string::npos ||
                    between.find("else") != std::string::npos) {
                    continue;
                }
            }
        }

        // Substitute in consumer.
        newConsumerInner = substituteArrayRead(newConsumerInner, arr, iv, expr);

        // Remove all producer assignment statements to this array.
        std::string needle = arr + "[" + iv + "]";
        for (auto it = assigns.rbegin(); it != assigns.rend(); ++it) {
            size_t pos = *it;
            size_t stmtStart = pos;
            while (stmtStart > 0 && newProducerInner[stmtStart - 1] != ';' &&
                   newProducerInner[stmtStart - 1] != '\n' &&
                   newProducerInner[stmtStart - 1] != '{') --stmtStart;
            size_t stmtEnd = newProducerInner.find(';', pos);
            if (stmtEnd != std::string::npos) {
                size_t removeEnd = stmtEnd + 1;
                while (removeEnd < newProducerInner.size() &&
                       (newProducerInner[removeEnd] == ' ' ||
                        newProducerInner[removeEnd] == '\t' ||
                        newProducerInner[removeEnd] == '\n')) ++removeEnd;
                newProducerInner.erase(stmtStart, removeEnd - stmtStart);
                changed = true;
            }
        }
    }
    return changed;
}

}  // namespace

// Parse a single map(...) clause and drop items whose base variable is never
// referenced inside the associated target data region.  Returns the rewritten
// clause, or an empty string if the clause has no surviving items.
static std::string pruneMapClause(const std::string& clause,
                                  const std::string& regionBody) {
    // Expected shape: map(modifier : item, item, ...) or map(item, item, ...)
    size_t open = clause.find('(');
    if (open == std::string::npos) return clause;
    size_t close = clause.rfind(')');
    if (close == std::string::npos || close <= open) return clause;

    std::string head = clause.substr(0, open + 1);  // "map("
    std::string inner = clause.substr(open + 1, close - open - 1);

    // Determine where the item list starts (after optional modifier ':').
    size_t colon = inner.find(':');
    std::string prefix;
    std::string itemList;
    if (colon == std::string::npos) {
        itemList = inner;
    } else {
        prefix = inner.substr(0, colon + 1);
        itemList = inner.substr(colon + 1);
    }

    // Split items by top-level commas.
    std::vector<std::string> items;
    std::string cur;
    int depth = 0;
    for (char c : itemList) {
        if (c == '(' || c == '[') ++depth;
        else if (c == ')' || c == ']') --depth;
        if (c == ',' && depth == 0) {
            items.push_back(cur);
            cur.clear();
        } else {
            cur += c;
        }
    }
    if (!cur.empty()) items.push_back(cur);

    // Keep items whose base identifier appears in the region body.
    std::vector<std::string> kept;
    for (const std::string& item : items) {
        std::string trimmed = item;
        size_t s = 0;
        while (s < trimmed.size() && std::isspace(static_cast<unsigned char>(trimmed[s]))) ++s;
        size_t e = trimmed.size();
        while (e > s && std::isspace(static_cast<unsigned char>(trimmed[e - 1]))) --e;
        trimmed = trimmed.substr(s, e - s);
        if (trimmed.empty()) continue;

        // Base identifier: up to first '[' or non-identifier char.
        size_t baseEnd = 0;
        while (baseEnd < trimmed.size() && isIdentChar(trimmed[baseEnd])) ++baseEnd;
        std::string base = trimmed.substr(0, baseEnd);
        if (base.empty()) {
            kept.push_back(item);
            continue;
        }

        // Check for a real use in the body.  A bare occurrence of the name is
        // enough; generated code always uses arrays as a[...].
        bool used = false;
        for (size_t p = 0; (p = regionBody.find(base, p)) != std::string::npos; ++p) {
            // Ensure this is a whole identifier, not a substring.
            bool before = (p == 0) || !isIdentChar(regionBody[p - 1]);
            size_t afterPos = p + base.size();
            bool after = (afterPos >= regionBody.size()) || !isIdentChar(regionBody[afterPos]);
            if (before && after) {
                used = true;
                break;
            }
        }
        if (used) kept.push_back(item);
    }

    if (kept.empty()) return "";
    std::string result = head + prefix;
    for (size_t i = 0; i < kept.size(); ++i) {
        if (i) result += ", ";
        result += kept[i];
    }
    result += ")";
    return result;
}

// Parse the tail of a GPU pragma into map(...) items and other clauses.
// Returns a map from direction (to/from/tofrom) to set of item strings, and
// appends any non-map clauses (e.g. reduction(+:x)) to nonMapOut.
static void parseMapClauses(const std::string& clauses,
                            std::map<std::string, std::set<std::string>>& mapsOut,
                            std::vector<std::string>& nonMapOut) {
    size_t p = 0;
    while (p < clauses.size()) {
        while (p < clauses.size() && std::isspace(static_cast<unsigned char>(clauses[p]))) ++p;
        if (p >= clauses.size()) break;
        if (clauses.compare(p, 4, "map(") == 0) {
            size_t clauseStart = p;
            size_t clauseEnd = matchDelim(clauses, p + 3, '(', ')');
            if (clauseEnd == std::string::npos) { ++p; continue; }
            std::string clause = clauses.substr(clauseStart, clauseEnd - clauseStart + 1);
            size_t open = clause.find('(');
            size_t close = clause.rfind(')');
            if (open != std::string::npos && close != std::string::npos && close > open) {
                std::string inner = clause.substr(open + 1, close - open - 1);
                size_t colon = inner.find(':');
                std::string direction = "tofrom";
                std::string itemList = inner;
                if (colon != std::string::npos) {
                    direction = normalize(inner.substr(0, colon));
                    itemList = inner.substr(colon + 1);
                }
                // Split items.
                std::string cur;
                int depth = 0;
                for (char c : itemList) {
                    if (c == '(' || c == '[') ++depth;
                    else if (c == ')' || c == ']') --depth;
                    if (c == ',' && depth == 0) {
                        std::string item = normalize(cur);
                        if (!item.empty()) mapsOut[direction].insert(item);
                        cur.clear();
                    } else {
                        cur += c;
                    }
                }
                std::string item = normalize(cur);
                if (!item.empty()) mapsOut[direction].insert(item);
            }
            p = clauseEnd + 1;
        } else {
            // Non-map clause: grab until next map( or end.
            size_t nextMap = clauses.find("map(", p + 1);
            std::string nonMap = clauses.substr(p, nextMap - p);
            std::string trimmed = normalize(nonMap);
            if (!trimmed.empty()) nonMapOut.push_back(trimmed);
            p = (nextMap == std::string::npos) ? clauses.size() : nextMap;
        }
    }
}

static std::string mapBaseName(const std::string& item) {
    size_t i = 0;
    while (i < item.size() && isIdentChar(item[i])) ++i;
    return item.substr(0, i);
}

static int mapDirRank(const std::string& d) {
    if (d == "tofrom") return 2;
    if (d == "from") return 1;
    if (d == "to") return 0;
    return -1;
}

// Merge two pragma clause tails.  Map items are unioned by base variable name,
// preferring stronger directions.  Non-map clauses are concatenated.
static std::string mergeClauses(const std::string& a, const std::string& b) {
    std::map<std::string, std::set<std::string>> mapsA, mapsB;
    std::vector<std::string> nonA, nonB;
    parseMapClauses(a, mapsA, nonA);
    parseMapClauses(b, mapsB, nonB);

    // Merge directions, then per base name keep highest rank.
    std::map<std::string, std::map<std::string, std::string>> byBase; // base -> dir -> item
    auto absorb = [&](const std::map<std::string, std::set<std::string>>& src) {
        for (const auto& kv : src) {
            const std::string& dir = kv.first;
            for (const std::string& item : kv.second) {
                std::string base = mapBaseName(item);
                if (base.empty()) continue;
                auto& dirMap = byBase[base];
                auto it = dirMap.find(dir);
                if (it == dirMap.end() || mapDirRank(dir) > mapDirRank(it->first)) {
                    dirMap[dir] = item;
                }
            }
        }
    };
    absorb(mapsA);
    absorb(mapsB);

    std::string result;
    // Emit merged map clauses grouped by direction.
    for (const auto& dir : {"to", "from", "tofrom"}) {
        std::vector<std::string> items;
        for (const auto& kv : byBase) {
            auto it = kv.second.find(dir);
            if (it != kv.second.end()) items.push_back(it->second);
        }
        if (!items.empty()) {
            if (!result.empty()) result += " ";
            result += std::string("map(") + dir + ":";
            for (size_t i = 0; i < items.size(); ++i) {
                if (i) result += ", ";
                result += items[i];
            }
            result += ")";
        }
    }
    for (const std::string& c : nonA) {
        if (!result.empty()) result += " ";
        result += c;
    }
    for (const std::string& c : nonB) {
        if (!result.empty()) result += " ";
        result += c;
    }
    return result;
}

// Remove dead map(...) items from #pragma omp target data regions.  An item is
// dead when its base variable is never referenced in the region body.
static std::string removeDeadTargetDataMaps(const std::string& source) {
    const std::string kDataPragma = "#pragma omp target data";
    std::string out;
    size_t copyFrom = 0;
    size_t cursor = 0;
    bool changed = false;

    while ((cursor = source.find(kDataPragma, cursor)) != std::string::npos) {
        size_t pragmaStart = cursor;
        size_t lineEnd = source.find('\n', pragmaStart);
        if (lineEnd == std::string::npos) lineEnd = source.size();
        size_t bodyStart = source.find('{', lineEnd);
        if (bodyStart == std::string::npos) break;
        size_t bodyEnd = matchDelim(source, bodyStart, '{', '}');
        if (bodyEnd == std::string::npos) break;

        std::string regionBody = source.substr(bodyStart, bodyEnd - bodyStart + 1);
        std::string pragmaLine = source.substr(pragmaStart, lineEnd - pragmaStart);

        // Rewrite each map(...) clause on this pragma line.
        std::string newPragma = pragmaLine;
        size_t p = 0;
        while ((p = newPragma.find("map(", p)) != std::string::npos) {
            size_t clauseStart = p;
            size_t clauseEnd = matchDelim(newPragma, clauseStart + 3, '(', ')');
            if (clauseEnd == std::string::npos) { ++p; continue; }
            std::string clause = newPragma.substr(clauseStart, clauseEnd - clauseStart + 1);
            std::string rewritten = pruneMapClause(clause, regionBody);
            if (rewritten.empty()) {
                // Remove the whole clause, including preceding whitespace.
                size_t eraseStart = clauseStart;
                while (eraseStart > 0 && std::isspace(static_cast<unsigned char>(newPragma[eraseStart - 1]))) --eraseStart;
                newPragma.erase(eraseStart, clauseEnd - eraseStart + 1);
                p = eraseStart;
            } else if (rewritten != clause) {
                newPragma.replace(clauseStart, clause.size(), rewritten);
                p = clauseStart + rewritten.size();
            } else {
                p = clauseEnd + 1;
            }
            changed = true;
        }

        if (newPragma != pragmaLine) {
            out.append(source, copyFrom, pragmaStart - copyFrom);
            out += newPragma;
            copyFrom = lineEnd;
        }
        cursor = bodyEnd + 1;
    }

    if (!changed) return source;
    out.append(source, copyFrom, source.size() - copyFrom);
    return out;
}

static std::string fuseOnePass(const std::string& source) {
    // Collect every bare GPU pragma + for-loop pair.
    const bool debug = std::getenv("LOOMX_FUSION_DEBUG") != nullptr;
    std::vector<LoopRec> recs;
    size_t cursor = 0;
    while ((cursor = source.find(kGpuPragma, cursor)) != std::string::npos) {
        size_t pragmaEnd = source.find('\n', cursor);
        if (pragmaEnd == std::string::npos) pragmaEnd = source.size();

        std::string tail = source.substr(
            cursor + std::char_traits<char>::length(kGpuPragma),
            pragmaEnd - cursor - std::char_traits<char>::length(kGpuPragma));
        std::string normTail = normalize(tail);

        // Accept the producer loop if it is bare or carries map/reduction
        // clauses.  Accept the consumer if its clauses can be merged with the
        // producer's (map, reduction, etc.).  Reject collapse clauses because
        // merging collapsed nests is not supported.
        bool bare = normTail.empty();
        bool reducibleConsumer = !bare && normTail.find("collapse(") == std::string::npos;
        if (bare || reducibleConsumer) {
            size_t forStart = source.find_first_not_of(" \t\n\r", pragmaEnd + 1);
            LoopRec rec;
            if (forStart != std::string::npos && forStart < source.size() &&
                source.compare(forStart, 3, "for") == 0 &&
                parseLoop(source, forStart, &rec.forEnd, &rec.bodyStart,
                          &rec.header, &rec.inner)) {
                rec.pragmaStart = cursor;
                rec.pragmaEnd = pragmaEnd + 1;
                rec.forStart = forStart;
                rec.iv = headerIV(rec.header);
                rec.clauses = normTail;
                rec.writes = assignedBases(rec.inner);
                recs.push_back(rec);
            }
        }
        cursor = pragmaEnd + 1;
    }
    if (recs.size() < 2) return source;

    // Build the output incrementally so accepted merges do not disturb the
    // offsets of the records still to be examined.
    std::string out;
    size_t copyFrom = 0;
    size_t merges = 0;
    std::set<size_t> consumed;  // indices of records folded into a producer

    for (size_t i = 0; i + 1 < recs.size(); ++i) {
        const LoopRec& a = recs[i];
        const LoopRec& b = recs[i + 1];
        if (consumed.count(i) || consumed.count(i + 1)) continue;

        std::string movableDecls = extractMovableDeclarations(source, a.forEnd, b.pragmaStart);
        bool adjacent = onlyWhitespace(source, a.forEnd, b.pragmaStart) || !movableDecls.empty();
        bool sameSpace = a.header == b.header && !a.iv.empty() && a.iv == b.iv;
        bool aPure = !a.writes.empty() &&
                     !containsWord(a.inner, "goto") && !containsWord(a.inner, "return");
        bool sameIndex = aPure && consumerUsesOnlySameIndex(source, a, b);
        bool prodIndex = aPure && producerWritesAtIndexOnly(source, a);

        if (debug) {
            std::cerr << "[LoopFusion] pair " << i << "/" << i + 1
                      << " adjacent=" << adjacent << " sameHeader=" << sameSpace
                      << " iv=" << a.iv << "/" << b.iv
                      << " pureWrite=" << aPure
                      << " sameIndex=" << sameIndex
                      << " prodIndex=" << prodIndex
                      << " aClauses=[" << a.clauses << "]"
                      << " bClauses=[" << b.clauses << "]";
            if (!aPure) std::cerr << " writes=" << a.writes.size();
            std::cerr << "\n";
        }
        if (!adjacent || !sameSpace || !aPure || !sameIndex || !prodIndex) continue;

        // Merge: keep one pragma and loop header, then emit a single body
        // holding the producer's statements followed by the consumer's.
        // Any declarations that sat between the loops are moved in front of the
        // merged construct so they remain in scope for the consumer body.
        out.append(source, copyFrom, a.pragmaStart - copyFrom);
        out += movableDecls;

        std::string mergedClauses = mergeClauses(a.clauses, b.clauses);
        out += "#pragma omp target teams distribute parallel for";
        if (!mergedClauses.empty()) out += " " + mergedClauses;
        out += "\n";
        out.append(source, a.forStart, a.bodyStart - a.forStart);
        out += "{\n";

        // Try to scalar-replace arrays produced by a and only read by b at the
        // same index.  This turns reduction_call into the hand-written ideal by
        // dropping the temporary a[] array entirely.
        std::string prodBody, consBody;
        contractProducerArrays(a.inner, b.inner, a.iv, a.writes, prodBody, consBody);

        out += prodBody;
        if (!prodBody.empty() && prodBody.back() != '\n') out += "\n";
        out += consBody;
        if (!consBody.empty() && consBody.back() != '\n') out += "\n";
        out += "  }";

        copyFrom = b.forEnd;
        consumed.insert(i + 1);
        ++merges;
    }

    if (!merges) {
        out = source;
    } else {
        out.append(source, copyFrom, source.size() - copyFrom);
        std::cerr << "[LoopFusion] fused " << merges
                  << " producer/consumer loop pair(s)\n";
    }

    // After fusion and array contraction some mapped arrays may no longer be
    // touched inside the target data region.  Dropping them avoids useless
    // host/device transfers.
    out = removeDeadTargetDataMaps(out);
    return out;
}

std::string fuseAdjacentOffloadLoops(const std::string& source) {
    // Repeatedly fuse adjacent pairs until a fixed point.  The first pass may
    // create new adjacent producer/consumer pairs (e.g. init+compute fused,
    // then that result fused with checksum), so iterating is required for
    // chain fusion.
    std::string current = source;
    int iterations = 0;
    while (true) {
        std::string next = fuseOnePass(current);
        if (next == current) break;
        current = next;
        if (++iterations > 10) break;  // safety limit
    }
    return current;
}

}  // namespace loomX
