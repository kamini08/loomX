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

}  // namespace

std::string fuseAdjacentOffloadLoops(const std::string& source) {
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

        // Accept the producer loop if it is bare.  Accept the consumer if it
        // only carries reduction/schedule clauses we can preserve on the merged
        // pragma.  Map clauses should already have been hoisted away; reject
        // any remaining 'map(' clause to stay safe.
        bool bare = normTail.empty();
        bool reducibleConsumer = !bare &&
                                 normTail.find("map(") == std::string::npos &&
                                 normTail.find("collapse(") == std::string::npos;
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
        // holding the producer's statements followed by the consumer's.  Use
        // the consumer's pragma if it carries clauses (e.g. reduction); otherwise
        // the producer's pragma is fine.
        // Emitting the producer's whole for-statement here would strand the
        // consumer's statements after the closing brace, outside any loop.
        // Any declarations that sat between the loops are moved in front of the
        // merged construct so they remain in scope for the consumer body.
        out.append(source, copyFrom, a.pragmaStart - copyFrom);
        out += movableDecls;
        size_t pragmaKeepStart = a.pragmaStart;
        size_t pragmaKeepEnd = a.bodyStart;
        if (!b.clauses.empty()) {
            // Consumer has clauses (e.g. reduction). Use the consumer's pragma
            // and for-header (up to the body opening brace).
            pragmaKeepStart = b.pragmaStart;
            pragmaKeepEnd = b.bodyStart;
        }
        out.append(source, pragmaKeepStart, pragmaKeepEnd - pragmaKeepStart);
        out += "{\n";
        out += a.inner;
        if (!a.inner.empty() && a.inner.back() != '\n') out += "\n";
        out += b.inner;
        if (!b.inner.empty() && b.inner.back() != '\n') out += "\n";
        out += "  }";

        copyFrom = b.forEnd;
        consumed.insert(i + 1);
        ++merges;
    }

    if (!merges) return source;
    out.append(source, copyFrom, source.size() - copyFrom);

    std::cerr << "[LoopFusion] fused " << merges
              << " producer/consumer loop pair(s)\n";
    return out;
}

}  // namespace loomX
