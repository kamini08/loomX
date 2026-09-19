#include "CudaCodeGen.h"
#include "LoopAnalysisUtil.h"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>

using namespace loomX;

namespace {

// Return a string trimmed of leading/trailing whitespace.
std::string trim(const std::string& s) {
    size_t a = s.find_first_not_of(" \t\n\r");
    if (a == std::string::npos) return "";
    size_t b = s.find_last_not_of(" \t\n\r");
    return s.substr(a, b - a + 1);
}

// True if c is a C identifier character.
bool isIdentChar(char c) {
    return std::isalnum(static_cast<unsigned char>(c)) || c == '_';
}

// Replace whole-word occurrences of oldName in s with newName.
std::string replaceWholeWord(const std::string& s, const std::string& oldName,
                             const std::string& newName) {
    if (oldName.empty()) return s;
    std::string result;
    size_t pos = 0;
    while (pos < s.size()) {
        size_t found = s.find(oldName, pos);
        if (found == std::string::npos) {
            result += s.substr(pos);
            break;
        }
        bool leftBoundary = (found == 0) || !isIdentChar(s[found - 1]);
        bool rightBoundary = (found + oldName.size() >= s.size()) ||
                           !isIdentChar(s[found + oldName.size()]);
        if (leftBoundary && rightBoundary) {
            result += s.substr(pos, found - pos) + newName;
            pos = found + oldName.size();
        } else {
            result += s.substr(pos, found - pos + 1);
            pos = found + 1;
        }
    }
    return result;
}

// Quote a size expression for use in sizeof.
std::string sizeOfExpr(const std::string& type) {
    return "sizeof(" + type + ")";
}

} // namespace

void CudaCodeGen::generatePragmas(const loomX::LoopSummary& summary) {
    if (summary.target != ParallelTarget::GPU_OFFLOAD) return;
    if (!canTransform(summary)) return;
    cudaLoops_.push_back(buildLoopInfo(summary));
}

bool CudaCodeGen::canTransform(const loomX::LoopSummary& summary) const {
    // For now we parallelize the outermost loop and keep any inner loops
    // sequential inside the CUDA kernel.  True collapsed flattening is future
    // work; outer-loop parallelization is always correct.
    if (summary.collapseDepth < 1) {
        return false;
    }
    if (!summary.reductions.empty()) {
        std::cerr << "[CudaCodeGen] reductions not yet supported (line "
                  << summary.loop->get_file_info()->get_line() << ")\n";
        return false;
    }
    return true;
}

CudaCodeGen::CudaLoopInfo CudaCodeGen::buildLoopInfo(
    const loomX::LoopSummary& summary) {
    CudaLoopInfo info;
    info.loop = summary.loop;
    info.collapseDepth = summary.collapseDepth;
    info.hasReduction = !summary.reductions.empty();

    Sg_File_Info* startFi = summary.loop->get_file_info();
    Sg_File_Info* endFi = summary.loop->get_endOfConstruct();
    if (startFi) {
        info.fileName = startFi->get_filenameString();
        info.startLine = static_cast<int>(startFi->get_line());
        info.startCol = static_cast<int>(startFi->get_col());
    }
    if (endFi) {
        info.endLine = static_cast<int>(endFi->get_line());
        info.endCol = static_cast<int>(endFi->get_col());
    }

    // Collect only the outermost loop bound.  Inner loops remain sequential
    // inside the CUDA kernel, so their bounds are taken from the original body.
    SgForStatement* current = summary.loop;
    SgInitializedName* idx = SageInterface::getLoopIndexVariable(current);
    SgExpression* ub = summary.canonical.upperBound;
    if (!ub) {
        // Try to extract from the loop test directly.
        SgStatement* test = current->get_test();
        if (test) {
            SgExprStatement* est = isSgExprStatement(test);
            if (est) {
                SgExpression* cond = est->get_expression();
                SgBinaryOp* bop = isSgBinaryOp(cond);
                if (bop) {
                    SgExpression* rhs = bop->get_rhs_operand();
                    if (rhs) ub = rhs;
                }
            }
        }
    }
    std::string idxName = idx ? idx->get_name().getString() : "";
    std::string ubExpr = ub ? ub->unparseToString() : "";
    info.loopBounds.push_back({idxName, ubExpr});

    info.vars = collectVariables(summary);
    for (SgInitializedName* pv : summary.privateVars) {
        info.privateVars.insert(pv->get_name().getString());
    }

    // Body of the outermost loop (including inner sequential loops).
    SgStatement* body = summary.loop->get_loop_body();
    if (body) info.bodyText = trim(body->unparseToString());

    return info;
}

std::vector<CudaCodeGen::CudaVariable> CudaCodeGen::collectVariables(
    const loomX::LoopSummary& summary) {
    std::vector<CudaVariable> result;
    for (const auto& entry : summary.mapClauses) {
        CudaVariable cv;
        cv.var = entry.first;
        cv.name = cv.var->get_name().getString();
        cv.direction = entry.second;
        cv.isArray = isArrayVariable(cv.var);
        cv.elemType = getElementType(cv.var);
        cv.dims = getArrayDimensions(cv.var);

        if (cv.isArray) {
            // Host transfer size.
            if (cv.dims.empty()) {
                // Pointer array: size from outer loop bound * sizeof(elem).
                cv.totalSizeExpr =
                    "(" + summary.canonical.upperBound->unparseToString() + 
                    ") * " + sizeOfExpr(cv.elemType);
            } else {
                // Static array: use sizeof the original variable.
                cv.totalSizeExpr = "sizeof(" + cv.name + ")";
            }
        } else {
            cv.totalSizeExpr = sizeOfExpr(cv.elemType);
        }

        result.push_back(cv);
    }
    return result;
}

bool CudaCodeGen::isArrayVariable(SgInitializedName* var) {
    if (!var) return false;
    SgType* t = var->get_type();
    if (!t) return false;
    t = t->stripType(SgType::STRIP_MODIFIER_TYPE | SgType::STRIP_TYPEDEF_TYPE);
    return isSgArrayType(t) != nullptr || isSgPointerType(t) != nullptr;
}

std::string CudaCodeGen::getElementType(SgInitializedName* var) {
    if (!var) return "void";
    SgType* t = var->get_type();
    if (!t) return "void";
    t = t->stripType(SgType::STRIP_MODIFIER_TYPE | SgType::STRIP_TYPEDEF_TYPE);

    // Strip array/pointer layers until base type.
    while (t) {
        if (SgArrayType* at = isSgArrayType(t)) {
            t = at->get_base_type();
        } else if (SgPointerType* pt = isSgPointerType(t)) {
            t = pt->get_base_type();
        } else {
            break;
        }
        if (t) t = t->stripType(SgType::STRIP_MODIFIER_TYPE | SgType::STRIP_TYPEDEF_TYPE);
    }
    return t ? t->unparseToString() : "void";
}

std::vector<std::string> CudaCodeGen::getArrayDimensions(SgInitializedName* var) {
    std::vector<std::string> dims;
    if (!var) return dims;
    SgType* t = var->get_type();
    if (!t) return dims;
    t = t->stripType(SgType::STRIP_MODIFIER_TYPE | SgType::STRIP_TYPEDEF_TYPE);

    while (t) {
        if (SgArrayType* at = isSgArrayType(t)) {
            SgExpression* idx = at->get_index();
            dims.push_back(idx ? idx->unparseToString() : "");
            SgType* base = at->get_base_type();
            if (!base) break;
            t = base->stripType(SgType::STRIP_MODIFIER_TYPE | SgType::STRIP_TYPEDEF_TYPE);
        } else if (SgPointerType* pt = isSgPointerType(t)) {
            // Pointer layers have no static dimension; stop.
            break;
        } else {
            break;
        }
    }
    return dims;
}

std::string CudaCodeGen::generateHostCode(const CudaLoopInfo& info) const {
    std::ostringstream oss;
    oss << "{\n";

    // Device pointer declarations.
    for (const auto& cv : info.vars) {
        if (cv.isArray) {
            oss << "    " << cv.elemType << "* d_" << cv.name << ";\n";
        } else {
            oss << "    " << cv.elemType << " d_" << cv.name << ";\n";
        }
    }

    // cudaMalloc for arrays and scalars that need device copies.
    for (const auto& cv : info.vars) {
        oss << "    cudaMalloc((void**)&d_" << cv.name << ", "
            << cv.totalSizeExpr << ");\n";
    }

    // Host-to-device copies.
    for (const auto& cv : info.vars) {
        if (cv.direction == "to" || cv.direction == "tofrom") {
            oss << "    cudaMemcpy(d_" << cv.name << ", " << cv.name << ", "
                << cv.totalSizeExpr << ", cudaMemcpyHostToDevice);\n";
        }
    }

    // Kernel launch.
    std::string totalExpr;
    for (size_t i = 0; i < info.loopBounds.size(); ++i) {
        if (!totalExpr.empty()) totalExpr += " * ";
        totalExpr += "(" + info.loopBounds[i].second + ")";
    }
    if (totalExpr.empty()) totalExpr = "1";

    int kid = info.loop->get_file_info()->get_line();  // use line as id
    oss << "    long loomx_total_" << kid << " = " << totalExpr << ";\n";
    oss << "    int loomx_threads_" << kid << " = 256;\n";
    oss << "    int loomx_blocks_" << kid << " = (loomx_total_" << kid
        << " + loomx_threads_" << kid << " - 1) / loomx_threads_" << kid << ";\n";

    oss << "    loomx_cuda_kernel_" << kid << "<<<loomx_blocks_" << kid
        << ", loomx_threads_" << kid << ">>>(";
    bool firstArg = true;
    for (const auto& cv : info.vars) {
        if (!firstArg) oss << ", ";
        firstArg = false;
        oss << "d_" << cv.name;
    }
    // Pass loop bounds as kernel arguments so the kernel can bounds-check and
    // reconstruct any inner loop bounds used by the body.
    for (const auto& b : info.loopBounds) {
        oss << ", " << b.second;
    }
    oss << ");\n";
    oss << "    cudaDeviceSynchronize();\n";

    // Device-to-host copies.
    for (const auto& cv : info.vars) {
        if (cv.direction == "from" || cv.direction == "tofrom") {
            oss << "    cudaMemcpy(" << cv.name << ", d_" << cv.name << ", "
                << cv.totalSizeExpr << ", cudaMemcpyDeviceToHost);\n";
        }
    }

    // cudaFree.
    for (const auto& cv : info.vars) {
        oss << "    cudaFree(d_" << cv.name << ");\n";
    }

    oss << "}";
    return oss.str();
}

std::string CudaCodeGen::generateKernel(const CudaLoopInfo& info) const {
    std::ostringstream oss;
    oss << "__global__ void loomx_cuda_kernel_"
        << info.loop->get_file_info()->get_line() << "(";

    bool first = true;
    for (const auto& cv : info.vars) {
        if (!first) oss << ", ";
        first = false;
        if (cv.isArray) {
            oss << cv.elemType << "* " << cv.name;
        } else {
            oss << cv.elemType << " " << cv.name;
        }
    }
    for (const auto& b : info.loopBounds) {
        if (!first) oss << ", ";
        first = false;
        oss << "long " << b.first << "_bound";
    }
    oss << ") {\n";

    // Compute the parallel index from CUDA thread coordinates.
    oss << "    long " << info.loopBounds[0].first
        << " = blockIdx.x * blockDim.x + threadIdx.x;\n";
    oss << "    if (" << info.loopBounds[0].first
        << " >= " << info.loopBounds[0].first << "_bound) return;\n";

    // Private variables become local declarations in the kernel.
    for (const auto& pv : info.privateVars) {
        // Skip if it is one of the loop index variables or a kernel parameter.
        bool isBound = false;
        for (const auto& b : info.loopBounds) {
            if (b.first == pv) { isBound = true; break; }
        }
        if (!isBound) {
            oss << "    int " << pv << " = 0;\n";
        }
    }

    // Transform the body (flatten array subscripts, rename parallel index).
    std::string body = transformKernelBody(info);
    if (!body.empty() && body.front() != '{') {
        oss << "    " << body << "\n";
    } else {
        oss << body << "\n";
    }

    oss << "}\n";
    return oss.str();
}

std::string CudaCodeGen::transformKernelBody(const CudaLoopInfo& info) const {
    std::string body = info.bodyText;

    // Build a map of array name -> dimensions for subscript flattening.
    std::map<std::string, std::vector<std::string>> arrayDims;
    for (const auto& cv : info.vars) {
        if (cv.isArray && !cv.dims.empty()) {
            arrayDims[cv.name] = cv.dims;
        }
    }
    body = flattenArraySubscripts(body, arrayDims);

    // Rename the parallel loop index variable if it shadows a parameter.
    // The kernel already has a parameter named <idx>_bound; keep the original
    // index name for clarity, but we must avoid name collisions with arrays.
    // No rename needed because the body already uses the original index name
    // and that name is not a kernel parameter (the bound variable is).

    return body;
}

std::string CudaCodeGen::flattenArraySubscripts(
    const std::string& body,
    const std::map<std::string, std::vector<std::string>>& arrayDims) {
    std::string result = body;

    // Process arrays in deterministic order to avoid overlapping replacements.
    for (const auto& entry : arrayDims) {
        const std::string& name = entry.first;
        const std::vector<std::string>& dims = entry.second;
        if (dims.empty()) continue;

        // Build regex that matches name[<expr0>][<expr1>]... with up to dims.size() subscripts.
        // We match bracket pairs greedily, one at a time.
        std::string pattern = name;
        std::vector<std::string> groupPatterns;
        for (size_t i = 0; i < dims.size(); ++i) {
            pattern += R"(\[([^\[\]]+)\])";
        }

        // Convert std::regex is overkill for nested brackets; use a simple scanner.
        std::string replaced;
        size_t pos = 0;
        while (pos < result.size()) {
            size_t found = result.find(name, pos);
            if (found == std::string::npos) {
                replaced += result.substr(pos);
                break;
            }
            // Ensure identifier boundary.
            bool leftBoundary = (found == 0) || !isIdentChar(result[found - 1]);
            size_t afterName = found + name.size();
            bool rightBoundary = (afterName >= result.size()) ||
                               result[afterName] != '[' || !leftBoundary;
            if (!leftBoundary || rightBoundary) {
                replaced += result.substr(pos, found - pos + 1);
                pos = found + 1;
                continue;
            }

            // Try to collect dims.size() bracket pairs.
            size_t scan = afterName;
            std::vector<std::string> indices;
            bool ok = true;
            for (size_t d = 0; d < dims.size() && ok; ++d) {
                if (scan >= result.size() || result[scan] != '[') { ok = false; break; }
                size_t close = scan + 1;
                int depth = 1;
                while (close < result.size() && depth > 0) {
                    if (result[close] == '[') ++depth;
                    else if (result[close] == ']') --depth;
                    ++close;
                }
                if (depth != 0) { ok = false; break; }
                indices.push_back(trim(result.substr(scan + 1, close - scan - 2)));
                scan = close;
            }

            if (!ok || indices.size() != dims.size()) {
                replaced += result.substr(pos, found - pos + 1);
                pos = found + 1;
                continue;
            }

            // Build flattened index: ((idx0)*N1 + (idx1))*N2 + ...
            std::string flat = "(" + indices[0] + ")";
            for (size_t d = 1; d < indices.size(); ++d) {
                flat = "(" + flat + " * (" + dims[d] + ") + (" + indices[d] + "))";
            }

            replaced += result.substr(pos, found - pos) + name + "[" + flat + "]";
            pos = scan;
        }
        result = replaced;
    }

    return result;
}

bool CudaCodeGen::lineColToOffset(const std::string& source, int line, int col,
                                   size_t& offset) {
    int currentLine = 1;
    size_t pos = 0;
    while (currentLine < line && pos < source.size()) {
        if (source[pos] == '\n') ++currentLine;
        ++pos;
    }
    if (currentLine != line) return false;
    // Column is 1-based.
    if (pos + static_cast<size_t>(col) - 1 > source.size()) return false;
    offset = pos + static_cast<size_t>(col) - 1;
    return true;
}

void CudaCodeGen::postProcessSource(std::string& source) {
    if (cudaLoops_.empty()) return;

    // Sort loops by reverse source position so earlier offsets remain valid.
    std::sort(cudaLoops_.begin(), cudaLoops_.end(),
              [](const CudaLoopInfo& a, const CudaLoopInfo& b) {
                  if (a.startLine != b.startLine) return a.startLine > b.startLine;
                  return a.startCol > b.startCol;
              });

    // Use the original source file as the base text.  ROSE's unparsed output may
    // reformat code, so text replacement on the original is more reliable.
    const std::string& originalFile = cudaLoops_[0].fileName;
    if (originalSourceCache_.find(originalFile) == originalSourceCache_.end()) {
        std::ifstream in(originalFile);
        std::string content((std::istreambuf_iterator<char>(in)),
                            std::istreambuf_iterator<char>());
        originalSourceCache_[originalFile] = content;
    }
    std::string base = originalSourceCache_[originalFile];

    for (const auto& info : cudaLoops_) {
        size_t startOffset = 0, endOffset = 0;
        if (!lineColToOffset(base, info.startLine, info.startCol, startOffset) ||
            !lineColToOffset(base, info.endLine, info.endCol, endOffset)) {
            std::cerr << "[CudaCodeGen] Could not locate loop at "
                      << info.startLine << ":" << info.startCol << " - "
                      << info.endLine << ":" << info.endCol << "\n";
            continue;
        }
        // get_endOfConstruct() points at the last token of the statement.
        // Include that token ('}' for a braced body, ';' for a single-statement
        // body) so the whole loop is replaced.
        if (endOffset < base.size() &&
            (base[endOffset] == '}' || base[endOffset] == ';')) {
            ++endOffset;
        }
        std::string hostCode = generateHostCode(info);
        base = base.substr(0, startOffset) + hostCode + base.substr(endOffset);
    }

    // Generate all kernels and prepend the CUDA runtime header.
    std::ostringstream header;
    header << "#include <cuda_runtime.h>\n\n";
    for (const auto& info : cudaLoops_) {
        header << generateKernel(info) << "\n";
    }

    source = header.str() + base;
}
