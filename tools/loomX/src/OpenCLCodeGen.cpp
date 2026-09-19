#include "OpenCLCodeGen.h"
#include "LoopAnalysisUtil.h"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>

using namespace loomX;

namespace {

std::string trim(const std::string& s) {
    size_t a = s.find_first_not_of(" \t\n\r");
    if (a == std::string::npos) return "";
    size_t b = s.find_last_not_of(" \t\n\r");
    return s.substr(a, b - a + 1);
}

bool isIdentChar(char c) {
    return std::isalnum(static_cast<unsigned char>(c)) || c == '_';
}

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

} // namespace

void OpenCLCodeGen::generatePragmas(const loomX::LoopSummary& summary) {
    if (summary.target != ParallelTarget::GPU_OFFLOAD) return;
    if (!canTransform(summary)) return;
    openclLoops_.push_back(buildLoopInfo(summary));
}

bool OpenCLCodeGen::canTransform(const loomX::LoopSummary& summary) const {
    if (summary.collapseDepth < 1) return false;
    if (!summary.reductions.empty()) {
        std::cerr << "[OpenCLCodeGen] reductions not yet supported (line "
                  << summary.loop->get_file_info()->get_line() << ")\n";
        return false;
    }
    return true;
}

OpenCLCodeGen::OpenCLLoopInfo OpenCLCodeGen::buildLoopInfo(
    const loomX::LoopSummary& summary) {
    OpenCLLoopInfo info;
    info.loop = summary.loop;
    info.collapseDepth = summary.collapseDepth;

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

    SgForStatement* current = summary.loop;
    SgInitializedName* idx = SageInterface::getLoopIndexVariable(current);
    SgExpression* ub = summary.canonical.upperBound;
    if (!ub) {
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

    SgStatement* body = summary.loop->get_loop_body();
    if (body) info.bodyText = trim(body->unparseToString());

    return info;
}

std::vector<OpenCLCodeGen::OpenCLVariable> OpenCLCodeGen::collectVariables(
    const loomX::LoopSummary& summary) {
    std::vector<OpenCLVariable> result;
    for (const auto& entry : summary.mapClauses) {
        OpenCLVariable cv;
        cv.var = entry.first;
        cv.name = cv.var->get_name().getString();
        cv.direction = entry.second;
        cv.isArray = isArrayVariable(cv.var);
        cv.elemType = getElementType(cv.var);
        cv.dims = getArrayDimensions(cv.var);

        if (cv.isArray) {
            if (cv.dims.empty()) {
                cv.totalSizeExpr =
                    "(" + summary.canonical.upperBound->unparseToString() +
                    ") * sizeof(" + cv.elemType + ")";
            } else {
                cv.totalSizeExpr = "sizeof(" + cv.name + ")";
            }
        } else {
            cv.totalSizeExpr = "sizeof(" + cv.elemType + ")";
        }

        result.push_back(cv);
    }
    return result;
}

bool OpenCLCodeGen::isArrayVariable(SgInitializedName* var) {
    if (!var) return false;
    SgType* t = var->get_type();
    if (!t) return false;
    t = t->stripType(SgType::STRIP_MODIFIER_TYPE | SgType::STRIP_TYPEDEF_TYPE);
    return isSgArrayType(t) != nullptr || isSgPointerType(t) != nullptr;
}

std::string OpenCLCodeGen::getElementType(SgInitializedName* var) {
    if (!var) return "void";
    SgType* t = var->get_type();
    if (!t) return "void";
    t = t->stripType(SgType::STRIP_MODIFIER_TYPE | SgType::STRIP_TYPEDEF_TYPE);
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

std::vector<std::string> OpenCLCodeGen::getArrayDimensions(SgInitializedName* var) {
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
        } else if (isSgPointerType(t)) {
            break;
        } else {
            break;
        }
    }
    return dims;
}

std::string OpenCLCodeGen::generateHostCode(const OpenCLLoopInfo& info) const {
    std::ostringstream oss;
    oss << "{\n";

    int kid = info.loop->get_file_info()->get_line();

    // OpenCL platform/device/context/queue setup.
    oss << "    cl_platform_id loomx_platform_" << kid << ";\n"
        << "    clGetPlatformIDs(1, &loomx_platform_" << kid << ", NULL);\n"
        << "    cl_device_id loomx_device_" << kid << ";\n"
        << "    clGetDeviceIDs(loomx_platform_" << kid
        << ", CL_DEVICE_TYPE_DEFAULT, 1, &loomx_device_" << kid << ", NULL);\n"
        << "    cl_context loomx_context_" << kid
        << " = clCreateContext(NULL, 1, &loomx_device_" << kid
        << ", NULL, NULL, NULL);\n"
        << "    cl_command_queue loomx_queue_" << kid
        << " = clCreateCommandQueueWithProperties(loomx_context_" << kid
        << ", loomx_device_" << kid << ", NULL, NULL);\n";

    // Device buffers.
    for (const auto& cv : info.vars) {
        if (cv.isArray) {
            oss << "    cl_mem d_" << cv.name << " = clCreateBuffer(loomx_context_"
                << kid << ", CL_MEM_READ_WRITE, " << cv.totalSizeExpr
                << ", NULL, NULL);\n";
        }
    }

    // Write buffers.
    for (const auto& cv : info.vars) {
        if (!cv.isArray) continue;
        if (cv.direction == "to" || cv.direction == "tofrom") {
            oss << "    clEnqueueWriteBuffer(loomx_queue_" << kid << ", d_"
                << cv.name << ", CL_TRUE, 0, " << cv.totalSizeExpr << ", "
                << cv.name << ", 0, NULL, NULL);\n";
        }
    }

    // Program and kernel.
    oss << "    const char* loomx_kernel_src_" << kid << " = \""
        << generateKernel(info) << "\";\n"
        << "    cl_program loomx_program_" << kid
        << " = clCreateProgramWithSource(loomx_context_" << kid
        << ", 1, &loomx_kernel_src_" << kid << ", NULL, NULL);\n"
        << "    clBuildProgram(loomx_program_" << kid
        << ", 1, &loomx_device_" << kid << ", NULL, NULL, NULL);\n"
        << "    cl_kernel loomx_kernel_" << kid
        << " = clCreateKernel(loomx_program_" << kid
        << ", \"loomx_opencl_kernel_" << kid << "\", NULL);\n";

    // Set kernel arguments.
    int argIdx = 0;
    for (const auto& cv : info.vars) {
        if (cv.isArray) {
            oss << "    clSetKernelArg(loomx_kernel_" << kid << ", " << argIdx++
                << ", sizeof(cl_mem), &d_" << cv.name << ");\n";
        } else {
            oss << "    " << cv.elemType << " host_" << cv.name << " = "
                << cv.name << ";\n";
            oss << "    clSetKernelArg(loomx_kernel_" << kid << ", " << argIdx++
                << ", sizeof(" << cv.elemType << "), &host_" << cv.name << ");\n";
        }
    }
    for (const auto& b : info.loopBounds) {
        oss << "    long " << b.first << "_bound_" << kid << " = " << b.second
            << ";\n";
        oss << "    clSetKernelArg(loomx_kernel_" << kid << ", " << argIdx++
            << ", sizeof(long), &" << b.first << "_bound_" << kid << ");\n";
    }

    // Launch.
    std::string totalExpr;
    for (const auto& b : info.loopBounds) {
        if (!totalExpr.empty()) totalExpr += " * ";
        totalExpr += "(" + b.second + ")";
    }
    if (totalExpr.empty()) totalExpr = "1";

    oss << "    size_t loomx_global_" << kid << " = ((" << totalExpr
        << " + 255) / 256) * 256;\n"
        << "    size_t loomx_local_" << kid << " = 256;\n"
        << "    clEnqueueNDRangeKernel(loomx_queue_" << kid << ", loomx_kernel_"
        << kid << ", 1, NULL, &loomx_global_" << kid << ", &loomx_local_"
        << kid << ", 0, NULL, NULL);\n"
        << "    clFinish(loomx_queue_" << kid << ");\n";

    // Read buffers.
    for (const auto& cv : info.vars) {
        if (!cv.isArray) continue;
        if (cv.direction == "from" || cv.direction == "tofrom") {
            oss << "    clEnqueueReadBuffer(loomx_queue_" << kid << ", d_"
                << cv.name << ", CL_TRUE, 0, " << cv.totalSizeExpr << ", "
                << cv.name << ", 0, NULL, NULL);\n";
        }
    }

    // Cleanup.
    for (const auto& cv : info.vars) {
        if (cv.isArray) {
            oss << "    clReleaseMemObject(d_" << cv.name << ");\n";
        }
    }
    oss << "    clReleaseKernel(loomx_kernel_" << kid << ");\n"
        << "    clReleaseProgram(loomx_program_" << kid << ");\n"
        << "    clReleaseCommandQueue(loomx_queue_" << kid << ");\n"
        << "    clReleaseContext(loomx_context_" << kid << ");\n"
        << "}";
    return oss.str();
}

std::string OpenCLCodeGen::generateKernel(const OpenCLLoopInfo& info) const {
    std::ostringstream kernel;
    int kid = info.loop->get_file_info()->get_line();

    kernel << "\\n";
    kernel << "__kernel void loomx_opencl_kernel_" << kid << "(";
    bool first = true;
    for (const auto& cv : info.vars) {
        if (!first) kernel << ", ";
        first = false;
        if (cv.isArray) {
            kernel << "__global " << cv.elemType << "* " << cv.name;
        } else {
            kernel << cv.elemType << " " << cv.name;
        }
    }
    for (const auto& b : info.loopBounds) {
        if (!first) kernel << ", ";
        first = false;
        kernel << "long " << b.first << "_bound";
    }
    kernel << ") {\\n";

    kernel << "    long " << info.loopBounds[0].first
           << " = get_global_id(0);\\n";
    kernel << "    if (" << info.loopBounds[0].first
           << " >= " << info.loopBounds[0].first << "_bound) return;\\n";

    for (const auto& pv : info.privateVars) {
        bool isBound = false;
        for (const auto& b : info.loopBounds) {
            if (b.first == pv) { isBound = true; break; }
        }
        if (!isBound) {
            kernel << "    int " << pv << " = 0;\\n";
        }
    }

    std::string body = transformKernelBody(info);
    // Escape newlines inside the body for the host string literal.
    std::string escapedBody;
    for (char c : body) {
        if (c == '\n') escapedBody += "\\n";
        else if (c == '"') escapedBody += "\\\"";
        else if (c == '\\') escapedBody += "\\\\";
        else escapedBody += c;
    }
    if (!escapedBody.empty()) {
        kernel << "    " << escapedBody << "\\n";
    }

    kernel << "}\\n";
    return kernel.str();
}

std::string OpenCLCodeGen::transformKernelBody(const OpenCLLoopInfo& info) const {
    std::string body = info.bodyText;

    std::map<std::string, std::vector<std::string>> arrayDims;
    for (const auto& cv : info.vars) {
        if (cv.isArray && !cv.dims.empty()) {
            arrayDims[cv.name] = cv.dims;
        }
    }
    body = flattenArraySubscripts(body, arrayDims);
    return body;
}

std::string OpenCLCodeGen::flattenArraySubscripts(
    const std::string& body,
    const std::map<std::string, std::vector<std::string>>& arrayDims) {
    std::string result = body;

    for (const auto& entry : arrayDims) {
        const std::string& name = entry.first;
        const std::vector<std::string>& dims = entry.second;
        if (dims.empty()) continue;

        std::string replaced;
        size_t pos = 0;
        while (pos < result.size()) {
            size_t found = result.find(name, pos);
            if (found == std::string::npos) {
                replaced += result.substr(pos);
                break;
            }
            bool leftBoundary = (found == 0) || !isIdentChar(result[found - 1]);
            size_t afterName = found + name.size();
            bool rightBoundary = (afterName >= result.size()) ||
                               result[afterName] != '[' || !leftBoundary;
            if (!leftBoundary || rightBoundary) {
                replaced += result.substr(pos, found - pos + 1);
                pos = found + 1;
                continue;
            }

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

bool OpenCLCodeGen::lineColToOffset(const std::string& source, int line, int col,
                                    size_t& offset) {
    int currentLine = 1;
    size_t pos = 0;
    while (currentLine < line && pos < source.size()) {
        if (source[pos] == '\n') ++currentLine;
        ++pos;
    }
    if (currentLine != line) return false;
    if (pos + static_cast<size_t>(col) - 1 > source.size()) return false;
    offset = pos + static_cast<size_t>(col) - 1;
    return true;
}

void OpenCLCodeGen::postProcessSource(std::string& source) {
    if (openclLoops_.empty()) return;

    std::sort(openclLoops_.begin(), openclLoops_.end(),
              [](const OpenCLLoopInfo& a, const OpenCLLoopInfo& b) {
                  if (a.startLine != b.startLine) return a.startLine > b.startLine;
                  return a.startCol > b.startCol;
              });

    const std::string& originalFile = openclLoops_[0].fileName;
    if (originalSourceCache_.find(originalFile) == originalSourceCache_.end()) {
        std::ifstream in(originalFile);
        std::string content((std::istreambuf_iterator<char>(in)),
                            std::istreambuf_iterator<char>());
        originalSourceCache_[originalFile] = content;
    }
    std::string base = originalSourceCache_[originalFile];

    for (const auto& info : openclLoops_) {
        size_t startOffset = 0, endOffset = 0;
        if (!lineColToOffset(base, info.startLine, info.startCol, startOffset) ||
            !lineColToOffset(base, info.endLine, info.endCol, endOffset)) {
            std::cerr << "[OpenCLCodeGen] Could not locate loop at "
                      << info.startLine << ":" << info.startCol << " - "
                      << info.endLine << ":" << info.endCol << "\n";
            continue;
        }
        if (endOffset < base.size() &&
            (base[endOffset] == '}' || base[endOffset] == ';')) {
            ++endOffset;
        }
        std::string hostCode = generateHostCode(info);
        base = base.substr(0, startOffset) + hostCode + base.substr(endOffset);
    }

    std::ostringstream header;
    header << "#define CL_TARGET_OPENCL_VERSION 220\n"
           << "#ifdef __APPLE__\n"
           << "#include <OpenCL/opencl.h>\n"
           << "#else\n"
           << "#include <CL/cl.h>\n"
           << "#endif\n\n";

    source = header.str() + base;
}
