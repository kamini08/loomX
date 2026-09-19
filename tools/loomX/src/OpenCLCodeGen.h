#pragma once
#include "CodeGen.h"
#include "rose.h"
#include "LoopSummary.h"
#include <map>
#include <set>
#include <string>
#include <vector>

// OpenCL source-to-source backend.  GPU_OFFLOAD loops are extracted into
// __kernel functions; the original loop is replaced by host OpenCL runtime
// calls for context/queue creation, buffer management, kernel build/launch,
// and cleanup.
//
// Current scope matches the CUDA backend:
//   - collapse depth >= 1 (outer loop parallelized, inner loops sequential)
//   - 1D/2D/3D static or parameter arrays with subscript flattening
//   - scalar bounds passed by value
//   - reductions are not yet supported
class OpenCLCodeGen : public CodeGen {
public:
    void generatePragmas(const loomX::LoopSummary& summary) override;
    void postProcessSource(std::string& source) override;

private:
    struct OpenCLVariable {
        SgInitializedName* var = nullptr;
        std::string name;
        std::string direction;  // "to", "from", "tofrom"
        bool isArray = false;
        std::string elemType;
        std::vector<std::string> dims;
        std::string totalSizeExpr;
    };

    struct OpenCLLoopInfo {
        std::string fileName;
        int startLine = 0, startCol = 0, endLine = 0, endCol = 0;
        SgForStatement* loop = nullptr;
        int collapseDepth = 1;
        std::vector<std::pair<std::string, std::string>> loopBounds;
        std::vector<OpenCLVariable> vars;
        std::set<std::string> privateVars;
        std::string bodyText;
    };

    std::vector<OpenCLLoopInfo> openclLoops_;
    std::map<std::string, std::string> originalSourceCache_;

    bool canTransform(const loomX::LoopSummary& summary) const;
    OpenCLLoopInfo buildLoopInfo(const loomX::LoopSummary& summary);
    std::vector<OpenCLVariable> collectVariables(const loomX::LoopSummary& summary);

    static bool isArrayVariable(SgInitializedName* var);
    static std::string getElementType(SgInitializedName* var);
    static std::vector<std::string> getArrayDimensions(SgInitializedName* var);

    std::string generateHostCode(const OpenCLLoopInfo& info) const;
    std::string generateKernel(const OpenCLLoopInfo& info) const;
    std::string transformKernelBody(const OpenCLLoopInfo& info) const;

    static std::string flattenArraySubscripts(
        const std::string& body,
        const std::map<std::string, std::vector<std::string>>& arrayDims);

    static bool lineColToOffset(const std::string& source, int line, int col,
                                size_t& offset);
};
