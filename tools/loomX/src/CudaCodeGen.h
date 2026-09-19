#pragma once
#include "CodeGen.h"
#include "rose.h"
#include "LoopSummary.h"
#include <map>
#include <set>
#include <string>
#include <vector>

// CUDA source-to-source backend.  GPU_OFFLOAD loops are extracted into
// __global__ kernels; the original loop is replaced by host CUDA runtime calls
// for allocation, transfer, launch, and cleanup.
//
// Current scope:
//   - collapse depth == 1 (outer loop parallelized, inner loops sequential)
//   - arrays with static or parameter array types (1D/2D/3D)
//   - scalar bounds passed by value
//   - reductions and scalar writes are not yet supported
class CudaCodeGen : public CodeGen {
public:
    void generatePragmas(const loomX::LoopSummary& summary) override;
    void postProcessSource(std::string& source) override;

private:
    struct CudaVariable {
        SgInitializedName* var = nullptr;
        std::string name;
        std::string direction;  // "to", "from", "tofrom"
        bool isArray = false;
        std::string elemType;
        std::vector<std::string> dims;  // innermost first for array subscript flattening
        std::string totalSizeExpr;      // host-side transfer size expression
    };

    struct CudaLoopInfo {
        std::string fileName;
        int startLine = 0;
        int startCol = 0;
        int endLine = 0;
        int endCol = 0;

        SgForStatement* loop = nullptr;
        int collapseDepth = 1;

        // Parallel loop bounds (outer-to-inner for collapsed loops).
        std::vector<std::pair<std::string, std::string>> loopBounds;  // (indexVar, upperBound)

        std::vector<CudaVariable> vars;
        std::set<std::string> privateVars;
        bool hasReduction = false;

        std::string bodyText;  // unparsed loop body (sequential remainder)
    };

    std::vector<CudaLoopInfo> cudaLoops_;
    std::map<std::string, std::string> originalSourceCache_;
    int nextKernelId_ = 0;

    bool canTransform(const loomX::LoopSummary& summary) const;
    CudaLoopInfo buildLoopInfo(const loomX::LoopSummary& summary);

    std::vector<CudaVariable> collectVariables(const loomX::LoopSummary& summary);
    static bool isArrayVariable(SgInitializedName* var);
    static std::string getElementType(SgInitializedName* var);
    static std::vector<std::string> getArrayDimensions(SgInitializedName* var);

    std::string generateHostCode(const CudaLoopInfo& info) const;
    std::string generateKernel(const CudaLoopInfo& info) const;
    std::string transformKernelBody(const CudaLoopInfo& info) const;

    static std::string flattenArraySubscripts(
        const std::string& body,
        const std::map<std::string, std::vector<std::string>>& arrayDims);

    static bool lineColToOffset(const std::string& source, int line, int col,
                                size_t& offset);
};
