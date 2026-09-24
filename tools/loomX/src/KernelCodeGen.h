#pragma once
#include "rose.h"
#include "LoopSummary.h"
#include "LoopAnalysisTypes.h"
#include <map>
#include <set>
#include <string>
#include <vector>

namespace loomX {

// Kind of a kernel/host-stub parameter.
enum class KernelParamKind { ARRAY, SCALAR, REDUCTION };

// One variable passed between the host launcher stub and the device kernel.
struct KernelParam {
    KernelParamKind kind = KernelParamKind::SCALAR;
    std::string name;      // original variable name ("A", "sum", "n")
    std::string typeName;  // scalar C type, or the array element type
    std::string sizeExpr;  // buffer size in bytes ("sizeof(double)*(N)*(M)")
    bool mappedTo = true;  // host -> device copy
    bool mappedFrom = true;// device -> host copy
};

// Everything needed to emit one device kernel and its host launcher stub.
struct GeneratedKernel {
    int id = -1;
    CodeGenBackend backend = CodeGenBackend::CUDA;
    std::string hostName;      // static host launcher: loomx_<id>
    std::string kernelName;    // device kernel:        loomx_<id>_kernel
    std::vector<KernelParam> params;
    std::string bodyText;               // transformed kernel body statements
    std::vector<std::string> prologue;  // declarations emitted at kernel top

    // Collapsed loop dimensions, outermost first.  For a non-collapsed loop
    // there is exactly one dim.
    struct Dim {
        std::string lowerText;
        std::string upperText;
        bool upperExclusive = false;  // true when the upper bound is `<`, not `<=`
        std::string indexName;
    };
    std::vector<Dim> dims;

    std::string hostSignature;  // full prototype, e.g. "static void loomx_0(void* A, int n, double* sum)"

    // Complete device-kernel + host-launcher text for CUDA, appended to the
    // output file by finalizeOutput().
    std::string definitionText;

    // OpenCL only: the whole kernel source (helpers + kernel function) with a
    // slot, %%LOOMX_DEFINES%%, replaced by the file's #define lines in
    // finalizeOutput().
    std::string openclSource;

    // Product of the collapsed trip counts, as a host-scope C expression using
    // the original loop-bound variables (used by the host launcher stubs).
    std::string totalTripText;

    bool needsReductionHelpers = false;
};

// Generates CUDA / OpenCL device code from loops whose LoopSummary decided
// GPU_OFFLOAD.  On success the loop is removed from the AST and replaced by a
// call to a generated host launcher; the launcher and kernel text are recorded
// and spliced into the unparsed file by finalizeOutput().  Loops that cannot
// be expressed as a kernel (unsupported reductions, incomplete array extents,
// live-out scalars, ...) are refused and left untouched; the caller falls back
// to CPU OpenMP.
class KernelCodeGen {
public:
    explicit KernelCodeGen(CodeGenBackend backend) : backend_(backend) {}

    // Try to kernelize `loop` (must have summary.target == GPU_OFFLOAD).
    // Returns true and records the generated code on success; returns false
    // without touching the AST when the loop is not kernelizable.
    bool generate(SgForStatement* loop, const loomX::LoopSummary& summary);

    // Called after project->unparse(): reads the emitted text, splices the
    // launcher prototypes near the top and the generated definitions at the
    // bottom, and adds the required device-runtime include.  `ompUsed` is true
    // when CPU OpenMP pragmas were also emitted (mixed-target output).
    void finalizeOutput(std::string& source, bool ompUsed) const;

    int kernelCount() const { return static_cast<int>(kernels_.size()); }

private:
    CodeGenBackend backend_;
    std::vector<GeneratedKernel> kernels_;
    int nextId_ = 0;

    // Build the CUDA host launcher stub referencing an already-emitted kernel.
    std::string buildHostStubCUDA(const GeneratedKernel& gen,
                                  const std::vector<std::string>& kernelParams,
                                  const std::string& hostParamsText,
                                  const std::string& totalTrip);

    // Build the OpenCL kernel source (helpers + kernel function + reduction
    // macros) for one kernel.
    std::string buildOpenCLKernel(const GeneratedKernel& gen,
                                  const std::vector<std::string>& kernelParams,
                                  const std::string& bodyText,
                                  const std::string& prologueText,
                                  const std::string& totalTrip);

    // Host launcher parameter list (without surrounding parentheses) for an
    // OpenCL stub.
    std::string hostLaunchSignature(const GeneratedKernel& g) const;
};

} // namespace loomX