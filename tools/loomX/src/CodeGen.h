#pragma once
#include "LoopSummary.h"
#include <string>

// Backend-agnostic interface for inserting parallelization directives into
// the ROSE AST and for post-processing the unparsed source.
class CodeGen {
public:
    virtual ~CodeGen() = default;

    // Insert all directives required by the summary decision.
    virtual void generatePragmas(const loomX::LoopSummary& summary) = 0;

    // Backend-specific post-processing of the unparsed source text.  Called
    // once per output file after project->unparse().  The default does nothing.
    virtual void postProcessSource(std::string& source) {
        (void)source;
    }
};
