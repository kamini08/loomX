// Demonstration: loop-carried disjoint pointer writes through a helper.
// An intraprocedural tool must reject this loop because apply_kernel writes
// through a pointer parameter. loomX's interprocedural analysis recognizes
// that the write is indexed by the loop iterator, so iterations are disjoint.
void apply_kernel(double* out, double x, int i) {
    out[i] = x * 2.0 + (double)i * 0.01;
}

int main(void) {
    int n = 1000;
    double out[1000];
    double in[1000];

    for (int j = 0; j < n; j++) {
        in[j] = (double)j / (double)n;
    }

    for (int i = 0; i < n; i++) {
        apply_kernel(out, in[i], i);
    }

    // Sanity checksum.
    double sum = 0.0;
    for (int j = 0; j < n; j++) {
        sum += out[j];
    }

    // Return a small integer derived from the checksum so the validation
    // suite can compare sequential vs. parallel results.
    return (int)(sum / (double)n);
}
