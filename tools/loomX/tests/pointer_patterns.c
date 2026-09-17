// Test: extended pointer-parameter patterns for interprocedural analysis.
// The read-only and loop-disjoint writes should be safe inside a parallel loop.

#define N 100000

double data[N];
double result[N];

// Read-only through pointer arithmetic.
void read_ptr_arith(double* p, int idx) {
    double x = *(p + idx);
    (void)x;
}

// Write through pointer arithmetic, indexed by parameter -> loop-disjoint.
void write_ptr_arith(double* p, int idx) {
    *(p + idx) = (double)idx;
}

// Write with constant offset, indexed by parameter -> loop-disjoint.
void write_offset(double* arr, int idx) {
    arr[idx + 1] = (double)idx;
}

// Write with constant subtraction, indexed by parameter -> loop-disjoint.
void write_offset_sub(double* arr, int idx) {
    arr[idx - 1] = (double)idx;
}

int main() {
    int i;

    for (i = 0; i < N; i++) {
        data[i] = (double)i / (double)N;
    }

    for (i = 0; i < N; i++) {
        read_ptr_arith(data, i);  // safe
    }

    for (i = 1; i < N - 1; i++) {
        write_ptr_arith(result, i);  // loop-disjoint safe
    }

    for (i = 0; i < N - 1; i++) {
        write_offset(result, i);  // loop-disjoint safe
    }

    for (i = 1; i < N; i++) {
        write_offset_sub(result, i);  // loop-disjoint safe
    }

    return 0;
}
