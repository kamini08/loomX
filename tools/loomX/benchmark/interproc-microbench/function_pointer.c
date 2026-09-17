/* function_pointer.c: hot loop dispatches through a function pointer.
 * This stresses interprocedural analysis when the callee set is unknown.
 * Expected: may be rejected by a conservative analyzer.
 */
#include "bench_minimal.h"

typedef double (*unary_op_t)(double);

static double scale(double x) {
    return x * 3.0;
}

static double offset(double x) {
    return x + 7.0;
}

int main(int argc, char** argv) {
    int n = (argc > 1) ? atoi(argv[1]) : 200000;
    double* a = (double*)malloc(sizeof(double) * n);
    double* b = (double*)malloc(sizeof(double) * n);
    for (int i = 0; i < n; i++) {
        a[i] = (double)i;
        b[i] = 0.0;
    }

    unary_op_t op = (n % 2 == 0) ? scale : offset;

    for (int i = 0; i < n; i++) {
        b[i] = op(a[i]);
    }

    double checksum = 0.0;
    for (int i = 0; i < n; i++) {
        checksum += b[i];
    }
    printf("checksum: %.6f\n", checksum);
    free(a);
    free(b);
    return 0;
}
