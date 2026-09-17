/* aliased_pointer_write.c: callee writes through a pointer that aliases
 * the input array in the caller. Expected: rejected due to possible aliasing.
 */
#include "bench_minimal.h"

static void write_scaled(double x, double* out) {
    *out = x * 2.0;
}

int main(int argc, char** argv) {
    int n = (argc > 1) ? atoi(argv[1]) : 200000;
    double* a = (double*)malloc(sizeof(double) * n);
    for (int i = 0; i < n; i++) {
        a[i] = (double)i;
    }

    for (int i = 0; i < n; i++) {
        write_scaled(a[i], &a[i]);
    }

    double checksum = 0.0;
    for (int i = 0; i < n; i++) {
        checksum += a[i];
    }
    printf("checksum: %.6f\n", checksum);
    free(a);
    return 0;
}
