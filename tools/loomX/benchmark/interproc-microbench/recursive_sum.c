/* recursive_sum.c: hot loop body contains a small recursive call.
 * This tests whether the analyzer safely handles recursion.
 * Expected: conservative analyzers reject; advanced ones may accept.
 */
#include "bench_minimal.h"

static int triangular(int k) {
    if (k <= 0) return 0;
    return k + triangular(k - 1);
}

int main(int argc, char** argv) {
    int n = (argc > 1) ? atoi(argv[1]) : 100000;
    int* a = (int*)malloc(sizeof(int) * n);
    long* b = (long*)malloc(sizeof(long) * n);
    for (int i = 0; i < n; i++) {
        a[i] = i % 32;
        b[i] = 0;
    }

    for (int i = 0; i < n; i++) {
        b[i] = triangular(a[i]);
    }

    long checksum = 0;
    for (int i = 0; i < n; i++) {
        checksum += b[i];
    }
    printf("checksum: %ld\n", checksum);
    free(a);
    free(b);
    return 0;
}
