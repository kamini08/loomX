/* pointer_escape.c: callee stashes a pointer in a global, causing escape.
 * Expected: rejected as unsafe.
 */
#include "bench_minimal.h"

static double* g_saved = NULL;

static void save_ptr(double* p) {
    g_saved = p;
}

int main(int argc, char** argv) {
    int n = (argc > 1) ? atoi(argv[1]) : 200000;
    double* a = (double*)malloc(sizeof(double) * n);
    for (int i = 0; i < n; i++) {
        a[i] = (double)i;
    }

    for (int i = 0; i < n; i++) {
        save_ptr(&a[i]);
    }

    printf("saved value: %.6f\n", g_saved ? *g_saved : 0.0);
    free(a);
    return 0;
}
