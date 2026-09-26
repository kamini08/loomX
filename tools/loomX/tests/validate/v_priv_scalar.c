// Numerical validation: per-iteration scalar privatization.
// Exercises: a scalar that must be private to each loop iteration, not shared
// across the gang. If private() is missing the interleaved iterations clobber
// each other and the count is wrong -- a race that an exit-code check on a
// truncated int would very likely miss.
#include <stdio.h>

#define N 500000

int main(void) {
    double total = 0.0;
    long i;

    for (i = 0; i < N; i++) {
        double local = (double)i * 2.0;
        local = local * 1.5;
        local = local + 1.0;
        total = total + local;
    }

    printf("priv_total %.17g\n", total);
    return 0;
}
