// Numerical validation: disjoint writes through aliased pointers.
// Exercises: the scalar loop-carried dependence guard. Two pointers derived
// from the same base array index disjoint element sets, so the loop IS
// parallelizable -- but only if the analysis resolves the pointer arithmetic.
// A conservative rejection is safe (just slower); an unsound acceptance here
// silently loses updates, which is what this checksum detects.
#include <stdio.h>

#define N 1000003

static double buf[N];
static double lo[N / 2];
static double hi[N / 2];

int main(void) {
    double *a = buf;
    double *b = buf + N / 2;
    double checksum = 0.0;
    long i;

    for (i = 0; i < N; i++) {
        buf[i] = 0.0;
    }

    /* Disjoint halves: no two iterations touch the same element. */
    for (i = 0; i < N / 2; i++) {
        lo[i] = (double)i * 1.25 + 1.0;
        hi[i] = (double)i * 0.75 + 2.0;
    }

    for (i = 0; i < N / 2; i++) {
        a[i] = lo[i];
        b[i] = hi[i];
    }

    for (i = 0; i < N; i++) {
        checksum = checksum + buf[i] * 0.001;
    }

    printf("alias_checksum %.17g\n", checksum);
    printf("alias_mid %.17g\n", buf[N / 2]);
    return 0;
}
