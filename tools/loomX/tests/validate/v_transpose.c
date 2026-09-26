// Numerical validation: 2D matrix transpose.
// Exercises: a loop nest with a reversed access pattern on the destination.
// The transpose is a permutation of a fixed data set, so the result is
// exactly reproducible and must be bitwise identical. It is the second
// control case alongside v_int_reduce, and it also stresses chunking, since
// the naive parallel form collides on cache lines without blocking.
#include <stdio.h>

#define N 768

static double src[N][N];
static double dst[N][N];

int main(void) {
    int i, j;
    double checksum = 0.0;

    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            src[i][j] = (double)(i * N + j) * 0.5;
            dst[i][j] = 0.0;
        }
    }

    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            dst[j][i] = src[i][j];
        }
    }

    for (i = 0; i < N; i++) {
        checksum = checksum + dst[i][i] + dst[i][N - 1 - i];
    }

    printf("transpose_checksum %.17g\n", checksum);
    printf("transpose_sample %.17g\n", dst[N / 3][N / 3 + 1]);
    return 0;
}
