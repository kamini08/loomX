// Numerical validation: 2D Jacobi stencil with halos.
// Exercises: read-only access to the previous time level, write-only to the
// next, and per-iteration privatized loop indices. The inner loop is parallel
// here, which is the case that suffers false sharing if the chunking is wrong
// -- a false-sharing bug shows up as a slowdown, not as a wrong answer, so
// this file's job is mainly to pin the numerics while the harness times it.
#include <stdio.h>

#define NX 1024
#define NY 1024

static double u[2][NY + 2][NX + 2];

int main(void) {
    int sweep, i, j;
    int cur = 0, nxt = 1;
    double checksum = 0.0;

    for (j = 0; j < NY + 2; j++) {
        for (i = 0; i < NX + 2; i++) {
            u[0][j][i] = 0.0;
            u[1][j][i] = 0.0;
        }
    }
    /* Non-trivial initial condition so the stencil actually propagates. */
    for (j = 1; j <= NY; j++) {
        for (i = 1; i <= NX; i++) {
            u[0][j][i] = 1.0 + (double)(i % 13) + (double)(j % 11) * 0.5;
        }
    }

    for (sweep = 0; sweep < 20; sweep++) {
        for (j = 1; j <= NY; j++) {
            for (i = 1; i <= NX; i++) {
                u[nxt][j][i] = 0.25 * (u[cur][j][i - 1] + u[cur][j][i + 1] +
                                       u[cur][j - 1][i] + u[cur][j + 1][i]);
            }
        }
        {
            int tmp = cur;
            cur = nxt;
            nxt = tmp;
        }
    }

    for (j = 1; j <= NY; j += 97) {
        for (i = 1; i <= NX; i += 89) {
            checksum = checksum + u[cur][j][i];
        }
    }

    printf("jacobi_checksum %.17g\n", checksum);
    return 0;
}
