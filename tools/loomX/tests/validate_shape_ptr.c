/* Fixture for check_pointer_map_shape.py. Never executed: the test inspects the
   emitted map clause only. Shaped like the interproc-microbench pointer
   kernels (main + malloc + int induction variable) so the loop actually
   reaches the offload path; N is large enough to clear the profitability
   gates without needing a big allocation at run time. */
#include <stdlib.h>
#include <stdio.h>
#define N 4000000
int main(void)
{
    int i;
    double *a = (double *)malloc(sizeof(double) * N);
    for (i = 0; i < N; i++)
        a[i] = (double)(i % 100);
    for (i = 0; i < N; i++)
        a[i] = a[i] + 1.0;
    printf("shape_ptr_sum %.6f\n", a[0] + a[N / 2] + a[N - 1]);
    free(a);
    return 0;
}
