/* Non-canonical but memory-safe pointer access: the subscript is `i + 1`, not
   the bare induction variable, and the trip bound is `i + 1 < N`. The extent
   is still provable (a[0:N] is a correct, slightly conservative mapping), so
   this checks that the shape logic handles an offset subscript and does not
   mistake it for the unprovable case.
   Deliberately avoids a[i*3+1]-style access: that walks past the allocation
   and overflows the heap in plain C too, so it tests nothing about loomX. */
#include <stdlib.h>
#include <stdio.h>
#define N 4000000
int main(void)
{
    int i;
    double *a = (double *)malloc(sizeof(double) * N);
    for (i = 0; i < N; i++)
        a[i] = (double)(i % 100);
    for (i = 0; i + 1 < N; i++)
        a[i + 1] = a[i + 1] + 1.0;
    printf("shape_ptr_odd_sum %.6f\n", a[1] + a[N - 1]);
    free(a);
    return 0;
}
