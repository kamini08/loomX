// Numerical validation: subtraction reduction.
// Exercises: the operator in a reduction must be '-', not '+'. The existing
// smoke test reduction_sub.c encodes its result in the exit code, where
// 500500/500756/501012 all alias to 20 -- so an operator bug is invisible
// there. Here the value is printed at full precision instead.
#include <stdio.h>

#define N 1000

int main(void) {
    int total = 1000000;
    int i;

    for (i = 0; i < N; i++) {
        total = total - i;
    }

    printf("sub_reduce %d\n", total);
    return 0;
}
