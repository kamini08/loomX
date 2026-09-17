#include <stdio.h>

int main(void) {
    int values[1000];
    int sum = 0;

#pragma scop
#pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < 1000; ++i) {
        values[i] = i;
        sum += values[i];
    }
#pragma endscop

    printf("%d\n", sum);
    return sum == 499500 ? 0 : 1;
}