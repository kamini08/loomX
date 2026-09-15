#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Leaf helper function: safe to call in parallel
// Computes a simple arithmetic operation
static double compute_element(double x, int idx) {
    double result = x * 2.0 + (double)idx * 0.01;
    // Simple polynomial instead of sin/cos to avoid __float128
    return result * result - result + 1.0;
}

// Another leaf helper
static int classify_value(double val) {
    if (val > 0.5) return 1;
    if (val < -0.5) return -1;
    return 0;
}

int main(int argc, char** argv) {
    int N = 100000;
    if (argc > 1) N = atoi(argv[1]);

    double* input = (double*)malloc(N * sizeof(double));
    double* output = (double*)malloc(N * sizeof(double));
    int* classes = (int*)malloc(N * sizeof(int));

    // Initialize input
    for (int i = 0; i < N; i++) {
        input[i] = (double)i / (double)N;
    }

    // --- Loop 1: GPU-suitable ---
    // Large iteration count, regular memory access, compute-intensive,
    // calls leaf helper function
    for (int i = 0; i < N; i++) {
        output[i] = compute_element(input[i], i);
    }

    // --- Loop 2: CPU-suitable ---
    // Small iteration count, not worth GPU launch overhead
    for (int i = 0; i < 100; i++) {
        classes[i] = classify_value(output[i]);
    }

    // --- Loop 3: GPU-suitable ---
    // Reduction-like pattern (sum), regular access
    double sum = 0.0;
    for (int i = 0; i < N; i++) {
        sum += output[i];
    }

    printf("N=%d, sum=%f, class[0]=%d\n", N, sum, classes[0]);

    free(input);
    free(output);
    free(classes);
    return 0;
}
