// Interprocedural analysis demo
// This loop calls a helper function. Without interprocedural analysis,
// a compiler would conservatively skip parallelization because it
// doesn't know if the helper has side effects.
// Our tool analyzes the helper and confirms it's a pure leaf function.

double helper_a(double x) {
    return x * x + 1.0;
}

double helper_b(double x) {
    return helper_a(x) * 2.0 - helper_a(x * 0.5);
}

double compute(double x, int idx) {
    double a = helper_b(x);
    double b = helper_a((double)idx);
    return a + b;
}

int main() {
    int N = 50000;
    double data[50000];
    double result[50000];
    int i;

    for (i = 0; i < N; i++) {
        data[i] = (double)i / (double)N;
    }

    for (i = 0; i < N; i++) {
        result[i] = compute(data[i], i);
    }

    double sum = 0.0;
    for (i = 0; i < N; i++) {
        sum = sum + result[i];
    }

    return (int)(sum / 10000.0);
}
