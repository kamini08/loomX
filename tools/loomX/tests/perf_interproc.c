// Large-scale interprocedural benchmark for loomX
// 3-level call tree, pure functions, should parallelize with IPA

#define N 5000000

double data[N];
double result[N];

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
    int i;
    double sum = 0.0;

    for (i = 0; i < N; i++) {
        data[i] = (double)i / (double)N;
    }

    for (i = 0; i < N; i++) {
        result[i] = compute(data[i], i);
    }

    for (i = 0; i < N; i++) {
        sum = sum + result[i];
    }

    return (int)(sum / (double)N);
}
