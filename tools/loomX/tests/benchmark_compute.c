// Compute-intensive benchmark: heavy per-element arithmetic
// N=5M, ~30 FLOPs per element
// Designed to show GPU speedup by amortizing data transfer overhead

static double heavy_compute(double x, int idx) {
    double a = x * 2.0 + (double)idx * 0.001;
    double b = a * a - a + 1.0;
    double c = b * b - b * 0.5 + 0.25;
    double d = c * c - c * 0.25 + 0.125;
    double e = d * d - d * 0.125 + 0.0625;
    return e;
}

static double helper1(double x) {
    return heavy_compute(x, 0) * heavy_compute(x, 1);
}

static double helper2(double x, int idx) {
    return helper1(x) + heavy_compute(x, idx);
}

int main() {
    int N = 5000000;
    double input[5000000];
    double output[5000000];
    int i;

    for (i = 0; i < N; i++) {
        input[i] = (double)i / (double)N;
    }

    for (i = 0; i < N; i++) {
        output[i] = helper2(input[i], i);
    }

    double sum = 0.0;
    for (i = 0; i < N; i++) {
        sum = sum + output[i];
    }

    return (int)(sum / 1000000.0);
}
