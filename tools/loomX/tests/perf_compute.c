// Compute-intensive benchmark for loomX GPU measurement
// Heavy floating-point ops per element to stress GPU compute

#define N 8000000

double input[N];
double output[N];

double heavy_compute(double x, int idx) {
    double a = x * 2.0 + (double)idx * 0.001;
    double b = a * a - a + 1.0;
    double c = b * b + b * 0.5;
    double d = c * c - c * 1.5 + 2.0;
    double e = d * d * 0.1 + d * 0.9;
    return e;
}

int main() {
    int i;
    double sum = 0.0;

    for (i = 0; i < N; i++) {
        input[i] = (double)i / (double)N;
    }

    for (i = 0; i < N; i++) {
        output[i] = heavy_compute(input[i], i);
    }

    for (i = 0; i < N; i++) {
        sum = sum + output[i];
    }

    return (int)(sum / (double)N);
}
