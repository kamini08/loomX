// Test: standard library pure function whitelist
// sin and sqrt are pure math functions and should be safe to call in parallel.

#define N 100000

// Forward declarations avoid pulling in <math.h> (ROSE cannot parse its
// __float128 declarations on this system).
double sin(double x);
double sqrt(double x);

double data[N];
double result[N];

int main() {
    int i;

    for (i = 0; i < N; i++) {
        data[i] = (double)i / (double)N;
    }

    for (i = 0; i < N; i++) {
        result[i] = sin(data[i]) + sqrt(data[i] + 1.0);
    }

    return 0;
}
