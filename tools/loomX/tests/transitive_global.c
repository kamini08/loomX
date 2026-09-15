// Test: transitive global side effect propagation
// helper_a writes to a global. helper_b calls helper_a. compute calls helper_b.
// All three should be marked unsafe by the interprocedural analysis.

#define N 100000

double data[N];
double result[N];
int global_counter = 0;

void helper_a(double x) {
    global_counter = global_counter + 1;  // global write
}

void helper_b(double x) {
    helper_a(x);
}

void compute(double x, int idx) {
    helper_b(x);
}

int main() {
    int i;

    for (i = 0; i < N; i++) {
        data[i] = (double)i / (double)N;
    }

    for (i = 0; i < N; i++) {
        compute(data[i], i);
    }

    return global_counter;
}
