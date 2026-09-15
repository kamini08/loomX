// Side-effect test: loop calls a function that modifies a global variable
// Should be rejected for parallelization

static int global_counter = 0;

double side_effect_func(double x) {
    global_counter = global_counter + 1;  // writes to global
    return x * 2.0;
}

int main() {
    int N = 10000;
    double data[10000];
    double result[10000];
    int i;

    for (i = 0; i < N; i++) {
        data[i] = (double)i / (double)N;
    }

    for (i = 0; i < N; i++) {
        result[i] = side_effect_func(data[i]);
    }

    return global_counter;
}
