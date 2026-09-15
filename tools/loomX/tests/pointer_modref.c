// Test: pointer parameter write detection
// read_array only reads through its pointer parameter -> safe.
// write_array writes through its pointer parameter -> unsafe.
// The loop calling write_array should stay sequential.

#define N 100000

double data[N];
double result[N];

void read_array(double* arr, int idx) {
    double x = arr[idx];  // read through pointer
    (void)x;
}

void write_array(double* arr, int idx) {
    arr[idx] = (double)idx;  // write through pointer
}

int main() {
    int i;

    for (i = 0; i < N; i++) {
        data[i] = (double)i / (double)N;
    }

    for (i = 0; i < N; i++) {
        read_array(data, i);  // safe
    }

    for (i = 0; i < N; i++) {
        write_array(result, i);  // unsafe
    }

    return 0;
}
