// #include <mpi.h>
#include <stdio.h>


double mult_col_by_row(double a_row[], double b_col[]) {
    double result = 0.0;
    int size = sizeof(a_row) / sizeof(a_row[0]);
    for (int i = 0; i < size; i++) {
        result += a_row[i] * b_col[i];
    }
    return result;
}

void mm_serial(double A[], double B[], double result[], int a_rows, int a_cols, int b_cols) {
    for (int row = 0; row < a_rows; row++) {
        for (int col = 0; col < b_cols; col++) {
            for (int elem = 0; elem < a_cols; elem++) {
                result[row*b_cols+col] += A[row*a_cols + elem] * B[elem*b_cols+col];
            }
        }
    }
}

int main() {
    double a[4] = {1, 1, 1, 1}, b[4] = {-1, 0, 0, -1}, c[4] = {0};
    mm_serial(a, b, c, 2, 2, 2);
    for (const auto &c_i : c) {
        printf("%lf\n", c_i);
    }

    return 0;
}