#include <stdio.h>
#include <omp.h>
#include <math.h>
#define thrdsCount 6

double function(double x) {
    double sine = sin(1 / (x * x));
    double result = 1 / (x * x) * pow(sine, 2);
    return result;
}

double integral_true(double a, double b) {
    double result = 0.25 * (2 * (b - a) / (a * b) + sin(2 / b) - sin(2/a));
    return result;
}

double integral_sequential(double a, double b, long n_points) {
    double x, internal_sum = 0;
    double increment = (b - a) /(double) n_points;
    int i;
    for (i = 1; i < n_points - 2; i++) {
        x = a + increment * i;
        internal_sum += function(x);
    }
    internal_sum += function(a) / 2 + function(b) / 2;
    return internal_sum * increment;
}

double integral_critical(double a, double b, long n_points) {
    double x, internal_sum = 0.0;
    double increment = (b - a) / (double) n_points;
    int i;
    #pragma omp parallel for num_threads(thrdsCount) private(x)
        for (i = 1; i < n_points - 2; i++) {
            x = a + increment * i;
            #pragma omp critical
            {
                internal_sum += function(x);
            }
        }
    internal_sum += function(a) / 2 + function(b) / 2;
    return internal_sum * increment;
}

double integral_atomic(double a, double b, long n_points) {
    double x, internal_sum = 0.0;
    double increment = (b - a) / (double) n_points;
    int i;
    #pragma omp parallel for num_threads(thrdsCount) private(x)
        for (i = 1; i < n_points - 2; i++) {
            x = a + increment * i;
            #pragma omp atomic
            internal_sum += function(x);
        }
    internal_sum += function(a) / 2 + function(b) / 2;
    return internal_sum * increment;
}

double integral_lock(double a, double b, long n_points) {
    double x, internal_sum = 0.0;
    double increment = (b - a) / (double) n_points;
    int i;
    omp_lock_t lock;
    omp_init_lock(&lock);
    #pragma omp parallel for num_threads(thrdsCount) private(x)
        for (i = 1; i < n_points - 2; i++) {
            x = a + increment * i;
            omp_set_lock (&lock);
            internal_sum += function(x);
            omp_unset_lock (&lock);
        }
    omp_destroy_lock (&lock);
    internal_sum += function(a) / 2 + function(b) / 2;
    return internal_sum * increment;
}

double integral_reduction(double a, double b, long n_points) {
    double x, internal_sum = 0.0;
    double increment = (b - a) / (double) n_points;
    int i;
    #pragma omp parallel for reduction(+:internal_sum) num_threads(thrdsCount) private(x)
        for (i = 1; i < n_points - 2; i++) {
            x = a + increment * i;
            internal_sum += function(x);
        }
    internal_sum += function(a) / 2 + function(b) / 2;
    return internal_sum * increment;
}

int main() {
    double t1, t2, dt, dt_sum = 0.0;
    double a, b, integral = 0.0;
    int n_points = 10000000;

    t1 = omp_get_wtime();
    integral = integral_true(0.00001, 0.0001);
    t2 = omp_get_wtime();
    dt = t2 - t1;
    printf("True: time %f, value %f\n", dt, integral);

    t1 = omp_get_wtime();
    integral = integral_sequential(0.00001, 0.0001, n_points);
    t2 = omp_get_wtime();
    dt = t2 - t1;
    printf("Sequential: time %f, value %f\n", dt, integral);

    t1 = omp_get_wtime();
    integral = integral_critical(0.00001, 0.0001, n_points);
    t2 = omp_get_wtime();
    dt = t2 - t1;
    printf("Critical: time %f, value %f\n", dt, integral);

    t1 = omp_get_wtime();
    integral = integral_atomic(0.00001, 0.0001, n_points);
    t2 = omp_get_wtime();
    dt = t2 - t1;
    printf("Atomic: time %f, value %f\n", dt, integral);

    t1 = omp_get_wtime();
    integral = integral_lock(0.00001, 0.0001, n_points);
    t2 = omp_get_wtime();
    dt = t2 - t1;
    printf("Lock: time %f, value %f\n", dt, integral);

    t1 = omp_get_wtime();
    integral = integral_reduction(0.00001, 0.0001, n_points);
    t2 = omp_get_wtime();
    dt = t2 - t1;
    printf("Reduction: time %f, value %f\n", dt, integral);

    return 0;
}