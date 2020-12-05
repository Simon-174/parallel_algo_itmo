#include <stdio.h>
#include <omp.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <iomanip>

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

double integral_critical(double a, double b, long n_points, int thrdsCount) {
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

double integral_atomic(double a, double b, long n_points, int thrdsCount) {
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

double integral_lock(double a, double b, long n_points, int thrdsCount) {
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

double integral_reduction(double a, double b, long n_points, int thrdsCount) {
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

void measure_time(char fname) {
    double t1, t2, dt_sum = 0.0, integral;
    double a[] = {0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0};
    double b[] = {0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0};
    long n_points[] = {100000, 1000000, 10000000};
    int n_threads[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20};

    for (const auto& n_point : n_points) {
        for (const auto& n_thread : n_threads) {
            for (int i = 0; i < 8; i++) {
                dt_sum = 0.0;
                for (int n_rep = 0; n_rep < 5; n_rep++) {
                    if (fname == 's') {
                        t1 = omp_get_wtime();
                        integral = integral_sequential(a[i], b[i], n_point);
                        t2 = omp_get_wtime();
                    }
                    else if (fname == 'c') {
                        t1 = omp_get_wtime();
                        integral = integral_critical(a[i], b[i], n_point, n_thread);
                        t2 = omp_get_wtime();
                    }
                    else if (fname == 'a') {
                        t1 = omp_get_wtime();
                        integral = integral_atomic(a[i], b[i], n_point, n_thread);
                        t2 = omp_get_wtime();
                    }
                    else if (fname == 'l') {
                        t1 = omp_get_wtime();
                        integral = integral_lock(a[i], b[i], n_point, n_thread);
                        t2 = omp_get_wtime();
                    }
                    else if (fname == 'r') {
                        t1 = omp_get_wtime();
                        integral = integral_reduction(a[i], b[i], n_point, n_thread);
                        t2 = omp_get_wtime();
                    }
                    dt_sum += t2 - t1;
                }
                dt_sum /= 5;
                std::ofstream outfile;
                outfile.open("task1.csv", outfile.app);
                outfile << fname;
                outfile << ",";
                outfile << n_point;
                outfile << ",";
                outfile << n_thread;
                outfile << ",";
                outfile << std::fixed << std::setprecision(8) << a[i];
                outfile << ",";
                outfile << std::fixed << std::setprecision(8) << b[i];
                outfile << ",";
                outfile << std::fixed << std::setprecision(8) << dt_sum;
                outfile << "\n";
                outfile.close();
                printf("%c, n_points %ld, n_threads %d, a %lf, b %lf, time %lf\n", fname, n_point, n_thread, a[i], b[i], dt_sum);
            }
        }
    }
}

int main() {

    std::ofstream outfile;
    outfile.open("task1.csv");
    outfile << "implementation,N_points,N_threads,A,B,time\n";
    outfile.close();

    measure_time('s');
    measure_time('a');
    measure_time('c');
    measure_time('l');
    measure_time('r');

    return 0;
}