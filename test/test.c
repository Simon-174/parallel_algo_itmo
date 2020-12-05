#include <stdio.h>
#include <omp.h>

void main()
{
    #pragma omp parallel
    {
    int id = omp_get_thread_num();
    printf("hello (%d)\n", id);
    printf("world (%d)\n", id);
}
}
