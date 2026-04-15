#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include "mpi.h"

#define MAXLINE 25
#define DEBUG   0

// to read in file
float* read_input(FILE* inputFile, int n_items);
float cmpfloat(const void* a, const void* b);
float* merge(float* A, float* B, int n_items);

/* Main Program -------------- */
int main (int argc, char *argv[])
{
    if( argc != 5)
    {
        printf("USE LIKE THIS: merge_sort_MPI n_items input.csv output.csv time.csv\n");
        return EXIT_FAILURE;
    }

    // input file and size
    int  n_items = strtol(argv[1], NULL, 10);

    // Start MPI
    int my_rank, comm_size;
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

    // arrays to use
    // TODO: initialize your arrays here
    float *arr, *local_arr;

    int local_n = (int) n_items / comm_size;
    local_arr = malloc(local_n*sizeof(float));

    if (my_rank == 0) {
        FILE* inputFile = fopen(argv[2], "r");
        arr = read_input(inputFile, n_items);
        fclose(inputFile);
    }

    // get start time
    double local_start, local_finish, local_elapsed, elapsed;
    MPI_Barrier(MPI_COMM_WORLD);
    local_start = MPI_Wtime();



    // TODO: implement solution here
    MPI_Scatter(arr, local_n, MPI_FLOAT, local_arr, local_n, MPI_FLOAT, 0, MPI_COMM_WORLD);
    qsort(local_arr, local_n, sizeof(float), cmpfloat);

    int step = 1;
    while (step < comm_size) {

        int step_n = local_n * step;

        if (my_rank % (2*step) == 0) {
            
            float *recv_arr, *sorted; 
            
            recv_arr = (float*)malloc(step_n*sizeof(float));
            MPI_Recv(recv_arr, step_n, MPI_FLOAT, my_rank + step, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            sorted = merge(local_arr, recv_arr, step_n);

            free(recv_arr);
            free(local_arr);

            local_arr = sorted;

        }
        else {
            MPI_Send(local_arr, step_n, MPI_FLOAT, my_rank - step, 0, MPI_COMM_WORLD);
            free(local_arr);
            break;
        }

        step *= 2;
    }
    


    // get elapsed time
    local_finish  = MPI_Wtime();
    local_elapsed = local_finish-local_start;

    // send time to main process
    MPI_Reduce(
        &local_elapsed, 
        &elapsed, 
        1, 
        MPI_DOUBLE,
        MPI_MAX, 
        0, 
        MPI_COMM_WORLD
    );

    // Write output (Step 5)
    if (my_rank == 0) {
        FILE* outputFile = fopen(argv[3], "w");
        FILE* timeFile = fopen(argv[4], "w");

        // Write arr
        for (int i = 0; i < n_items; i++) {
            fprintf(outputFile, "%f\n", local_arr[i]);
        }

        // Write time
        fprintf(timeFile, "%.20lf", elapsed);

        fclose(outputFile);
        fclose(timeFile);

        free(arr);
        free(local_arr);
    }


    MPI_Finalize();

    if(DEBUG) printf("Finished!\n");
    return 0;
} // End Main //



/* Read Input -------------------- */
float* read_input(FILE* inputFile, int n_items) {
    float* arr = (float*)malloc(n_items * sizeof(float));
    char line[MAXLINE] = {0};
    int i = 0;
    char* ptr;
    while (fgets(line, MAXLINE, inputFile)) {
        sscanf(line, "%f", &(arr[i]));
        ++i;
    }
    return arr;
} // Read Input //



/* Cmp Int ----------------------------- */
// use this for qsort
// source: https://stackoverflow.com/questions/3886446/problem-trying-to-use-the-c-qsort-function
float cmpfloat(const void* a, const void* b) {
    float fa = *(const float*)a;
    float fb = *(const float*)b;
    return (fa > fb) - (fa < fb);
} // Cmp Int //



float* merge(float* A, float* B, int n_items) {
    
    float* C = (float*)malloc(2*n_items * sizeof(float));
    int idx1, idx2 = 0;
    float curr1, curr2;

    for (int i = 0; i < 2*n_items; i++) {

        curr1 = A[idx1];
        curr2 = B[idx2];

        if (curr1 <= curr2) {
            C[i] = curr1;
            idx1++;
        }
        else {
            C[i] = curr2;
            idx2++;
        }

        // End of one array reached
        if (idx1 == n_items || idx2 == n_items) {
            
            int j = 0;
            
            while (idx1 < n_items) {
                curr1 = A[idx1];
                C[i + j] = curr1;
                idx1++; j++;
            }

            while (idx2 < n_items) {
                curr2 = B[idx2];
                C[i + j] = curr2;
                idx2++; j++;
            }
        }

    }

    return C;

}