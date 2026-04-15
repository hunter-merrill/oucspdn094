#include <cuda_runtime_api.h>
#include <curand_kernel.h>
#include <driver_types.h>
#include <curand.h>
#include <unistd.h>
#include <cstdio>
#include <cuda.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>

#include "kernel.cu"

#define BILLION  1000000000.0
#define MAX_LINE_LENGTH 25000

#define BLOCK_SIZE 1024
#define BLUR_SIZE 2

void err_check(cudaError_t ret, char* msg, int exit_code);

int main (int argc, char *argv[])
{
    // Check console errors
    if( argc != 6)
    {
        printf("USE LIKE THIS: convolution_serial n_row n_col mat_input.csv mat_output.csv time.csv\n");
        return EXIT_FAILURE;
    }

    // Get dims
    int n_row = strtol(argv[1], NULL, 10);
    int n_col = strtol(argv[2], NULL, 10);
    
    dim3 dimBlock(32, 32, 1);
    dim3 dimGrid(
        ceil((float)n_col / dimBlock.x), 
        ceil((float)n_row / dimBlock.y), 
        1);

    // Get files to read/write 
    FILE* inputFile1 = fopen(argv[3], "r");
    if (inputFile1 == NULL){
        printf("Could not open file %s",argv[2]);
        return EXIT_FAILURE;
    }
    FILE* outputFile = fopen(argv[4], "w");
    FILE* timeFile  = fopen(argv[5], "w");

    // Matrices to use
    int* inputMatrix_h  = (int*) malloc(n_row * n_col * sizeof(int));
    int* outputMatrix_h = (int*) malloc(n_row * n_col * sizeof(int));
    int filterMatrix_h[25] = {
        1, 0, 0, 0, 1,
        0, 1, 0, 1, 0,
        0, 0, 1, 0, 0,
        0, 1, 0, 1, 0,
        1, 0, 0, 0, 1
    };

    // read the data from the file
    int row_count = 0;
    char line[MAX_LINE_LENGTH] = {0};
    while (fgets(line, MAX_LINE_LENGTH, inputFile1)) {
        if (line[strlen(line) - 1] != '\n') printf("\n");
        char *token;
        const char s[2] = ",";
        token = strtok(line, s);
        int i_col = 0;
        while (token != NULL) {
            inputMatrix_h[row_count*n_col + i_col] = strtol(token, NULL,10 );
            i_col++;
            token = strtok (NULL, s);
        }
        row_count++;
    }

    fclose(inputFile1); 


    // --------------------------------------------------------------------------- //
    // ------ Algorithm Start ---------------------------------------------------- //

    struct timespec start, time1, time2, time3;    
    clock_gettime(CLOCK_REALTIME, &start);

    cudaError_t cuda_ret;

    int* inputMatrix_d;
    cuda_ret = cudaMalloc((void**)&inputMatrix_d, n_row * n_col * sizeof(int));
    err_check(cuda_ret, (char*)"Unable to allocate input matrix to device memory!", 1);
    cuda_ret = cudaMemcpy(inputMatrix_d, inputMatrix_h, n_row * n_col * sizeof(int), cudaMemcpyHostToDevice);
    err_check(cuda_ret, (char*)"Unable to copy input to device memory!", 2);

    int* filterMatrix_d;
    cuda_ret = cudaMalloc((void**)&filterMatrix_d, 25 * sizeof(int));
    err_check(cuda_ret, (char*)"Unable to allocate filter matrix to device memory!", 3);
    cuda_ret = cudaMemcpy(filterMatrix_d, filterMatrix_h, 25 * sizeof(int), cudaMemcpyHostToDevice);
    err_check(cuda_ret, (char*)"Unable to copy filter to device memory!", 4);

    int* outputMatrix_d;
    cuda_ret = cudaMalloc((void**)&outputMatrix_d, n_row * n_col * sizeof(int));
    err_check(cuda_ret, (char*)"Unable to allocate output matrix to device memory!", 5);

    clock_gettime(CLOCK_REALTIME, &time1);

    kernel <<< dimGrid, dimBlock >>> (
        inputMatrix_d,
        outputMatrix_d,
        filterMatrix_d,
        n_row,
        n_col
    );
    cuda_ret = cudaDeviceSynchronize();                                                                            // Barrier 1
    err_check(cuda_ret, (char*)"Unable to launch kernel!", 6);

    clock_gettime(CLOCK_REALTIME, &time2);

    cuda_ret = cudaMemcpy(outputMatrix_h, outputMatrix_d, n_row * n_col * sizeof(int), cudaMemcpyDeviceToHost);    // Barrier 2
    err_check(cuda_ret, (char*)"Unable to read output from device memory!", 7);

    clock_gettime(CLOCK_REALTIME, &time3);

    double time_spent_1 = (time1.tv_sec - start.tv_sec) +
                        (time1.tv_nsec - start.tv_nsec) / BILLION;
    double time_spent_2 = (time2.tv_sec - time1.tv_sec) +
                        (time2.tv_nsec - time1.tv_nsec) / BILLION;
    double time_spent_3 = (time3.tv_sec - time2.tv_sec) +
                        (time3.tv_nsec - time2.tv_nsec) / BILLION;

    // --------------------------------------------------------------------------- //
    // ------ Algorithm End ------------------------------------------------------ //


	// Save output matrix as csv file
    for (int i = 0; i<n_row; i++)
    {
        for (int j = 0; j<n_col; j++)
        {
            fprintf(outputFile, "%d", outputMatrix_h[i*n_col +j]);
            if (j != n_col -1)
                fprintf(outputFile, ",");
            else if ( i < n_row-1)
                fprintf(outputFile, "\n");
        }
    }

    // Print time
    fprintf(timeFile, "%f\n%f\n%f\n", time_spent_1, time_spent_2, time_spent_3);

    // Cleanup
    fclose (outputFile);
    fclose (timeFile);

    free(inputMatrix_h);
    free(outputMatrix_h);

    cudaFree(inputMatrix_d);
    cudaFree(outputMatrix_d);
    cudaFree(filterMatrix_d);

    return 0;
}



/* Error Check ----------------- //
*   Exits if there is a CUDA error.
*/
void err_check(cudaError_t ret, char* msg, int exit_code) {
    if (ret != cudaSuccess)
        fprintf(stderr, "%s \"%s\".\n", msg, cudaGetErrorString(ret)),
        exit(exit_code);
} // End Error Check ----------- //
