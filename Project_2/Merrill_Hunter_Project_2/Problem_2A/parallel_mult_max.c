#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#define DEBUG 0

void fill_mat(long int* mat, FILE* infile, int n_row, int n_col) {
    
    int max_line_len = n_col * 20; 
    char* buffer = malloc(max_line_len);
    if (buffer == NULL) return;

    for (int row = 0; row < n_row; row++) {
        fgets(buffer, max_line_len, infile);
        char* token = strtok(buffer, ",");
        for (int column = 0; column < n_col; column++) {
            long int n = atol(token);
            mat[row * n_col + column] = n;
            token = strtok(NULL, ",");
        }
    }
    free(buffer);
}

int main(int argc, char* argv[])
{
    // Catch console errors
    if (argc != 10)
    {
        printf("USE LIKE THIS: parallel_mult_mat_mat file_A.csv n_row_A n_col_A file_B.csv n_row_B n_col_B result_matrix.csv time.csv num_threads \n");
        return EXIT_FAILURE;
    }

    // Get the input files
    FILE* inputMatrix1 = fopen(argv[1], "r");
    FILE* inputMatrix2 = fopen(argv[4], "r");

    char* p1;
    char* p2;

    // Get matrix 1's dims
    int n_row1 = strtol(argv[2], &p1, 10);
    int n_col1 = strtol(argv[3], &p2, 10);

    // Get matrix 2's dims
    int n_row2 = strtol(argv[5], &p1, 10);
    int n_col2 = strtol(argv[6], &p2, 10);

    // Get num threads
    int thread_count = strtol(argv[9], NULL, 10);

    // Get output files
    FILE* outputFile = fopen(argv[7], "w");
    FILE* outputTime = fopen(argv[8], "w");

    // TODO: malloc the two input matrices and the output matrix
    // Please use long int as the variable type
    int n_elem1 = n_row1 * n_col1 * sizeof(long int);
    int n_elem2 = n_row2 * n_col2 * sizeof(long int);

    long int* mat1 = (long int*)malloc(n_elem1);
    long int* mat2 = (long int*)malloc(n_elem2);

    // TODO: Parse the input csv files and fill in the input matrices
    fill_mat(mat1, inputMatrix1, n_row1, n_col1);
    fill_mat(mat2, inputMatrix2, n_row2, n_col2);

    // We are interesting in timing the matrix-matrix multiplication only
    // Record the start time
    double start = omp_get_wtime();

    // TODO: Parallelize the matrix-matrix multiplication
    long int maximum = 0;

#pragma omp parallel for reduction(max:maximum) num_threads(thread_count)
    for (int col_mat2 = 0; col_mat2 < n_col2; col_mat2++) { // for each column vector in matrix B
        for (int row_mat1 = 0; row_mat1 < n_row1; row_mat1++) { // for each row vector in matrix A do
            
            long int x = 0; // compute x = the dot product of the two vectors
            
            for (int col_mat1 = 0; col_mat1 < n_col1; col_mat1++) {
                long int idx1 = (row_mat1 * n_col1) + col_mat1;
                long int idx2 = (col_mat1 * n_col2) + col_mat2;

                x += mat1[idx1] * mat2[idx2]; 
            }
            if (x > maximum) maximum = x;
        }
    }


    // Record the finish time
    double end = omp_get_wtime();

    // Time calculation (in seconds)
    double time_passed = end - start;

    // Save time to file
    fprintf(outputTime, "%f", time_passed);

    // TODO: save the output to the output csv file
    fprintf(outputFile, "%ld", maximum);

    // Cleanup
    fclose(inputMatrix1);
    fclose(inputMatrix2);
    fclose(outputFile);
    fclose(outputTime);
    // Remember to free your buffers!
    free(mat1);
    free(mat2);

    return 0;
}
