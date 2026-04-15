#include <cuda_runtime_api.h>
#include <curand_kernel.h>
#include <driver_types.h>
#include <curand.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <cstdio>
#include <cuda.h>

#include "support.h"
#include "hash_kernel.cu"
#include "nonce_kernel.cu"
#include "reduction_kernel.cu"

// to activate debug statements
#define DEBUG 1

// program constants
#define BLOCK_SIZE 1024
#define SEED       123

// solution constants
#define MAX     123123123
#define TARGET  20

// functions used
unsigned int generate_hash(unsigned int nonce, unsigned int index, unsigned int* transactions, unsigned int n_transactions);
void read_file(char* file, unsigned int* transactions, unsigned int n_transactions);
void err_check(cudaError_t ret, char* msg, int exit_code);


/* Main ------------------ //
*   This is the main program.
*/
int main(int argc, char* argv[]) {

    // Catch console errors
    if (argc != 6) {
        printf("USE LIKE THIS: gpu_mining transactions.csv n_transactions trials out.csv time.csv\n");
        return EXIT_FAILURE;
    }


    // Output files
    FILE* output_file = fopen(argv[4], "w");
    FILE* time_file   = fopen(argv[5], "w");

    // Read in the transactions
    unsigned int n_transactions = strtoul(argv[2], NULL, 10);
    unsigned int* transactions = (unsigned int*)calloc(n_transactions, sizeof(unsigned int));
    read_file(argv[1], transactions, n_transactions);

    // get the number of trials
    unsigned int trials = strtoul(argv[3], NULL, 10);


    // -------- Start Mining ------------------------------------------------------- //
    // ----------------------------------------------------------------------------- //
    
    // Set timer and cuda error return
    Timer timer;
    startTime(&timer);
    cudaError_t cuda_ret;

    // To use with kernels
    int num_blocks = ceil((float)trials / (float)BLOCK_SIZE);
    dim3 dimGrid(num_blocks, 1, 1);
    dim3 dimBlock(BLOCK_SIZE, 1, 1);


    // ------ Step 1: generate the nonce values ------ //

    // Allocate the nonce device memory
    unsigned int* device_nonce_array;
    cuda_ret = cudaMalloc((void**)&device_nonce_array, trials * sizeof(unsigned int));
    err_check(cuda_ret, (char*)"Unable to allocate nonces to device memory!", 1);

    // Launch the nonce kernel
    nonce_kernel <<< dimGrid, dimBlock >>> (
        device_nonce_array, // put nonces into here
        trials,             // size of array
        MAX,                // to mod with
        SEED                // random seed
        );
    cuda_ret = cudaDeviceSynchronize();
    err_check(cuda_ret, (char*)"Unable to launch nonce kernel!", 2);

    // Get nonces from device memory
    unsigned int* nonce_array = (unsigned int*)calloc(trials, sizeof(unsigned int));
    cuda_ret = cudaMemcpy(nonce_array, device_nonce_array, trials * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    err_check(cuda_ret, (char*)"Unable to read nonce from device memory!", 3);


    // ------ Step 2: Generate the hash values ------ //

    // TODO Problem 1: perform this hash generation in the GPU
    // Hint: You need both nonces and transactions to compute a hash.

    // Allocate the transactions device memory
    unsigned int* device_transactions_array;
    cuda_ret = cudaMalloc((void**)&device_transactions_array, n_transactions * sizeof(unsigned int));
    err_check(cuda_ret, (char*)"Unable to allocate transactions to device memory!", 4);
    cuda_ret = cudaMemcpy(device_transactions_array, transactions, n_transactions * sizeof(unsigned int), cudaMemcpyHostToDevice);
    err_check(cuda_ret, (char*)"Unable to copy transactions to device memory!", 5);

    // Allocate the hash devicem emory
    unsigned int* device_hash_array;
    cuda_ret = cudaMalloc((void**)&device_hash_array, trials * sizeof(unsigned int));
    err_check(cuda_ret, (char*)"Unable to allocate hashes to device memory!", 6);

    // Launch the hash kernel
    hash_kernel <<< dimGrid, dimBlock >>> (
        device_hash_array,
        device_nonce_array,
        trials,
        device_transactions_array,
        n_transactions,
        MAX
    );
    cuda_ret = cudaDeviceSynchronize();
    err_check(cuda_ret, (char*)"Unable to launch hash kernel!", 7);

    unsigned int* hash_array = (unsigned int*)calloc(trials, sizeof(unsigned int));
    cuda_ret = cudaMemcpy(hash_array, device_hash_array, trials * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    err_check(cuda_ret, (char*)"Unable to copy hashes to system memory!", 8);

    // Free memory
    free(transactions);


    // ------ Step 3: Find the nonce with the minimum hash value ------ //

    // TODO Problem 2: find the minimum in the GPU by reduction

    // Calc size of output arrays (=num blocks)
    unsigned int reduced_size = ceil((float)trials / (2.0 * BLOCK_SIZE));
    dim3 dimGridReduced(reduced_size, 1, 1);

    // Init arrays in system memory to store results
    unsigned int* reduced_hash_array = (unsigned int*)calloc(reduced_size, sizeof(unsigned int));
    unsigned int* reduced_nonce_array = (unsigned int*)calloc(reduced_size, sizeof(unsigned int));

    unsigned int* device_hash_out;
    cuda_ret = cudaMalloc((void**)&device_hash_out, reduced_size * sizeof(unsigned int));
    err_check(cuda_ret, (char*)"Unable to allocate hash reduction to device memory!", 9);

    unsigned int* device_nonce_out;
    cuda_ret = cudaMalloc((void**)&device_nonce_out, reduced_size * sizeof(unsigned int));
    err_check(cuda_ret, (char*)"Unable to allocate nonce reduction to device memory!", 10);

    reduction_kernel <<< dimGridReduced, dimBlock >>> (
        device_hash_out,
        device_nonce_out,
        device_hash_array,
        device_nonce_array,
        trials
    );
    cuda_ret = cudaDeviceSynchronize();
    err_check(cuda_ret, (char*)"Unable to launch reduction kernel!", 11);

    unsigned int* hash_out = (unsigned int*)calloc(reduced_size, sizeof(unsigned int));
    cuda_ret = cudaMemcpy(hash_out, device_hash_out, reduced_size * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    err_check(cuda_ret, (char*)"Unable to copy minimum hashes to system memory!", 12);

    unsigned int* nonce_out = (unsigned int*)calloc(reduced_size, sizeof(unsigned int));
    cuda_ret = cudaMemcpy(nonce_out, device_nonce_out, reduced_size * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    err_check(cuda_ret, (char*)"Unable to copy minimum nonces to system memory!", 13);

    unsigned int min_hash  = MAX;
    unsigned int min_nonce = MAX;
    for (int i = 0; i < reduced_size; i++) {
        if (hash_out[i] < min_hash) {
            min_hash  = hash_out[i];
            min_nonce = nonce_out[i];
        }
    }

    // Free memory
    free(nonce_array);
    free(nonce_out);
    free(hash_array);
    free(hash_out);

    cudaFree(device_nonce_array);
    cudaFree(device_transactions_array);
    cudaFree(device_hash_array);
    cudaFree(device_hash_out);
    cudaFree(device_nonce_out);

    stopTime(&timer);
    // ----------------------------------------------------------------------------- //
    // -------- Finish Mining ------------------------------------------------------ //


    // Get if suceeded
    char* res = (char*)malloc(8 * sizeof(char));
    if (min_hash < TARGET)  res = (char*)"Success!";
    else                    res = (char*)"Failure.";

    // Show results in console
    if (DEBUG) 
        printf("%s\n   Min hash:  %u\n   Min nonce: %u\n   %f seconds\n",
            res,
            min_hash,
            min_nonce,
            elapsedTime(timer)
        );

    // Print results
    fprintf(output_file, "%s\n%u\n%u\n", res, min_hash, min_nonce);
    fprintf(time_file, "%f\n", elapsedTime(timer));

    // Cleanup
    fclose(time_file);
    fclose(output_file);

    return 0;
} // End Main -------------------------------------------- //



/* Generate Hash ----------------------------------------- //
*   Generates a hash value from a nonce and transaction list.
*/
unsigned int generate_hash(unsigned int nonce, unsigned int index, unsigned int* transactions, unsigned int n_transactions) {

    unsigned int hash = (nonce + transactions[0] * (index + 1)) % MAX;
    for (int j = 1; j < n_transactions; j++) {
        hash = (hash + transactions[j] * (index + 1)) % MAX;
    }
    return hash;

} // End Generate Hash ---------- //



/* Read File -------------------- //
*   Reads in a file of transactions. 
*/
void read_file(char* file, unsigned int* transactions, unsigned int n_transactions) {

    // open file
    FILE* trans_file = fopen(file, "r");
    if (trans_file == NULL)
        fprintf(stderr, "ERROR: could not read the transaction file.\n"),
        exit(-1);

    // read items
    char line[100] = { 0 };
    for (int i = 0; i < n_transactions && fgets(line, 100, trans_file); ++i) {
        char* p;
        transactions[i] = strtof(line, &p);
    }

    fclose(trans_file);

} // End Read File ------------- //



/* Error Check ----------------- //
*   Exits if there is a CUDA error.
*/
void err_check(cudaError_t ret, char* msg, int exit_code) {
    if (ret != cudaSuccess)
        fprintf(stderr, "%s \"%s\".\n", msg, cudaGetErrorString(ret)),
        exit(exit_code);
} // End Error Check ----------- //
