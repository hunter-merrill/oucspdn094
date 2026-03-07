#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <omp.h>

#define MAX_LINE_LENGTH 1000000
#define DEBUG 1

/* ------------ Project 2 - Problem 4 - Decryption ------------
*/ // ------------------------------------------------------ //

int decrypt(int x, int k) {
    return (x + k) % 256;
}

int main(int argc, char* argv[])
{
    // Catch console errors
    //  Make sure you include the # of threads and your output time file.
    if (argc != 5) {
        printf("USE LIKE THIS: decrypt_parllel input_text.txt key.txt time.txt num_threads\n");
        return EXIT_FAILURE;
    }

    // Open the input, encrypted text file
    FILE* inputFile = fopen(argv[1], "r");

    // Open the output, text file containing key
    FILE* outputFile = fopen(argv[2], "w");

    // File to store timing
    FILE* timeFile = fopen(argv[3], "w");

    // Get num threads
    int thread_count = strtol(argv[4], NULL, 10);

    // Allocate and open a buffer to read in the input
    fseek(inputFile, 0L, SEEK_END);
    long lSize = ftell(inputFile);
    rewind(inputFile);
    unsigned char* buffer = calloc(1, lSize + 1);
    if (!buffer)
        fclose(inputFile),
        fclose(outputFile),
        fclose(timeFile),
        free(buffer),
        fputs("Memory alloc for inputFile1 failed!\n", stderr),
        exit(1);

    // Read the input into the buffer
    if (1 != fread(buffer, lSize, 1, inputFile))
        fclose(inputFile),
        fclose(outputFile),
        fclose(timeFile),
        free(buffer),
        fputs("Failed reading into the input buffer!\n", stderr),
        exit(2);

    // Allocate a buffer for the encrypted data
    unsigned char* encrypted_buffer = calloc(1, lSize + 1);
    if (!encrypted_buffer)
        fclose(inputFile),
        fclose(outputFile),
        fclose(timeFile),
        free(encrypted_buffer),
        free(buffer),
        fputs("Memory alloc for the encrypted buffer failed!\n", stderr),
        exit(3);



    // Rabin-Karp based off of https://www.francofernando.com/blog/algorithms/2021-05-16-rolling-hash/

    /* After getting very far into implementation, I realized that maybe an optimized string
     * search algorithm is not necessary for brute force, but it's too late to go back so we ball
    */

    // Hashing constants
    const int b = 257, M = 1000000009;
    const int bPow = (b * b) % M;

    const char The[3] = "The";
    const char the[3] = "the";

    int hash_The = 0, hash_the = 0;

    // Calculate rolled hash values for 'The' and 'the'
    for (int i = 0; i < 3; i++) {
        hash_The = (hash_The * b + The[i]) % M;
        hash_the = (hash_the * b + the[i]) % M;
    }

    int global_max_occurrences = 0;
    int global_key = -123;

    // Record the start time
    double start = omp_get_wtime();

#pragma omp parallel num_threads(thread_count)
    {

        int local_max_occurrences = 0;
        int local_key = -123;

#pragma omp for
        for (int k = 0; k < 256; k++) { // Each thread gets assigned 256/thread_count keys to check

            int occurrences = 0;

            // calculate the hash value of the first substring of length 3
            int ht = 0;
            for (int j = 0; j < 3; j++) ht = (ht * b + decrypt(buffer[j], k)) % M;

            // check if the first substring matches the pattern
            if (ht == hash_The || ht == hash_the) occurrences++;

            // start the rolling hash 
            for (int i = 0; i < lSize - 3; i++) {

                long long old_char = decrypt(buffer[i], k);
                long long new_char = decrypt(buffer[i + 3], k);

                ht = (ht - (old_char * bPow)) % M;
                ht = (ht * b) % M;
                ht = (ht + new_char) % M;

                if (ht == hash_The || ht == hash_the) occurrences++;
            }

            if (occurrences > local_max_occurrences) {
                local_max_occurrences = occurrences;
                local_key = k;
            }
        }

#pragma omp critical
        {
            if (local_max_occurrences > global_max_occurrences) {
                global_max_occurrences = local_max_occurrences;
                global_key = local_key;
            }
        }
    }
    // Print to the output file
    fprintf(outputFile, "%i", global_key);

#ifdef DEBUG
    printf("BRUTE-FORCED KEY: %d\n", global_key);
#endif

    // Record the finish time
    double end = omp_get_wtime();

    // Time calculation (in seconds)
    double time_passed = end - start;

    // Save time to file
    fprintf(timeFile, "%f", time_passed);

    // Cleanup
    fclose(inputFile);
    fclose(outputFile);
    fclose(timeFile);
    free(encrypted_buffer);
    free(buffer);

    return 0;
} // End main //