#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <time.h>
#include "mpi.h"

#define MAXLINE 25
#define DEBUG   0

int isInCircle(double x, double y);

/* Main Program -------------- */
int main (int argc, char *argv[])
{
    if( argc != 3)
    {
        printf("USE LIKE THIS: pi_MPI result.csv time.csv\n");
        return EXIT_FAILURE;
    }

    // Start MPI
    int my_rank, comm_size;
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

    int n = (int) pow(2.0, 16.0);
    int local_n = (int) n / comm_size;
    int count, local_count = 0;

    // get start time
    double local_start, local_finish, local_elapsed, elapsed;
    MPI_Barrier(MPI_COMM_WORLD);
    local_start = MPI_Wtime();



    srand(my_rank);
    for (int i = 0; i < local_n; i++) {
        double x = (double)rand() / (double)RAND_MAX;
        double y = (double)rand() / (double)RAND_MAX;
        local_count += isInCircle(x, y);
    }

    MPI_Reduce(&local_count, &count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);



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
        FILE* outputFile = fopen(argv[1], "w");
        FILE* timeFile = fopen(argv[2], "w");

        double pi = 4.0 * (double)count / (double)n;

        fprintf(outputFile, "%f\n", pi);
        fprintf(timeFile, "%.20lf", elapsed);

        fclose(outputFile);
        fclose(timeFile);
    }

    MPI_Finalize();

    if(DEBUG) printf("Finished!\n");
    return 0;
} // End Main //

int isInCircle(double x, double y) {
    if (pow(x, 2) + pow(y, 2) > 1) {
        return 0;
    }
    else {
        return 1;
    }
}