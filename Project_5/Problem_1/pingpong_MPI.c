#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include "mpi.h"

int main(int argc, char* argv[]) {
    // Catch console errors
    if (argc != 3)
    {
        printf("USE LIKE THIS: pingpong_MPI n_items time_prob1_MPI.csv\n");
        return EXIT_FAILURE;
    }

    /* Read in command line items */
    int n_items = strtol(argv[1], NULL, 10);
    FILE* outputFile = fopen(argv[2], "w");

    /* Start up MPI */
    
    // TODO: finish setting up MPI
    int my_rank, comm_sz;
    MPI_Comm comm;
    MPI_Init(&argc, &argv);
    comm = MPI_COMM_WORLD;
    MPI_Comm_size(comm, &comm_sz);
    MPI_Comm_rank(comm, &my_rank);

    int* ping_array = malloc(sizeof(int)*n_items);
    


    // Start time
    double starttime;
    starttime = MPI_Wtime();




    // TODO: Create your MPI program.
    if (my_rank == 0) {

        // Fill array with incremental values
        for (int i = 0; i < n_items; i++)
            ping_array[i] = i;

        // TODO: if myrank is 0
        for (int i = 0; i < 1000; i++) {
            MPI_Send(ping_array, n_items, MPI_INT, 1, i, comm);
            MPI_Recv(ping_array, n_items, MPI_INT, 1, i, comm, MPI_STATUS_IGNORE);
        }


        // End time
        double endtime = MPI_Wtime();
        // TODO: output
        double elapsed = endtime - starttime;
        printf("%d: %lf\n", n_items, elapsed);
        fprintf(outputFile, "%lf", elapsed);
    }
    else {

        // TODO: if my rank not 0
        for (int i = 0; i < 1000; i++) {
            MPI_Recv(ping_array, n_items, MPI_INT, 0, i, comm, MPI_STATUS_IGNORE);
            MPI_Send(ping_array, n_items, MPI_INT, 0, i, comm);
        }

    }

    free(ping_array);
    MPI_Finalize();

    return 0;
} /* main */

