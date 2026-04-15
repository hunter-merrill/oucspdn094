#include <stdio.h>
#include <stdlib.h> // for strtol
#include <string.h>
#include <time.h>
#include "mpi.h"

#define MAXCHAR 25
#define BILLION  1000000000.0

void Check_for_error(int local_ok, char fname[], char message[], 
      MPI_Comm comm);
void Allocate_vectors(double** local_x_pp, double** local_y_pp,
      int local_n, int comm_sz, MPI_Comm comm);
void Read_vector(double local_a[], int local_n, int n, char filename[], 
      int my_rank, MPI_Comm comm);
void Gather_sum(double* local_product, int local_n, int n, 
      int my_rank, double* sum, int comm_sz, MPI_Comm comm);
void Parallel_dot_product(double local_x[], double local_y[], 
      double* local_product, int local_n, int my_rank);

// Template parallelization functions are based on 3.2.4 mpi_vector_add.c

int main (int argc, char *argv[])
{
    if( argc != 6)
    {
        printf("USE LIKE THIS: dotprod_serial vector_size vec_1.csv vec_2.csv result.csv time.csv\n");
        return EXIT_FAILURE;
    }

    // info about timing in C -------------------------------------
    // https://www.techiedelight.com/find-execution-time-c-program/
    struct timespec start, end;

    int n, local_n;
    int comm_sz, my_rank;
    double *local_x, *local_y;
    double local_product, final_product = 0.0;
    MPI_Comm comm;

    // MPI setup
    MPI_Init(&argc, &argv);
    comm = MPI_COMM_WORLD;
    MPI_Comm_size(comm, &comm_sz);
    MPI_Comm_rank(comm, &my_rank);

    // Size
    n = strtol(argv[1], NULL, 10);
    local_n = n / comm_sz;

    Allocate_vectors(&local_x, &local_y, local_n, comm_sz, comm);
    Read_vector(local_x, local_n, n, argv[2], my_rank, comm);
    Read_vector(local_y, local_n, n, argv[3], my_rank, comm);
    
    // Get start time
    clock_gettime(CLOCK_REALTIME, &start);

    Parallel_dot_product(local_x, local_y, &local_product, local_n, my_rank);
    Gather_sum(&local_product, local_n, n, my_rank, &final_product, comm_sz, comm);

    // Get finish time
    clock_gettime(CLOCK_REALTIME, &end);
    double time_spent = (end.tv_sec - start.tv_sec) +
                        (end.tv_nsec - start.tv_nsec) / BILLION;


    if (my_rank == 0) {
        FILE* outputFile = fopen(argv[4], "w");
        FILE* timeFile = fopen(argv[5], "w");
        
        // Print to output
        fprintf(outputFile, "%lf", final_product);
        fprintf(timeFile, "%.20f", time_spent);

        // Cleanup
        fclose (outputFile);
        fclose (timeFile);
    }

    free(local_x);
    free(local_y);

    MPI_Finalize();

    return 0;
}

/*-------------------------------------------------------------------
 * Function:  Check_for_error
 * Purpose:   Check whether any process has found an error.  If so,
 *            print message and terminate all processes.  Otherwise,
 *            continue execution.
 * In args:   local_ok:  1 if calling process has found an error, 0
 *               otherwise
 *            fname:     name of function calling Check_for_error
 *            message:   message to print if there's an error
 *            comm:      communicator containing processes calling
 *                       Check_for_error:  should be MPI_COMM_WORLD.
 *
 * Note:
 *    The communicator containing the processes calling Check_for_error
 *    should be MPI_COMM_WORLD.
 */
void Check_for_error(
      int       local_ok   /* in */, 
      char      fname[]    /* in */,
      char      message[]  /* in */, 
      MPI_Comm  comm       /* in */) {
   int ok;

   MPI_Allreduce(&local_ok, &ok, 1, MPI_INT, MPI_MIN, comm);
   if (ok == 0) {
      int my_rank;
      MPI_Comm_rank(comm, &my_rank);
      if (my_rank == 0) {
         fprintf(stderr, "Proc %d > In %s, %s\n", my_rank, fname, 
               message);
         fflush(stderr);
      }
      MPI_Finalize();
      exit(-1);
   }
}  /* Check_for_error */


/*-------------------------------------------------------------------
 * Function:  Allocate_vectors
 * Purpose:   Allocate storage for x, y, and products
 * In args:   local_n:  the size of the local vectors
 *            comm_sz:  number of processes
 *            comm:     the communicator containing the calling processes
 * Out args:  local_x_pp, local_y_pp:  pointers to memory
 *               blocks to be allocated for local vectors
 *
 * Errors:    One or more of the calls to malloc fails
 */
void Allocate_vectors(
      double**   local_x_pp         /* out */, 
      double**   local_y_pp         /* out */,
      int        local_n            /* in  */,
      int        comm_sz            /* in  */,
      MPI_Comm   comm               /* in  */) {
   int local_ok = 1;
   char* fname = "Allocate_vectors";

   *local_x_pp = malloc(local_n*sizeof(double));
   *local_y_pp = malloc(local_n*sizeof(double));

   if (*local_x_pp == NULL || *local_y_pp == NULL) local_ok = 0;
   Check_for_error(local_ok, fname, "Can't allocate local vector(s)", 
         comm);
}  /* Allocate_vectors */


/*-------------------------------------------------------------------
 * Function:   Read_vector
 * Purpose:    Read a vector from file on process 0 and distribute
 *             among the processes using a block distribution.
 * In args:    local_n:  size of local vectors
 *             n:        size of global vector
 *             vec_name: name of vector being read (e.g., "x")
 *             my_rank:  calling process' rank in comm
 *             comm:     communicator containing calling processes
 * Out arg:    local_a:  local vector read
 *
 * Errors:     if the malloc on process 0 for temporary storage
 *             fails the program terminates
 *
 * Note: 
 *    This function assumes a block distribution and the order
 *   of the vector evenly divisible by comm_sz.
 */
void Read_vector(
      double    local_a[]   /* out */, 
      int       local_n     /* in  */, 
      int       n           /* in  */,
      char      filename[]  /* in  */,
      int       my_rank     /* in  */, 
      MPI_Comm  comm        /* in  */) {

   double* a = NULL;
   int local_ok = 1;
   char* fname = "Read_vector";

   if (my_rank == 0) {

        // Store values of vector
        FILE* inputFile = fopen(filename, "r");
        if (inputFile == NULL) printf("Could not open file %s", filename);

        a = malloc(n*sizeof(double));
        if (a == NULL) local_ok = 0;
        Check_for_error(local_ok, fname, "Can't allocate temporary vector", 
            comm);

        int k = 0;
        char str[MAXCHAR];
        while (fgets(str, MAXCHAR, inputFile) != NULL)
        {
            sscanf( str, "%lf", &(a[k]) );
            k++;
        }
        fclose(inputFile);

        MPI_Scatter(a, local_n, MPI_DOUBLE, local_a, local_n, MPI_DOUBLE, 0,
            comm);
        free(a);

   } else {
      Check_for_error(local_ok, fname, "Can't allocate temporary vector", 
            comm);
      MPI_Scatter(a, local_n, MPI_DOUBLE, local_a, local_n, MPI_DOUBLE, 0,
         comm);
   }
}  /* Read_vector */  


/*-------------------------------------------------------------------
 * Function:  Gather_sum
 * Purpose:   Save a vector that has a block distribution to stdout
 * In args:   local_product:  local storage for vector to be saved
 *            local_n:  order of local vectors
 *            n:        order of global vector (local_n*comm_sz)
 *            title:    title to precede save
 *            comm:     communicator containing processes calling
 *                      Gather_sum
 *
 * Error:     if process 0 can't allocate temporary storage for
 *            the full vector, the program terminates.
 *
 * Note:
 *    Assumes order of vector is evenly divisible by the number of
 *    processes
 */
void Gather_sum(
    double*   local_product  /* in  */, 
    int       local_n    /* in  */, 
    int       n          /* in  */, 
    int       my_rank    /* in  */,
    double*   sum        /* out */,
    int       comm_sz    /* in  */,
    MPI_Comm  comm       /* in  */) {

    double* b = NULL;
    int i;
    int local_ok = 1;
    char* fname = "Gather_sum";

    if (my_rank == 0) {
        b = malloc(n*sizeof(double));
        if (b == NULL) local_ok = 0;
        Check_for_error(local_ok, fname, "Can't allocate temporary vector", 
                comm);
        MPI_Gather(local_product, 1, MPI_DOUBLE, b, 1, MPI_DOUBLE,
                0, comm);

        for (i = 0; i < comm_sz; i++)
            *sum += b[i];
            
        free(b);
    } else {
        Check_for_error(local_ok, fname, "Can't allocate temporary vector", 
                comm);
        MPI_Gather(local_product, 1, MPI_DOUBLE, b, 1, MPI_DOUBLE, 0,
            comm);
    }
}  /* Gather_sum */


/*-------------------------------------------------------------------
 * Function:  Parallel_dot_product
 * Purpose:   Multiply a vector that's been distributed among the processes
 * In args:   local_x:  local storage of one of the vectors being multiplied
 *            local_y:  local storage for the second vector being multiplied
 *            local_n:  the number of components in local_x and local_y
 * Out arg:   local_product:  local storage for the partial products of the two vectors
 */
void Parallel_dot_product(
      double  local_x[]  /* in  */, 
      double  local_y[]  /* in  */, 
      double* local_product  /* out */,
      int     local_n    /* in  */,
      int     my_rank    /* in  */) {
   int local_i;

   *local_product = 0.0;
   for (local_i = 0; local_i < local_n; local_i++)
      *local_product += local_x[local_i] * local_y[local_i];
}  /* Parallel_dot_product */