#include <stdio.h>
#include <stdlib.h> 
#include <string.h>
#include <omp.h>
#include <math.h> 
#include <float.h>

#define MAXCHAR 25
#define DEBUG   1

double euclidean(double x1, double y1, double x2, double y2) {
    return sqrt(pow(x2 - x1, 2.0) + pow(y2 - y1, 2.0));
}

int main(int argc, char* argv[]) {
    // Catch console errors
    if (argc != 8) {
        printf("argc=%d, USE LIKE THIS: kmeans_clustering n_points points.csv n_centroids centroids.csv output.csv time.csv num_threads\n", argc);
        for (int i = 0; i < argc; i++) {
            printf(argv[i]); printf("\n");
        }
        exit(-1);
    }

    // points ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
    int num_points = strtol(argv[1], NULL, 10);
    FILE* pointsFile = fopen(argv[2], "r");
    if (pointsFile == NULL) {
        printf("Could not open file %s", argv[2]);
        exit(-2);
    }

    // centroids ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
    int num_centroids = strtol(argv[3], NULL, 10);
    FILE* centroidsFile = fopen(argv[4], "r");
    if (centroidsFile == NULL) {
        printf("Could not open file %s", argv[4]);
        fclose(pointsFile);
        exit(-3);
    }

    // output ~~~~~~~~~~~~~~~~~~~~~~~~~~ //
    FILE* outputFile = fopen(argv[5], "w");
    FILE* timeFile = fopen(argv[6], "w");

    // threads ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
    int num_threads = strtol(argv[7], NULL, 10);

    // array ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
    double* points_x = malloc(num_points * sizeof(double));
    double* points_y = malloc(num_points * sizeof(double));

    // centroid array /////////////////////////////////////////
    double* centroids_x = malloc(num_centroids * sizeof(double));
    double* centroids_y = malloc(num_centroids * sizeof(double));



    // Store values ~~~~~~~~ //  
    // temporarily store values
    char str[MAXCHAR];

    // Storing point values //
    int k = 0;
    while (fgets(str, MAXCHAR, pointsFile) != NULL) {
        sscanf(str, "%lf,%lf", &(points_x[k]), &(points_y[k]));
        k++;
    }
    fclose(pointsFile);

    // Storing centroid values //
    k = 0;
    while (fgets(str, MAXCHAR, centroidsFile) != NULL) {
        sscanf(str, "%lf,%lf", &(centroids_x[k]), &(centroids_y[k]));;
        k++;
    }
    fclose(centroidsFile);



    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
    // start time
    double start = omp_get_wtime();

    // TODO: implement algorithm here :)

    int clusters[num_points]; // Array of point index : closest cluster
    double avg_moving_dist = FLT_MAX;

    // Store parameters of cluster averages so we don't iterate through all the points again later
    double* cluster_sum_x = calloc(num_centroids, sizeof(double));
    double* cluster_sum_y = calloc(num_centroids, sizeof(double));
    int* cluster_num_points = calloc(num_centroids, sizeof(int));

    // While not converged (i.e. the average moving distance of all the centroids in the last iteration is more than 1.0)
#pragma omp parallel num_threads(num_threads) default(none) shared(num_points, num_centroids, points_x, points_y, centroids_x, centroids_y, clusters, avg_moving_dist, cluster_sum_x, cluster_sum_y, cluster_num_points)
    {

        while (avg_moving_dist > 1.0) {

        // Reset running totals
    #pragma omp single
        {  
            avg_moving_dist = 0.0;
            memset(cluster_sum_x, 0.0, num_centroids * sizeof(double));
            memset(cluster_sum_y, 0.0, num_centroids * sizeof(double));
            memset(cluster_num_points, 0.0, num_centroids * sizeof(int));
        }

            // Local versions to avoid sharing overhead during loop
            double local_sum_x[num_centroids];
            double local_sum_y[num_centroids];
            int local_num_points[num_centroids];
            memset(local_sum_x, 0.0, sizeof(local_sum_x));
            memset(local_sum_y, 0.0, sizeof(local_sum_y));
            memset(local_num_points, 0.0, sizeof(local_num_points));

        // For every point, assign it to its nearest centroid (for loop 1)
    #pragma omp for schedule(static)
            for (int i = 0; i < num_points; i++) {
                double px = points_x[i];
                double py = points_y[i];
                
                int closest = 0;
                double closestDist = DBL_MAX;

                for (int j = 0; j < num_centroids; j++) {
                    double cx = centroids_x[j];
                    double cy = centroids_y[j];
                    
                    double dist = euclidean(px,py,cx,cy);
                    if (dist < closestDist) {
                        closest = j;
                        closestDist = dist;
                    }
                }

                clusters[i] = closest; // Points are distributed across threads so this is safe
                local_sum_x[closest] += px;
                local_sum_y[closest] += py;
                local_num_points[closest]++;
            }

    // Converge
    #pragma omp critical
        {
            for (int i = 0; i < num_centroids; i++) {
                cluster_sum_x[i] += local_sum_x[i];
                cluster_sum_y[i] += local_sum_y[i];
                cluster_num_points[i] += local_num_points[i];
            }
        }

    // For every centroid, compute its new location as the geometric center of its assigned data points and then compute the moving distance by which it is moved from the previous location (for loop 2)
    // Embarrassingly parallel
    #pragma omp for schedule(static)
            for (int i = 0; i < num_centroids; i++) {
                double pts = cluster_num_points[i];
                double newx = cluster_sum_x[i] / pts;
                double newy = cluster_sum_y[i] / pts;
                
                double cx = centroids_x[i];
                double cy = centroids_y[i];
                
        #pragma omp atomic
                avg_moving_dist += euclidean(cx,cy,newx,newy); 

                centroids_x[i] = newx;
                centroids_y[i] = newy;
            }

        // Compute the average moving distance by which all the centroids are moved in this iteration.
        // Also ensures that all threads wait up between while iterations
        #pragma omp single
            {
                avg_moving_dist /= num_centroids;
            }
        }
    }

    // end time
    double end = omp_get_wtime();
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //



    // print time //
    double time_passed = end - start;
    fprintf(timeFile, "%f", time_passed);

    // print centroids //
    for (int c = 0; c < num_centroids; ++c) {
        fprintf(outputFile, "%f, %f", centroids_x[c], centroids_y[c]);
        if (c != num_centroids - 1) fprintf(outputFile, "\n");
    }



    // close files //
    fclose(outputFile);
    fclose(timeFile);

    // free memory //
    free(points_x);
    free(points_y);
    free(centroids_x);
    free(centroids_y);

    free(cluster_sum_x);
    free(cluster_sum_y);
    free(cluster_num_points);

    // return //
    return 0;
}

// INITIAL SERIAL IMPLEMENTATION (for reference idk)

// // While not converged (i.e. the average moving distance of all the centroids in the last iteration is more than 1.0)
//     while (avg_moving_dist > 1.0) {

//         // Store parameters of cluster averages so we don't iterate through all the points again later
//         double cluster_sum_x[num_centroids]; memset(cluster_sum_x, 0.0, sizeof(cluster_sum_x));
//         double cluster_sum_y[num_centroids]; memset(cluster_sum_y, 0.0, sizeof(cluster_sum_y));
//         int cluster_num_points[num_centroids]; memset(cluster_num_points, 0.0, sizeof(cluster_num_points));

//         // For every point, assign it to its nearest centroid (for loop 1)
//         for (int i = 0; i < num_points; i++) {
//             double px = points_x[i];
//             double py = points_y[i];
            
//             int closest = 0;
//             double closestDist = DBL_MAX;

//             for (int j = 0; j < num_centroids; j++) {
//                 double cx = centroids_x[j];
//                 double cy = centroids_y[j];
                
//                 double dist = euclidean(px,py,cx,cy);
//                 if (dist < closestDist) {
//                     closest = j;
//                     closestDist = dist;
//                 }
//             }

//             clusters[i] = closest;
//             cluster_sum_x[closest] += px;
//             cluster_sum_y[closest] += py;
//             cluster_num_points[closest]++;
//         }

//         avg_moving_dist = 0.0;

//         // For every centroid, compute its new location as the geometric center of its assigned data points and then compute the moving distance by which it is moved from the previous location (for loop 2)
//         for (int i = 0; i < num_centroids; i++) {
//             double pts = cluster_num_points[i];
//             double newx = cluster_sum_x[i] / pts;
//             double newy = cluster_sum_y[i] / pts;
            
//             double cx = centroids_x[i];
//             double cy = centroids_y[i];
            
//             avg_moving_dist += euclidean(cx,cy,newx,newy); 

//             centroids_x[i] = newx;
//             centroids_y[i] = newy;
//         }

//         avg_moving_dist /= num_centroids; // Compute the average moving distance by which all the centroids are moved in this iteration.
//     }