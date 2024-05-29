#include <stdio.h>
#include <math.h>
#include <omp.h>

#include "jacobi.h"

double*** jacobi(double ***matrix, double ***matrix_new, double ***f, int N, int iterations) {
    printf("Starting OpenMP version\n");
    
    // Multiplication is easier than division
    float dec = 1.0 / 6;
    float d, dif = 100.0;
    int rounds = 0;
    int n_bounds = N - 1;
    int n_dif = N - 2;
    int i, j, k;
    double start_time, end_time;

    // Get the start time
    start_time = omp_get_wtime();

    // Loop through all fields
    while (rounds < iterations && dif > .00001) {
        dif = 0.0;
        
        #pragma omp parallel reduction(+ : dif) shared(matrix, matrix_new, f, N) private(i, j, k, d)
        {
            #pragma omp for
            for (i = 1; i < n_bounds; i++) {
                for (j = 1; j < n_bounds; j++) {
                    for (k = 1; k < n_bounds; k++) {
                        // Calculate new value
                        matrix_new[i][j][k] = dec * (matrix[i - 1][j][k] +
                                                     matrix[i + 1][j][k] +
                                                     matrix[i][j - 1][k] +
                                                     matrix[i][j + 1][k] +
                                                     matrix[i][j][k - 1] +
                                                     matrix[i][j][k + 1] +
                                                     f[i][j][k]);
                                                        
                        d = matrix_new[i][j][k] - matrix[i][j][k];
                        dif += d * d;
                    }
                }
            } // implicit barrier here
        
            // Copy the matrix
            #pragma omp for
            for (i = 0; i < N; i++) {
                for (j = 0; j < N; j++) {
                    for (k = 0; k < N; k++) {
                        matrix[i][j][k] = matrix_new[i][j][k];
                    }
                }
            } // implicit barrier here
        } // end of parallelized section
        
        // Updates while loop
        rounds++;
        dif = sqrt(dif / (n_dif * n_dif * n_dif));
        if (rounds % 100 == 0) {
            end_time = omp_get_wtime();
            printf("Round %d diff: %.4f after %.4f seconds\n", rounds, dif, end_time - start_time);  
        }  
    }  
    
    end_time = omp_get_wtime(); 
    printf("Final rounds: %d with difference %.4f after %.4f seconds\n", rounds, dif, end_time - start_time);

    return matrix;
}