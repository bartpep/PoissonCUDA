#include <stdio.h>
#include <math.h>
#include <omp.h>
#include "jacobi.h"



// Transfer data from device to host
double*** jacobi(double ***matrix, double ***matrix_new, double ***f, int N, int iterations, int *p_iter,double difference ){
    printf("Starting sequential  version\n");
    // Multiplication is easier than division
    float dec = 1.0/6;
    float dif_t, dif = 100.0;
    int rounds = 0;
    int n_bounds = N-1;
    int n_dif = N-2;
    double start_time, end_time;
    
    // Get the start time
    start_time = omp_get_wtime();
    
    while(rounds < iterations and dif > difference){
        dif = 0;
        for(int i = 1; i <n_bounds;i++){
            for(int j = 1; j < n_bounds; j++){
                for(int k=1; k < n_bounds; k++){

                    matrix_new[i][j][k] = dec * (matrix[i-1][j][k] +
                                                matrix[i+1][j][k] +
                                                matrix[i][j-1][k] +
                                                matrix[i][j+1][k] +
                                                matrix[i][j][k-1] +
                                                matrix[i][j][k+1] +
                                                f[i][j][k]);
                    
                    dif_t = matrix_new[i][j][k]-matrix[i][j][k];
                    dif += dif_t*dif_t;
                    //printf("Element (%d,%d,%d) with temp = %.3f and %.3f has been successfully added\n",i,j,k, matrix_new[i][j][k], f[i][j][k]);
                }
            }
        }
        

        //Copy the matrix
        for(int i = 1; i <N-1;i++){
            for(int j = 1; j < N-1; j++){
                for(int k=1; k < N-1; k++){
                    matrix[i][j][k] = matrix_new[i][j][k];
                }
            }
        }
        // Update test values for the while statement
        dif = sqrt(dif / (n_dif*n_dif*n_dif));
        rounds++;

        if (rounds % 50 == 0) {
            end_time = omp_get_wtime();
            printf("Round %d diff: %.4f after %.4f seconds\n", rounds, dif, end_time - start_time);  
        }  
    }

    // print final and return matrix 
    end_time = omp_get_wtime();
    printf("Round %d diff: %.5f after %.0f seconds\n", rounds, dif, end_time - start_time);
    *p_iter = rounds;
    return matrix;
}