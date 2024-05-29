#include <stdio.h>
#include <math.h>
#include "jacobi.h"


// Transfer data from device to host
double*** jacobi(double ***matrix, double ***matrix_new, double ***f, int N, int iterations){
    printf("Starting sequential  version\n");
    // Multiplication is easier than division
    float dec = 1.0/6;
    float dif = 100.0;
    int rounds = 0;
    int n_bounds = N-1;
    int n_dif = N-2;
    

    
    
    while(rounds < iterations and dif > .00001){
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
                    
                    dif += (matrix_new[i][j][k]-matrix[i][j][k])*(matrix_new[i][j][k]-matrix[i][j][k]);
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
        if(rounds % 100 == 0){
            printf("Round %d diff: %.4f\n", rounds, dif);
        }
    }

    // Transfer back from device to host
    // Matrix 

    // Clear f and Matrix_new
        
    return matrix;
}