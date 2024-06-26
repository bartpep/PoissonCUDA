#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "initial_functions.h"
#include "alloc3d.h"



// Create and fill the initial conditions
double*** initial_conditions(double ***matrix, int N, int M, int K){
    //printf("Initial conditions has been called succesfully.\nN = %d\n", N);
    int n_bound = N-1;

    // define walls
    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            for(int k = 0; k < N; k++){
                if ((i == 0 or i == n_bound or j == 0 or j == n_bound or k == n_bound) and k != 0){
                    matrix[i][j][k] = 20;
                }else{
                    matrix[i][j][k] = 0;
                }
            }
        }
    }
    return matrix;
}

// Define f matrix
double*** f_matrix(double ***matrix, int N, int M, int K){
    //define heater (First move 1 to the left to create 0-2 frame, then stretch by N/2: (heater cordinates + 1) * N/2)
    float delta = 2.0/N;
    delta = delta*delta;

    for(int i = floor(0*N/2);i < floor(5*N/16);i++){
        for(int j = floor(0*N/2);j < floor(N/4);j++){
            for(int k = floor(N/6);k < floor(N/2);k++){
                matrix[i][j][k] = delta * 2000;
            }
        }
    }
    return matrix;
}
// Print function to check matrix results
void print_matrix(double*** matrix, int N){ 

    for(int i =0; i<N; i++){
        for(int j = 0; j < N; j++){
            for(int k = 0; k < N; k++){
                std::cout << matrix[i][j][k] << " ";
            }
            std::cout << std::endl;
        }
            std::cout << std::endl;
    }
}