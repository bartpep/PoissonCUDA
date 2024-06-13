 #include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>
#include <omp.h>

// Define auxiliary functions
#include "auxiliaryFunctions/initial_functions.h"
#include "auxiliaryFunctions/print.h"
#include "auxiliaryFunctions/alloc3d.h"


//Defines which version to use
#ifdef CUDA
    #include "jacobiCUDA/jacobi.cuh"
#elif MP
    #include "jacobiOpen/jacobi.h"
#else 
    #include "jacobiSequential/jacobi.h"
#endif

// Wrapper function for Jacobi depending on type being run
double ***jacobi_wrapper(double ***matrix, double ***matrix_out, double ***f, int N, int iterations, double dif_threshold, int NB, int TPB){
    #ifdef CUDA
        matrix = jacobi(matrix, matrix_out,f,N, iterations, NB,TPB);
    #else
        matrix = jacobi(matrix, matrix_out,f,N, iterations,dif_threshold);
    #endif

    return matrix;
}

// VTK wrapper function
void vtk_wrapper(double ***matrix, int N){
    #ifdef CUDA
        print_vtk("vtk/cuda.vtk",N,matrix);
    #elif MP
        print_vtk("vtk/mp.vtk",N,matrix);
    #else
        print_vtk("vtk/seq.vtk",N,matrix);
    #endif
}


// Set parameters for the simulation
int iterations = 400;
int N = 250;
int TPB  = 32;
int NB = 10;
double difference = .00001;

int main(int argv, char *argc[]){
     // Initialize the matrixes at starting values    
    //printf("Main executes up to initial conditions\n");
    // Update the 3d_malloc function as this will create a solid block of data that can be copied in cuda
    double*** matrix = initial_conditions(N);
    double*** matrix_out = malloc_3d(N,N,N);
    double*** f = f_matrix(N);
    
    //Run the Jacobi
    matrix = jacobi_wrapper(matrix, matrix_out,f,N,iterations,difference,NB,TPB);
    
    
    //print final matrix
    vtk_wrapper(matrix,N);
 

   return 0;
}

