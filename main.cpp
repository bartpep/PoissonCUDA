#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

// Define auxiliary functions
#include "auxiliaryFunctions/initial_functions.h"
#include "auxiliaryFunctions/print.h"

//Defines which version to use
#ifdef CUDA
    #include "jacobiCUDA/jacobi.h"
#elif MP
    #include "jacobiOpen/jacobi.h"
#else 
    #include "jacobiSequential/jacobi.h"
#endif


// Set parameters for the simulation
#define iterations 1000
#define N 100
#define TPB 32
#define NB 10

int main(){
     // Initialize the matrixes at starting values    
    //printf("Main executes up to initial conditions\n");
    double*** matrix = initial_conditions(N);
    double*** matrix_out = allocate_matrix(N);
    double*** f = f_matrix(N);
    
    //Run the Jacobi
    matrix = jacobi(matrix, matrix_out,f,N, iterations);
    
    //print final matrix
    print_vtk("print_matrix.vtk",N,matrix);

   return 0;
}

