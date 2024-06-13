 #include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>
#include <omp.h>
#include <iostream>
#include <fstream>

// Define auxiliary functions
#include "auxiliaryFunctions/initial_functions.h"
#include "auxiliaryFunctions/print.h"
#include "auxiliaryFunctions/alloc3d.h"


//Defines which version to use
#ifdef MP
    #include "jacobiOpen/jacobi.h"
#else 
    #include "jacobiSequential/jacobi.h"
#endif


// VTK wrapper function
void vtk_wrapper(double ***matrix, int N){
    #ifdef MP
        print_vtk("vtk/mp.vtk",N,matrix);
    #else
        print_vtk("vtk/seq.vtk",N,matrix);
    #endif
}


// Set parameters for the simulation
int iterations = 10000;
int N = 10;
double difference = .0005;

int main(int argv, char *argc[]){
     // Initialize the matrixes at starting values    
    //printf("Main executes up to initial conditions\n");
    // Update the 3d_malloc function as this will create a solid block of data that can be copied in cuda
    double*** matrix = malloc_3d(N,N,N);
    double*** matrix_out = malloc_3d(N,N,N);
    double*** f = malloc_3d(N,N,N);
    
    f = f_matrix(f,N,N,N);
    matrix = initial_conditions(matrix, N,N,N);
    
    //Run the Jacobi
    int iter = 0;
    int *p_iter = &iter;
    double start = omp_get_wtime();
    matrix = jacobi(matrix, matrix_out,f,N,iterations,p_iter,difference);
    double end = omp_get_wtime();
    
    //print final matrix
    vtk_wrapper(matrix,N);
    
    //Calculate KPI
    double time = end - start;
    int thread_num = 1;
    double MLUPS = (N-2)*(N-2)*(N-2) *iterations / time;
    
    std::ofstream myfile;
    myfile.open("./results/sequential.csv");
    myfile << N << ", " <<  time << ", " << iter << ", " << MLUPS << ", " << thread_num <<"\n";
    myfile.close();


   return 0;
}

