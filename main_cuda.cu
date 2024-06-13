#include <stdio.h>
#include <iostream>
#include <cuda_runtime.h>
#include "auxiliaryFunctions/alloc3d.h"
#include "auxiliaryFunctions/alloc3d_gpu.h"
#include "auxiliaryFunctions/initial_functions.h"
#include "auxiliaryFunctions/print.h"
#include "jacobiCUDA/jacobi.cuh"

#define NUM_THREADS 3
#define BLOCKS 2
#define ITERATIONS 2

/*__global__ void d_jacobi(int N, int M, int K, double ***x, double ***y, double ***z){
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    int j = threadIdx.y + blockIdx.y*blockDim.y;
    int k = threadIdx.z + blockIdx.z*blockDim.z;
    
    //Check if not out of bounds
    if (i < N && j < M && k < K) {        
        z[i][j][k] = x[i][j][k] + y[i][j][k];
        //if(verbose ==printf("Kernel (%d,%d,%d, %f): \n", i,j,k, z[i][j][k]);
    }
} */

int main(int argc, char *argv[]){
    int N = atoi(argv[1]);  
    int M = atoi(argv[2]);
    int K = atoi(argv[3]); 
    int count = 0;

    printf("Vector length: (%d,%d,%d) \n",N,M,K);


    // Initialize memory host
    double ***h_matrix = (double***) malloc_3d(M,N,K);
    double ***h_matrix_new = (double***) malloc_3d(M,N,K); 
    double ***h_f = (double***) malloc_3d(M,N,K); 
    double h_dif = 100.0;
    printf("Host allocation complete\n");

    // Initialize memory device
    double ***d_matrix, ***d_matrix_new, ***d_f;
    double d_dif;
    cudaMalloc((void **)&d_dif, sizeof(double));
    d_matrix = d_malloc_3d_gpu(N,M,K);
    d_matrix_new = d_malloc_3d_gpu(N,M,K);
    d_f = d_malloc_3d_gpu(N,M,K);
   

    printf("Device memory allocated\n");
    
    // allocate values 
    h_f = f_matrix(h_f, N, M, K);
    h_matrix = initial_conditions(h_matrix, N, M, K);
    h_matrix_new = initial_conditions(h_matrix, N, M, K);
    printf("Values added to vectors\n");

    // copy to device
    int size = N*M*K*sizeof(double);
    cudaMemcpy(d_matrix,h_matrix,size,cudaMemcpyHostToDevice);
    cudaMemcpy(d_matrix_new,h_matrix_new,size,cudaMemcpyHostToDevice);
    cudaMemcpy(d_f,h_f,size,cudaMemcpyHostToDevice);

    printf("Data copied to device\n");


    // Run Kernel
    int thread_per_block_x = NUM_THREADS, thread_per_block_y = NUM_THREADS,thread_per_block_z = NUM_THREADS;
    dim3 block(thread_per_block_x, thread_per_block_y, thread_per_block_z);
    dim3 grid((N + thread_per_block_x - 1) / thread_per_block_x, 
          (M + thread_per_block_y - 1) / thread_per_block_y,
          (K + thread_per_block_z - 1) / thread_per_block_z);
    
    std::cout << "Dimensions: (" << block.x << ", " << block.y << ", " << block.z << ")" << std::endl;
    std::cout << "Dimensions: (" << grid.x << ", " << grid.y << ", " << grid.z << ")" << std::endl;

    while(count < ITERATIONS){
        d_jacobi<<<grid,block>>>(d_matrix,d_matrix_new,d_f,N,M,K);
        cudaDeviceSynchronize();
        count++;
        h_dif = cudaMemcpy(&h_dif,&d_dif,size,cudaMemcpyDeviceToHost);
        printf("h_dif: %.3f\n", h_dif);
    }
    printf("Kernel completed\n");

    // copy to host
    //cudaMemcpy(h_x,d_x,size,cudaMemcpyDeviceToHost);
    cudaMemcpy(h_matrix,d_matrix,size,cudaMemcpyDeviceToHost);
    printf("Data copied to host\n");

    // Print to terminal 
    print_matrix(h_matrix,N);
    
    // Print for VTK file
    char file_name[50];
    sprintf(file_name, "vtk/cuda_%d_%d_%d_%d", N,M,K,ITERATIONS);
    printf("%s\n", file_name);
    print_vtk(file_name,N,h_matrix);

}