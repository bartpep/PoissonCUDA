#include <stdio.h>
#include <math.h>
#include <omp.h>

#include "jacobi.cuh"


__global__ void d_jacobi(double* d_matrix, double* d_matrix_new, double* d_f, int N, double dec,  double* dif_out) {
    int i = blockIdx.x*lockDim.x + threadIdx.x;
    int j = blockIdx.y*lockDim.y + threadIdx.y;
    int k = blockIdx.z*lockDim.z + threadIdx.z;

    if(i > 0 && i < N-1 && j > 0 && j < N-1 && k > 0 & k < N-1){
        int idx = i + j*N +k*N*N;
        d_matrix_new[idx] = dec * ( d_matrix[(i-1) + N * j + N*N*k] +
                                    d_matrix[(i+1) + N * j + N*N*k] +
                                    d_matrix[i + N * (j-1) + N*N*k] +
                                    d_matrix[i + N * (j+1) + N*N*k] +
                                    d_matrix[i + N * j + N*N*(k-1)] +
                                    d_matrix[i + N * j + N*N*(k+1)] +
                                    d_f[idx]);
        atomicAdd(dif_out,d_matrix_new[idx]-d_matrix[idx])*(d_matrix_new[idx]-d_matrix[idx]);
    }
    cudaDeviceSynchronize();

    //Update d_matrix to be the new matrix
    d_matrix = d_matrix_new;
    cudaDeviceSynchronize();

    cudacudaMemcpy(dif, dif_out, cudaDeviceToHost);
}



double*** jacobi(double ***matrix, double ***matrix_new, double ***f, int N, int iterations){
    printf("Starting sequential version\n");
    // Multiplication is easier than division
    double end_time, start_time = omp_get_wtime(); 


    //Allocate GPU memory: device matrixes
    size_t size = N*N*N*sizeof(double);
    double *d_matrix, *d_matrix_new, *d_f, *dif_out, *h_dif_out;
    
    cudaMalloc((void**)&d_matrix,size);
    cudaMalloc((void**)&d_f,size);
    cudaMalloc((void**)&dif_out, sizeof(double));
    
    end_time = omp_get_wtime();
    printf("Initialization time: %.2f", omp_get_wtime()- start_time);
    
    // Copy matrix values from host to device
    cudaMemcpy(d_matrix,matrix,size,cudaMemcpyHostToDevice);
    cudaMemcpy(d_matrix_new,matrix_new,size,cudaMemcpyHostToDevice);
    cudaMemcpy(d_f,f,size,cudaMemcpyHostToDevice);

    printf("Copy time: %.2f",omp_get_wtime() + end_time - start_time);
    end_time = omp_get_wtime();
    
    int count = 0;
    double dif = 100;
    while(count < iterations && dif > 1e-5){
        // Run Jacobi simulation on GPUs
        d_jacobi<<<NB,TPB>>>(d_matrix,d_matrix_new,d_f,N,iterations,start_time);
        
        //Update the necessary variables 
        end_time = omp_get_wtime(); 
        count++;
        printf("Final rounds: %.4f seconds\n", end_time - start_time);
    }

    // Transfer Result matrix back to host
    cudacudaMemcpy(matrix, d_matrix, cudaDeviceToHost);
    return matrix;
}