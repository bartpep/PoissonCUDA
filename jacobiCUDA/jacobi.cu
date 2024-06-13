#include <stdio.h>
#include <math.h>
#include <omp.h>

__global__ void d_jacobi(double ***d_matrix, double ***d_matrix_new, double ***d_f,  int N, int M, int K) {
    double d_dif = 100.1;
    double temp;
    
    double d_dec = 1.0/6.0;
    
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;
    int k = blockIdx.z*blockDim.z + threadIdx.z;
    
    if(i > 0 && i < N-1 && j > 0 && j < N-1 && k > 0 && k < N-1){
        d_matrix_new[i][j][k] = d_dec * ( d_matrix[(i-1)][j][k] +
                                    d_matrix[i+1][j][k] +
                                    d_matrix[i][j-1][k] +
                                    d_matrix[i][j+1][k] +
                                    d_matrix[i][j][k-1] +
                                    d_matrix[i][j][k+1] +
                                    d_f[i][j][k]);
        temp = d_matrix_new[i][j][k]-d_matrix[i][j][k];
        atomicAdd(d_dif,temp*temp);
    }
    //Update d_matrix to be the new matrix
    d_matrix[i][j][k] = d_matrix_new[i][j][k];

   
}