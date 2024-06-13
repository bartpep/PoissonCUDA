#include <stdio.h>
#include <cuda_runtime_api.h>

#define CUDACHECK(cmd) do {                         \
  cudaError_t err = cmd;                            \
  if (err != cudaSuccess) {                         \
    printf("Failed: Cuda error %s:%d '%s'\n",       \
        __FILE__,__LINE__,cudaGetErrorString(err)); \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

__global__ void d_malloc_3d_gpu_kernel1(double *** array3D, int m, int n, int k)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < m) {
        array3D[i] = (double **) array3D + m + i * n;
        //printf("k1: %i | %i\n", i, i* n);
    }
}

__global__ void d_malloc_3d_gpu_kernel2(double *** array3D, int m, int n, int k)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < m && j < n) {
        array3D[i][j] = (double *) array3D + m + m * n + (i * k * n) + (j * k);
        //printf("k2: %i %i | %i\n", i, j, (i * k * n) + (j * k));
    }
}

double ***
d_malloc_3d_gpu(int m, int n, int k) {

    if (k <= 0 || n <= 0 || m <= 0)
        return NULL;
    
    double ***array3D; 
    CUDACHECK( cudaMalloc((void**)&array3D, m * sizeof(double **) + m * n * sizeof(double *) + m * n * k * sizeof(double)));
    if (array3D == NULL) {
        return NULL;
    }

    dim3 block(16, 16);
    dim3 grid((m + 15) / 16, (n + 15) /16);
    d_malloc_3d_gpu_kernel1<<<grid.x, block.x>>>(array3D, m, n, k);
    d_malloc_3d_gpu_kernel2<<<grid, block>>>(array3D, m, n, k);
    CUDACHECK( cudaDeviceSynchronize() );
    return array3D;
}

void
free_gpu(double ***array3D) {
    CUDACHECK( cudaFree(array3D) );
}
