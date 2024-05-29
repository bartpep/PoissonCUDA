double*** jacobi(double ***matrix, double ***matrix_new, double ***f, int N, int iterations);
void d_jacobi(double* d_matrix,double* d_matrix_new,double* d_f, int N, int iterations, double start_time)