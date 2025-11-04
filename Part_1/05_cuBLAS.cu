#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <cublas_v2.h> 


void init_matrix(float *mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        mat[i] = (float)rand() / RAND_MAX;
    }
}


int main()
{
    const int M = 1024;
    const int K = 1024;
    const int N = 1024;
    const int ITERATIONS = 100;
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

 
    float *h_a, *h_b;
    size_t size_a = M * K * sizeof(float);
    size_t size_b = K * N * sizeof(float);
    size_t size_c = M * N * sizeof(float);
    
    h_a = (float *)malloc(size_a);
    h_b = (float *)malloc(size_b);
    
    init_matrix(h_a, M, K);
    init_matrix(h_b, K, N);
    

    float *d_a, *d_b, *d_c;
    cudaMalloc((void **)&d_a, size_a);
    cudaMalloc((void **)&d_b, size_b);
    cudaMalloc((void **)&d_c, size_c); 


    cublasHandle_t handle;
    cublasStatus_t status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("CUBLAS initialization failed!\n");
        return 1;
    }
    

    cudaMemcpy(d_a, h_a, size_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size_b, cudaMemcpyHostToDevice);
    
    const float alpha = 1.0f;
    const float beta = 0.0f;
    

    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_b, N, d_a, K, &beta, d_c, N);
    cudaDeviceSynchronize();

    
    float total_time_cublas = 0.0f;

    for (int i = 0; i < ITERATIONS; i++) {
        cudaEventRecord(start);
        
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_b, N, d_a, K, &beta, d_c, N);
        
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float iter_time;
        cudaEventElapsedTime(&iter_time, start, stop);
        total_time_cublas += iter_time;
    }
    printf("Average CUBLAS kernel time:  %f ms\n", total_time_cublas / ITERATIONS);
    printf("Average CUBLAS GFLOPS:  %f\n", 2.0f * N * M * K / (total_time_cublas / ITERATIONS) / 1e6);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cublasDestroy(handle);
    
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    
    free(h_a);
    free(h_b);
    
    return 0;
}
