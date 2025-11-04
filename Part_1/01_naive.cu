#include <stdio.h>
#include<cuda_runtime.h> 
#include <stdlib.h> // for rand()

__global__ void naivemult(float *a, float *b, float *c, int m, int k, int n) 
{
    int col = blockIdx.x * blockDim.x + threadIdx.x; // column index
    int row = blockIdx.y * blockDim.y + threadIdx.y; // row index
    if (row < m && col < n) {
        float sum = 0.0f;
        for (int i = 0; i < k; i++) {
            sum += a[row * k + i] * b[i * n + col]; // multply (dot product) & accumulate
        }
        c[row * n + col] = sum; // store result
    }
}
void init_matrix(float *mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        mat[i] = (float)rand() / RAND_MAX; // generate random float between 0 and 1
    }
}


int  main()
{
    const int M = 1024;
    const int K = 1024;  // dimensions for matrix A(MxK) and B(KxN)
    const int N = 1024;

    cudaEvent_t start, stop;
    cudaEventCreate(&start); //initializing cuda event objects
    cudaEventCreate(&stop);

    //initialize host pointers
    float *h_a, *h_b, *h_c_gpu;
    //initialize device pointers
    float *d_a, *d_b, *d_c;
    // initialize sizes
    size_t size_a = M * K * sizeof(float);
    size_t size_b = K * N * sizeof(float);
    size_t size_c = M * N * sizeof(float);
    // allocate host memory
    h_a = (float *)malloc(size_a);
    h_b = (float *)malloc(size_b);
    h_c_gpu = (float *)malloc(size_c);
    // initialize matrices
    init_matrix(h_a, M, K);
    init_matrix(h_b, K, N);
    // allocate device memory
    cudaMalloc((void **)&d_a, size_a);
    cudaMalloc((void **)&d_b, size_b);
    cudaMalloc((void **)&d_c, size_c);
    // copy matrices from host to device
    cudaMemcpy(d_a, h_a, size_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size_b, cudaMemcpyHostToDevice);
    // initialize grid and block dimensions
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y); // ceiling division
    // execute kernel
    naivemult<<<grid, block>>>(d_a, d_b, d_c, M, N, K);
    //synchronize device
    cudaDeviceSynchronize();
    // Timed Benchmarking
    int iterations = 100;
    float total_time = 0.0f;

    for (int i = 0; i < iterations; i++) {
        
        cudaEventRecord(start);

        
        naivemult<<<grid, block>>>(d_a, d_b, d_c, M, N, K);

        
        cudaEventRecord(stop);

        
        cudaEventSynchronize(stop);

        
        float iter_time;
        cudaEventElapsedTime(&iter_time, start, stop);
        total_time += iter_time;
    }
    printf("Average kernel execution time: %f ms\n", total_time / iterations);
    printf("Average GFLOPS: %f\n", 2.0f * N * M * K / (total_time / iterations) / 1e6);
    // copy result matrix from device to host
    cudaMemcpy(h_c_gpu, d_c, size_c, cudaMemcpyDeviceToHost);
    // destroy cuda event objects
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    // free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    // free host memory
    free(h_a);
    free(h_b);
    free(h_c_gpu);
    return 0;
}
