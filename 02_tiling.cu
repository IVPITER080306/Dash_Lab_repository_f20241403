#include <stdio.h>
#include<cuda_runtime.h>
#include <stdlib.h>

#define TILE_WIDTH 32 // Define tile width for tiling optimization

__global__ void tiling(float *a, float *b, float *c, int m, int n, int k) 
{
    __shared__ float tile_a[TILE_WIDTH][TILE_WIDTH]; // Shared memory for tiles of A
    __shared__ float tile_b[TILE_WIDTH][TILE_WIDTH]; // Shared memory for tiles of B
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    for (int t = 0; t < (k + TILE_WIDTH - 1) / TILE_WIDTH; t++) // Loop over tiles
    {
        if (row < m && t * TILE_WIDTH + threadIdx.x < k)
            tile_a[threadIdx.y][threadIdx.x] = a[row * k + t * TILE_WIDTH + threadIdx.x]; // Load tile from A
        else
            tile_a[threadIdx.y][threadIdx.x] = 0.0f; // Handle boundary conditions
        if (col < n && t * TILE_WIDTH + threadIdx.y < k)
            tile_b[threadIdx.y][threadIdx.x] = b[(t * TILE_WIDTH + threadIdx.y) * n + col];// Load tile from B
        else
            tile_b[threadIdx.y][threadIdx.x] = 0.0f; // Handle boundary conditions
        __syncthreads(); // Synchronize to ensure tiles are loaded before computation
        // Compute dot product of tile rows and columns
        for (int i = 0; i < TILE_WIDTH; i++) {
            sum += tile_a[threadIdx.y][i] * tile_b[i][threadIdx.x];
        }
        __syncthreads(); // Synchronize before loading new tiles
    }
    if (row < m && col < n)
        c[row * n + col] = sum; // Write the result
}
void init_matrix(float *mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        mat[i] = (float)rand() / RAND_MAX;
    }
}


int  main()
{
    const int M = 1024;
    const int K = 1024;
    const int N = 1024;

        
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float *h_a, *h_b, *h_c_gpu;
    float *d_a, *d_b, *d_c;
    size_t size_a = M * K * sizeof(float);
    size_t size_b = K * N * sizeof(float);
    size_t size_c = M * N * sizeof(float);
    h_a = (float *)malloc(size_a);
    h_b = (float *)malloc(size_b);
    h_c_gpu = (float *)malloc(size_c);
    init_matrix(h_a, M, K);
    init_matrix(h_b, K, N);
    cudaMalloc((void **)&d_a, size_a);
    cudaMalloc((void **)&d_b, size_b);
    cudaMalloc((void **)&d_c, size_c);
    cudaMemcpy(d_a, h_a, size_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size_b, cudaMemcpyHostToDevice);
    dim3 block(TILE_WIDTH, TILE_WIDTH);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    tiling<<<grid, block>>>(d_a, d_b, d_c, M, N, K);
    cudaDeviceSynchronize();
    
    int iterations = 100;
    float total_time = 0.0f;

    for (int i = 0; i < iterations; i++) {
        
        cudaEventRecord(start);

        
        tiling<<<grid, block>>>(d_a, d_b, d_c, M, N, K);

        
        cudaEventRecord(stop);


        cudaEventSynchronize(stop);

        float iter_time;
        cudaEventElapsedTime(&iter_time, start, stop);
        total_time += iter_time;
    }
    printf("Average kernel execution time: %f ms\n", total_time / iterations);
    printf("Average GFLOPS: %f\n", 2.0f * N * M * K / (total_time / iterations) / 1e6);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaMemcpy(h_c_gpu, d_c, size_c, cudaMemcpyDeviceToHost);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c_gpu);
    return 0;
}