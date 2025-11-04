#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>

#define TILE_WIDTH 32

#define REG_TILE_WIDTH 4 // Each thread computes 4 elements in a row


#define BLOCK_COLS 8 // Number of columns per block
#define BLOCK_ROWS 32 // Number of rows per block
//These blocks of 8x32 threads will cover a tile of 32x32 elements in the output matrix C
//Registers of length 4 are used to hold intermediate sums for 4 elements in a row of C

//The size of shared memory tiles is TILE_WIDTH x TILE_WIDTH (32X32)
//The size of blocks is BLOCK_ROWS x BLOCK_COLS (8X32)
//Each thread computes REG_TILE_WIDTH elements in a row (4 elements)


//The reduction in shared memory accesses is achieved by having each thread load multiple elements from global memory into registers and then store them into shared memory.

__global__ void tiling_reg_block(float *a, float *b, float *c, int m, int n, int k) 
{
    __shared__ float tile_a[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tile_b[TILE_WIDTH][TILE_WIDTH];

    int row = blockIdx.y * TILE_WIDTH + threadIdx.y; 
    
    // Column indexing adjusted for REG_TILE_WIDTH.
    // col is the starting column for the 4 elements this thread will compute.
    // int col = blockIdx.x * TILE_WIDTH + threadIdx.x * REG_TILE_WIDTH; // <-- REMOVED, UNUSED

    
    // Array to hold intermediate sums for 4 elements in a row
    float sum[REG_TILE_WIDTH];
    for (int i = 0; i < REG_TILE_WIDTH; i++) {
        sum[i] = 0.0f;
    }

    for (int t = 0; t < (k + TILE_WIDTH - 1) / TILE_WIDTH; t++) {
        

        // We have a (8, 32) block. We loop 4 times (TILE_WIDTH / BLOCK_COLS = 32/8 = 4).
        // In each loop, all 8 threads load one column, ensuring
        // threadIdx.x maps to consecutive memory addresses.
        for (int j = 0; j < TILE_WIDTH; j += BLOCK_COLS) { // j = 0, 8, 16, 24
            
            // Shared memory indices
            int s_row = threadIdx.y;
            int s_col = threadIdx.x + j;


            int g_row_a = row; // Use the global row for this thread
            int g_col_a = t * TILE_WIDTH + s_col;

            if (g_row_a < m && g_col_a < k) {
                tile_a[s_row][s_col] = a[g_row_a * k + g_col_a];
            } else {
                tile_a[s_row][s_col] = 0.0f;
            }


            int g_row_b = t * TILE_WIDTH + s_row;
            int g_col_b = blockIdx.x * TILE_WIDTH + s_col;

            if (g_row_b < k && g_col_b < n) {
                tile_b[s_row][s_col] = b[g_row_b * n + g_col_b];
            } else {
                tile_b[s_row][s_col] = 0.0f;
            }
        }
        
        __syncthreads(); // Ensure all threads have loaded their elements


        for (int i = 0; i < TILE_WIDTH; i++) {
            float a_reg = tile_a[threadIdx.y][i]; // Load A element from shared memory
            
            for (int j = 0; j < REG_TILE_WIDTH; j++) {
                // Get the sequential column index this thread is responsible for
                int b_col_idx = threadIdx.x * REG_TILE_WIDTH + j; 
                float b_reg = tile_b[i][b_col_idx]; // Load B element from shared memory
                sum[j] += a_reg * b_reg; // Accumulate the product
            }
        }
        __syncthreads(); // Ensure all threads have completed computation
    }
    

    // The computation stored results in registers in a strided layout.
    // To write coalesced, we must first write these register values
    // to shared memory, then read them back in a coalesced pattern.

    // Step 1: Write register sums to shared memory (strided write to shared)
    // We can reuse tile_a since the computation is done.
    for (int j = 0; j < REG_TILE_WIDTH; j++) {
        int s_col = threadIdx.x * REG_TILE_WIDTH + j;
        tile_a[threadIdx.y][s_col] = sum[j];
    }

    __syncthreads(); // Ensure all sums are written to shared memory


    // This pattern is identical to the coalesced loading pattern.
    for (int j = 0; j < TILE_WIDTH; j += BLOCK_COLS) { // j = 0, 8, 16, 24
        
        int s_row = threadIdx.y;
        int s_col = threadIdx.x + j;

        int g_row = blockIdx.y * TILE_WIDTH + s_row;
        int g_col = blockIdx.x * TILE_WIDTH + s_col;

        if (g_row < m && g_col < n) {
            c[g_row * n + g_col] = tile_a[s_row][s_col];
        }
    }
}


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
    dim3 block(BLOCK_COLS, BLOCK_ROWS); // (8X32)
    dim3 grid((N + TILE_WIDTH - 1) / TILE_WIDTH, (M + TILE_WIDTH - 1) / TILE_WIDTH);
    tiling_reg_block<<<grid, block>>>(d_a, d_b, d_c, M, N, K);
    cudaDeviceSynchronize();

    
    int iterations = 100;
    float total_time = 0.0f;

    for (int i = 0; i < iterations; i++) {
        
        cudaEventRecord(start);

        
        tiling_reg_block<<<grid, block>>>(d_a, d_b, d_c, M, N, K);

        
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

