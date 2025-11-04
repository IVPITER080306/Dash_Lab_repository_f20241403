#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>

#define TILE_WIDTH 32

#define REG_TILE_ROWS 4 // Each thread computes 4 elements in a row
#define REG_TILE_COLS 4 // Each thread computes 4 elements in a columne


#define BLOCK_COLS 8 // Number of columns per block
#define BLOCK_ROWS 8 // Number of rows per block
//These blocks of 8x8 threads will cover a tile of 32x32 elements in the output matrix C
//Registers of length 4 are used to hold intermediate sums for 4 elements in a row of C

//The size of shared memory tiles is TILE_WIDTH x TILE_WIDTH (32X32)
//The size of blocks is BLOCK_ROWS x BLOCK_COLS (8X8)
//Each thread computes REG_TILE_ROWS x REG_TILE_COLS (4X4) elements


//The reduction in shared memory accesses is achieved by having each thread load multiple elements from global memory into registers and then store them into shared memory.

__global__ void tiling_reg_block(float *a, float *b, float *c, int m, int n, int k) 
{
    __shared__ float tile_a[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tile_b[TILE_WIDTH][TILE_WIDTH];

    // Row indexing adjusted for REG_TILE_ROWS.
    // row is the starting row for the 4 elements this thread will compute.
    int row = blockIdx.y * TILE_WIDTH  + threadIdx.y*REG_TILE_ROWS; 
    
    // Column indexing adjusted for REG_TILE_COLS.
    // col is the starting column for the 4 elements this thread will compute.
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x * REG_TILE_COLS;

    
    // Array to hold intermediate sums for 4X4 elements
    // Initialize all sums to 0.0f
    float sums[REG_TILE_ROWS][REG_TILE_COLS];
    for (int i = 0; i < REG_TILE_ROWS; i++) {
        for (int j = 0; j < REG_TILE_COLS; j++) {
            sums[i][j] = 0.0f;
        }
    }
    for (int t = 0; t < (k + TILE_WIDTH - 1) / TILE_WIDTH; t++) {
        // Load elements into shared memory tiles
        for (int i = 0; i < REG_TILE_ROWS; i++) {
            for (int j = 0; j < REG_TILE_COLS; j++) {
                int tiled_row = row + i;
                int tiled_col = t * TILE_WIDTH + threadIdx.x * REG_TILE_COLS + j;
                if (tiled_row < m && tiled_col < k) {
                    tile_a[threadIdx.y * REG_TILE_ROWS + i][threadIdx.x * REG_TILE_COLS + j] = a[tiled_row * k + tiled_col];
                } else {
                    tile_a[threadIdx.y * REG_TILE_ROWS + i][threadIdx.x * REG_TILE_COLS + j] = 0.0f;
                }
            }
        }

        for (int i = 0; i < REG_TILE_ROWS; i++) {
            for (int j = 0; j < REG_TILE_COLS; j++) {
                int tiled_row = t * TILE_WIDTH + threadIdx.y * REG_TILE_ROWS + i;
                int tiled_col = col + j;
                if (tiled_row < k && tiled_col < n) {
                    tile_b[threadIdx.y * REG_TILE_ROWS + i][threadIdx.x * REG_TILE_COLS + j] = b[tiled_row * n + tiled_col];
                } else {
                    tile_b[threadIdx.y * REG_TILE_ROWS + i][threadIdx.x * REG_TILE_COLS + j] = 0.0f;
                }
            }
        }

        __syncthreads();

        // Compute partial sums
        for (int p = 0; p < TILE_WIDTH; p++) {
            for (int i = 0; i < REG_TILE_ROWS; i++) {
                for (int j = 0; j < REG_TILE_COLS; j++) {
                    sums[i][j] += tile_a[threadIdx.y * REG_TILE_ROWS + i][p] * tile_b[p][threadIdx.x * REG_TILE_COLS + j];
                }
            }
        }

        __syncthreads();
    }
    // Write results to global memory
    for (int i = 0; i < REG_TILE_ROWS; i++) {
        for (int j = 0; j < REG_TILE_COLS; j++) {
            int row = blockIdx.y * REG_TILE_ROWS + i;
            int col = blockIdx.x * REG_TILE_COLS + j;
            if (row < m && col < n) {
                c[row * n + col] = sums[i][j];
            }
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
