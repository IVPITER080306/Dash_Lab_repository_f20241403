#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>

#define TILE_WIDTH 32

// Each thread computes a 4x4 tile of C
#define REG_TILE_ROWS 4 
#define REG_TILE_COLS 4 

// Block dimensions: 8x8 threads
// 32 / REG_TILE_COLS = 32 / 4 = 8 threads in X
#define BLOCK_THREADS_X 8 
// 32 / REG_TILE_ROWS = 32 / 4 = 8 threads in Y
#define BLOCK_THREADS_Y 8
// Total 64 threads per block

__global__ void tiling_double_buffer_4x4(float *a, float *b, float *c, int m, int n, int k) 
{
    __shared__ float tile_a[2][TILE_WIDTH][TILE_WIDTH];
    __shared__ float tile_b[2][TILE_WIDTH][TILE_WIDTH];

    // New Row/Col indexing for a 4x4 register tile
    // row is the STARTING row for the 4 elements this thread will compute
    int row = blockIdx.y * TILE_WIDTH + threadIdx.y * REG_TILE_ROWS;
    // col is the STARTING col for the 4 elements this thread will compute
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x * REG_TILE_COLS;

    // New 4x4 array (16 elements) to hold intermediate sums in registers.
    float sum[REG_TILE_ROWS][REG_TILE_COLS];
    for (int r = 0; r < REG_TILE_ROWS; r++) {
        for (int c_ = 0; c_ < REG_TILE_COLS; c_++) {
            sum[r][c_] = 0.0f;
        }
    }

    int a_start_row_global = blockIdx.y * TILE_WIDTH;
    int b_start_col_global = blockIdx.x * TILE_WIDTH;

    int num_tiles = (k + TILE_WIDTH - 1) / TILE_WIDTH;

    {
        int t = 0; 
        int a_start_col = t * TILE_WIDTH;
        int b_start_row = t * TILE_WIDTH;

        // New 4x4 spatial load
        // Each thread (8x8) loads its 4x4 patch into tile_a[0] and tile_b[0]
        for (int r = 0; r < REG_TILE_ROWS; r++) {
            int local_row = threadIdx.y * REG_TILE_ROWS + r;
            int global_row_a = a_start_row_global + local_row;
            int global_row_b = b_start_row + local_row;

            for (int c_ = 0; c_ < REG_TILE_COLS; c_++) {
                int local_col = threadIdx.x * REG_TILE_COLS + c_;
                int global_col_a = a_start_col + local_col;
                int global_col_b = b_start_col_global + local_col;

                if (global_row_a < m && global_col_a < k)
                    tile_a[0][local_row][local_col] = a[global_row_a * k + global_col_a];
                else
                    tile_a[0][local_row][local_col] = 0.0f;
                
                if (global_row_b < k && global_col_b < n)
                    tile_b[0][local_row][local_col] = b[global_row_b * n + global_col_b];
                else
                    tile_b[0][local_row][local_col] = 0.0f;
            }
        }
    }
    __syncthreads(); // Wait for the first tile (t=0) to be loaded

    for (int t = 0; t < num_tiles; t++) {
        
        int current_buffer = t % 2;
        int next_buffer = (t + 1) % 2;

        if (t < num_tiles - 1) {
            int next_tile_k_a = (t + 1) * TILE_WIDTH; // Col for A
            int next_tile_k_b = (t + 1) * TILE_WIDTH; // Row for B
            
            // New 4x4 spatial load for the *next* tile
            for (int r = 0; r < REG_TILE_ROWS; r++) {
                int local_row = threadIdx.y * REG_TILE_ROWS + r;
                int global_row_a = a_start_row_global + local_row; // Row of A is constant
                int global_row_b = next_tile_k_b + local_row;      // Row of B changes with t

                for (int c_ = 0; c_ < REG_TILE_COLS; c_++) {
                    int local_col = threadIdx.x * REG_TILE_COLS + c_;
                    int global_col_a = next_tile_k_a + local_col; // Col of A changes with t
                    int global_col_b = b_start_col_global + local_col; // Col of B is constant

                    // Load tile_a[next_buffer]
                    if (global_row_a < m && global_col_a < k)
                        tile_a[next_buffer][local_row][local_col] = a[global_row_a * k + global_col_a];
                    else
                        tile_a[next_buffer][local_row][local_col] = 0.0f;
                    
                    // Load tile_b[next_buffer]
                    if (global_row_b < k && global_col_b < n)
                        tile_b[next_buffer][local_row][local_col] = b[global_row_b * n + global_col_b];
                    else
                        tile_b[next_buffer][local_row][local_col] = 0.0f;
                }
            }
        }

        // New 4x4 compute loop
        for (int i = 0; i < TILE_WIDTH; i++) {
            
            // Load a column-vector (4 elements) from tile_a into registers
            float a_reg[REG_TILE_ROWS];
            for (int r = 0; r < REG_TILE_ROWS; r++) {
                a_reg[r] = tile_a[current_buffer][threadIdx.y * REG_TILE_ROWS + r][i];
            }
            
            // Load a row-vector (4 elements) from tile_b into registers
            float b_reg[REG_TILE_COLS];
            for (int c_ = 0; c_ < REG_TILE_COLS; c_++) {
                int b_col_idx = threadIdx.x * REG_TILE_COLS + c_;
                b_reg[c_] = tile_b[current_buffer][i][b_col_idx];
            }

            // Perform the 4x4 outer-product and accumulate
            for (int r = 0; r < REG_TILE_ROWS; r++) {
                for (int c_ = 0; c_ < REG_TILE_COLS; c_++) {
                    sum[r][c_] += a_reg[r] * b_reg[c_];
                }
            }
        }

        //Wait for compute on t AND load on t+1 to finish
        __syncthreads(); 
    }
    
    // New Final write-back from 4x4 register tile
    for (int r = 0; r < REG_TILE_ROWS; r++) {
        int output_row = row + r;
        for (int c_ = 0; c_ < REG_TILE_COLS; c_++) {
            int output_col = col + c_;
            if (output_row < m && output_col < n)
                c[output_row * n + output_col] = sum[r][c_];
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

    dim3 block(BLOCK_THREADS_X, BLOCK_THREADS_Y); // (8, 8)
    dim3 grid((N + TILE_WIDTH - 1) / TILE_WIDTH, (M + TILE_WIDTH - 1) / TILE_WIDTH);
    
    tiling_double_buffer_4x4<<<grid, block>>>(d_a, d_b, d_c, M, N, K);
    cudaDeviceSynchronize();

    
    int iterations = 100;
    float total_time = 0.0f;

    for (int i = 0; i < iterations; i++) {
        
        cudaEventRecord(start);


        tiling_double_buffer_4x4<<<grid, block>>>(d_a, d_b, d_c, M, N, K);
        
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float iter_time;
        cudaEventElapsedTime(&iter_time, start, stop);
        total_time += iter_time;
    }
    printf("Average kernel execution time (4x4 Double Buffer): %f ms\n", total_time / iterations);
    printf("Average GFLOPS (4x4 Double Buffer): %f\n", 2.0f * N * M * K / (total_time / iterations) / 1e6);

    
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
