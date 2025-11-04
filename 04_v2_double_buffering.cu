#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>

#define TILE_WIDTH 32
// We pad the column dimension of shared memory tiles.
#define TILE_WIDTH_PADDED (TILE_WIDTH + 1)

// Each thread computes a 4x4 tile of C
#define REG_TILE_ROWS 4 
#define REG_TILE_COLS 4 

// Block dimensions: 8x8 threads

#define BLOCK_THREADS_X 8 
#define BLOCK_THREADS_Y 8
// Total 64 threads per block

__global__ void tiling_double_buffer_4x4(float *a, float *b, float *c, int m, int n, int k) 
{

    __shared__ float tile_a[2][TILE_WIDTH][TILE_WIDTH_PADDED];
    __shared__ float tile_b[2][TILE_WIDTH][TILE_WIDTH_PADDED];

    // This tile is used *after* the main loop to coalesce writes.
    // It does not need padding as it's written/read with 1D coalesced logic.
    __shared__ float s_c[TILE_WIDTH][TILE_WIDTH];

    
    // New 4x4 array (16 elements) to hold intermediate sums in registers.
    float sum[REG_TILE_ROWS][REG_TILE_COLS];
    for (int r = 0; r < REG_TILE_ROWS; r++) {
        for (int c_ = 0; c_ < REG_TILE_COLS; c_++) {
            sum[r][c_] = 0.0f;
        }
    }


    int thread_idx_1d = threadIdx.y * blockDim.x + threadIdx.x; // 0-63
    int block_size = blockDim.x * blockDim.y; // 64

    int a_start_row_global = blockIdx.y * TILE_WIDTH;
    int b_start_col_global = blockIdx.x * TILE_WIDTH;

    int num_tiles = (k + TILE_WIDTH - 1) / TILE_WIDTH;


    // This is the fast 1D load. Each thread loads 16 elements.
    {
        int t = 0; 
        int a_start_col = t * TILE_WIDTH;
        int b_start_row = t * TILE_WIDTH;

        for (int i = 0; i < 16; i++) { // 16 elements per thread (16 * 64 = 1024)
            int tile_idx = thread_idx_1d + i * block_size;
            int load_row = tile_idx / TILE_WIDTH; // Row within the 32x32 tile
            int load_col = tile_idx % TILE_WIDTH; // Col within the 32x32 tile

            int global_row_a = a_start_row_global + load_row;
            int global_col_a = a_start_col + load_col;
            int global_row_b = b_start_row + load_row;
            int global_col_b = b_start_col_global + load_col;

            // Write to shared memory using the *padded* column dimension
            if (global_row_a < m && global_col_a < k)
                tile_a[0][load_row][load_col] = a[global_row_a * k + global_col_a];
            else
                tile_a[0][load_row][load_col] = 0.0f;
            
            if (global_row_b < k && global_col_b < n)
                tile_b[0][load_row][load_col] = b[global_row_b * n + global_col_b];
            else
                tile_b[0][load_row][load_col] = 0.0f;
        }
    }

    __syncthreads(); // Wait for the first tile (t=0) to be loaded


    // This loop is now branch-free. It runs (num_tiles - 1) times.
    for (int t = 0; t < num_tiles - 1; t++) {
        
        int current_buffer = t % 2;
        int next_buffer = (t + 1) % 2;


        // This logic is now *unconditional*
        {
            int next_tile_k_a = (t + 1) * TILE_WIDTH; // Col for A
            int next_tile_k_b = (t + 1) * TILE_WIDTH; // Row for B
            
            for (int i = 0; i < 16; i++) { // 16 elements per thread
                int tile_idx = thread_idx_1d + i * block_size;
                int load_row = tile_idx / TILE_WIDTH; 
                int load_col = tile_idx % TILE_WIDTH; 

                int global_row_a = a_start_row_global + load_row; 
                int global_col_a = next_tile_k_a + load_col; 
                int global_row_b = next_tile_k_b + load_row; 
                int global_col_b = b_start_col_global + load_col; 

                // Load tile_a[next_buffer]
                if (global_row_a < m && global_col_a < k)
                    tile_a[next_buffer][load_row][load_col] = a[global_row_a * k + global_col_a];
                else
                    tile_a[next_buffer][load_row][load_col] = 0.0f;
                
                // Load tile_b[next_buffer]
                if (global_row_b < k && global_col_b < n)
                    tile_b[next_buffer][load_row][load_col] = b[global_row_b * n + global_col_b];
                else
                    tile_b[next_buffer][load_row][load_col] = 0.0f;
            }
        }

        #pragma unroll
        for (int i = 0; i < TILE_WIDTH; i++) {
            
            // Load a column-vector (4 elements) from tile_a into registers
            // Accessing [i] on the padded dimension is now bank-conflict-free.
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


    // We've finished loading, now we just compute on the last tile that was
    // loaded in the final iteration of the loop above.
    int last_buffer = (num_tiles - 1) % 2;
    #pragma unroll
    for (int i = 0; i < TILE_WIDTH; i++) {
        
        float a_reg[REG_TILE_ROWS];
        for (int r = 0; r < REG_TILE_ROWS; r++) {
            a_reg[r] = tile_a[last_buffer][threadIdx.y * REG_TILE_ROWS + r][i];
        }
        
        float b_reg[REG_TILE_COLS];
        for (int c_ = 0; c_ < REG_TILE_COLS; c_++) {
            int b_col_idx = threadIdx.x * REG_TILE_COLS + c_;
            b_reg[c_] = tile_b[last_buffer][i][b_col_idx];
        }

        for (int r = 0; r < REG_TILE_ROWS; r++) {
            for (int c_ = 0; c_ < REG_TILE_COLS; c_++) {
                sum[r][c_] += a_reg[r] * b_reg[c_];
            }
        }
    }

    
    // Step 1: Write results from 4x4 registers (sum) to shared memory (s_c)
    // This write uses the *compute-mapping* (uncoalesced)
    for (int r = 0; r < REG_TILE_ROWS; r++) {
        for (int c_ = 0; c_ < REG_TILE_COLS; c_++) {
            int local_row = threadIdx.y * REG_TILE_ROWS + r;
            int local_col = threadIdx.x * REG_TILE_COLS + c_;
            // We use the un-padded s_c tile here.
            s_c[local_row][local_col] = sum[r][c_];
        }
    }

    __syncthreads(); // Wait for all sums to be in shared memory

    // Step 2: Read from shared memory and write to global memory (c)
    // This write uses the 1D *coalesced-mapping*
    for (int i = 0; i < 16; i++) { // Each thread writes 16 elements
        int tile_idx = thread_idx_1d + i * block_size;
        int s_row = tile_idx / TILE_WIDTH; // Row in shared mem
        int s_col = tile_idx % TILE_WIDTH; // Col in shared mem

        int g_row = blockIdx.y * TILE_WIDTH + s_row;
        int g_col = blockIdx.x * TILE_WIDTH + s_col;

        if (g_row < m && g_col < n) {
            c[g_row * n + g_col] = s_c[s_row][s_col];
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
    // Grid calculation is correct.
    dim3 grid((N + TILE_WIDTH - 1) / TILE_WIDTH, (M + TILE_WIDTH - 1) / TILE_WIDTH);
    
    // Warm-up run
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
    printf("Average kernel execution time (Max Optimized): %f ms\n", total_time / iterations);
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

