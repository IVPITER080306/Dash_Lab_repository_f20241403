#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>

#define TILE_WIDTH 32

#define REG_TILE_WIDTH 4 

#define BLOCK_COLS 8 
#define BLOCK_ROWS 32 

__global__ void tiling_double_buffer(float *a, float *b, float *c, int m, int n, int k) 
{
    // We use two "pages" (0 and 1) for shared memory.
    // While computing on page 0, we load into page 1, and vice versa.
    __shared__ float tile_a[2][TILE_WIDTH][TILE_WIDTH];
    __shared__ float tile_b[2][TILE_WIDTH][TILE_WIDTH];

    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x * REG_TILE_WIDTH;

    float sum[REG_TILE_WIDTH];
    for (int i = 0; i < REG_TILE_WIDTH; i++) {
        sum[i] = 0.0f;
    }

    int thread_idx_1d = threadIdx.y * blockDim.x + threadIdx.x; 
    int block_size = blockDim.x * blockDim.y; 

    int a_start_row = blockIdx.y * TILE_WIDTH;
    int b_start_col = blockIdx.x * TILE_WIDTH;

    int num_tiles = (k + TILE_WIDTH - 1) / TILE_WIDTH;

    // All threads cooperate to load tile 0 into buffer 0.
    {
        int t = 0; // Tile index
        int a_start_col = t * TILE_WIDTH;
        int b_start_row = t * TILE_WIDTH;
        
        for (int i = 0; i < 4; i++) { // Each thread loads 4 elements
            int tile_idx = thread_idx_1d + i * block_size;
            int load_row = tile_idx / TILE_WIDTH; // Row within the 32x32 tile
            int load_col = tile_idx % TILE_WIDTH; // Col within the 32x32 tile

            // Load tile_a[0]
            int a_global_row = a_start_row + load_row;
            int a_global_col = a_start_col + load_col;
            if (a_global_row < m && a_global_col < k)
                tile_a[0][load_row][load_col] = a[a_global_row * k + a_global_col];
            else
                tile_a[0][load_row][load_col] = 0.0f;

            // Load tile_b[0]
            int b_global_row = b_start_row + load_row;
            int b_global_col = b_start_col + load_col;
            if (b_global_row < k && b_global_col < n)
                tile_b[0][load_row][load_col] = b[b_global_row * n + b_global_col];
            else
                tile_b[0][load_row][load_col] = 0.0f;
        }
    }
    __syncthreads(); // Wait for the first tile to be loaded

    //Main Loop with Double Buffering
    for (int t = 0; t < num_tiles; t++) {
        
        // current_buffer: buffer to compute on (loaded in previous iteration)
        // next_buffer: buffer to load into (for next iteration)
        int current_buffer = t % 2;
        int next_buffer = (t + 1) % 2;

        // Start loading the next tile while we compute on the current one.
        if (t < num_tiles - 1) {
            int next_tile_k_a = (t + 1) * TILE_WIDTH; // Col for A
            int next_tile_k_b = (t + 1) * TILE_WIDTH; // Row for B
            
            for (int i = 0; i < 4; i++) { // Each thread loads 4 elements
                int tile_idx = thread_idx_1d + i * block_size;
                int load_row = tile_idx / TILE_WIDTH; 
                int load_col = tile_idx % TILE_WIDTH; 

                // Load tile_a[next_buffer]
                int a_global_row = a_start_row + load_row;
                int a_global_col = next_tile_k_a + load_col;
                if (a_global_row < m && a_global_col < k)
                    tile_a[next_buffer][load_row][load_col] = a[a_global_row * k + a_global_col];
                else
                    tile_a[next_buffer][load_row][load_col] = 0.0f;

                // Load tile_b[next_buffer]
                int b_global_row = next_tile_k_b + load_row;
                int b_global_col = b_start_col + load_col;
                if (b_global_row < k && b_global_col < n)
                    tile_b[next_buffer][load_row][load_col] = b[b_global_row * n + b_global_col];
                else
                    tile_b[next_buffer][load_row][load_col] = 0.0f;
            }
        }

        // Compute using the data in the current_buffer
        for (int i = 0; i < TILE_WIDTH; i++) {
            float a_reg = tile_a[current_buffer][threadIdx.y][i]; 
            
            for (int j = 0; j < REG_TILE_WIDTH; j++) {
                int b_col_idx = threadIdx.x * REG_TILE_WIDTH + j; 
                float b_reg = tile_b[current_buffer][i][b_col_idx]; 
                sum[j] += a_reg * b_reg; 
            }
        }

        // This barrier ensures two things:
        // 1. The computation on tile t is finished.
        // 2. The loading of tile t+1 is finished.
        // Now we are ready to loop again, swapping buffers.
        __syncthreads(); 
    }
    
    for (int j = 0; j < REG_TILE_WIDTH; j++) {
        int output_col = col + j;
        if (row < m && output_col < n)
            c[row * n + output_col] = sum[j]; 
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
    
    // Warm-up run
    tiling_double_buffer<<<grid, block>>>(d_a, d_b, d_c, M, N, K);
    cudaDeviceSynchronize();

    
    int iterations = 100;
    float total_time = 0.0f;

    for (int i = 0; i < iterations; i++) {
        
        cudaEventRecord(start);

        tiling_double_buffer<<<grid, block>>>(d_a, d_b, d_c, M, N, K);
        
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