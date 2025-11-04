    #include <stdio.h>
    #include <cuda_runtime.h>
    #include <stdlib.h>

    #define TILE_WIDTH 32

    #define REG_TILE_ROWS 4 // Each thread computes 4 elements in a row
    #define REG_TILE_COLS 4 // Each thread computes 4 elements in a columne


    #define BLOCK_COLS 8 // Number of columns per block (8 threads)
    #define BLOCK_ROWS 8 // Number of rows per block (8 threads)
    // 8x8 threads * (4x4 elements/thread) = 32x32 elements/block

    __global__ void tiling_reg_block(float *a, float *b, float *c, int m, int n, int k) 
    {
        __shared__ float tile_a[TILE_WIDTH][TILE_WIDTH];
        __shared__ float tile_b[TILE_WIDTH][TILE_WIDTH];

        // Row indexing adjusted for REG_TILE_ROWS.
        // row is the starting row for the 4x4 elements this thread will compute.
        int row = blockIdx.y * TILE_WIDTH  + threadIdx.y * REG_TILE_ROWS; 
        
        // Column indexing adjusted for REG_TILE_COLS.
        // col is the starting column for the 4x4 elements this thread will compute.
        int col = blockIdx.x * TILE_WIDTH + threadIdx.x * REG_TILE_COLS;

        
        // Array to hold intermediate sums for 4X4 elements
        // Initialize all sums to 0.0f
        float sums[REG_TILE_ROWS][REG_TILE_COLS];
        for (int i = 0; i < REG_TILE_ROWS; i++) {
            for (int j = 0; j < REG_TILE_COLS; j++) {
                sums[i][j] = 0.0f;
            }
        }
        
        // Loop over the tiles of A and B
        for (int t = 0; t < (k + TILE_WIDTH - 1) / TILE_WIDTH; t++) {
            

            // This pattern ensures coalesced reads from global memory.
            // Each thread in the 8x8 block loads a 4x4 portion of the 32x32 tile.
            // We do this by iterating 4 times (32/8=4) in each dimension.
            for (int i = 0; i < TILE_WIDTH; i += BLOCK_ROWS) { // i = 0, 8, 16, 24
                for (int j = 0; j < TILE_WIDTH; j += BLOCK_COLS) { // j = 0, 8, 16, 24
                    
                    // Shared memory indices
                    int s_row = threadIdx.y + i;
                    int s_col = threadIdx.x + j;

                    // Global indices for A
                    int g_row_a = blockIdx.y * TILE_WIDTH + s_row;
                    int g_col_a = t * TILE_WIDTH + s_col;

                    // Coalesced Read: consecutive threadIdx.x access consecutive g_col_a
                    if (g_row_a < m && g_col_a < k) {
                        tile_a[s_row][s_col] = a[g_row_a * k + g_col_a];
                    } else {
                        tile_a[s_row][s_col] = 0.0f;
                    }

                    // Global indices for B
                    int g_row_b = t * TILE_WIDTH + s_row;
                    int g_col_b = blockIdx.x * TILE_WIDTH + s_col;

                    // Coalesced Read: consecutive threadIdx.x access consecutive g_col_b
                    if (g_row_b < k && g_col_b < n) {
                        tile_b[s_row][s_col] = b[g_row_b * n + g_col_b];
                    } else {
                        tile_b[s_row][s_col] = 0.0f;
                    }
                }
            }
            
            __syncthreads();

            // Compute partial sums
            // This loop was correct in the original code.
            // Each thread computes its 4x4 sums matrix.
            for (int p = 0; p < TILE_WIDTH; p++) {
                for (int i = 0; i < REG_TILE_ROWS; i++) {
                    for (int j = 0; j < REG_TILE_COLS; j++) {
                        sums[i][j] += tile_a[threadIdx.y * REG_TILE_ROWS + i][p] * tile_b[p][threadIdx.x * REG_TILE_COLS + j];
                    }
                }
            }

            __syncthreads();
        }
        

        // Write results from registers to global memory
        // This fixes the race condition by using the 'row' and 'col'
        // variables defined at the top of the kernel.
        for (int i = 0; i < REG_TILE_ROWS; i++) {
            for (int j = 0; j < REG_TILE_COLS; j++) {
                
                int final_row = row + i; // row is this thread's starting row
                int final_col = col + j; // col is this thread's starting col
                
                if (final_row < m && final_col < n) {
                    c[final_row * n + final_col] = sums[i][j];
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
        

        dim3 block(BLOCK_COLS, BLOCK_ROWS); 
        

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
        printf("Average GFLOPS: %f\n", (2.0f * M * N * K / (total_time / iterations) / 1e6));

        
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
