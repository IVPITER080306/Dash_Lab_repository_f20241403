# “SYSTEMS” REPORT

This report contains an overview of my journey in understanding and applying algorithms for hardware acceleration, specifically to increase the speed of GEMM calculations. A GEMM (General Matrix Multiplication) is the process in which two multi- dimensional matrices (say dim(M*K) and dim(K*N)) are multiplied to get another matrix (dim(M*N)). This operation is crucial towards fields like AI where often the machine makes results on the basis of these GEMMs. 

Before we get to the breadth of my report, a lot of my testing was done on my personal GPU, the RTX 4060 (Ada Lovelace architecture). This GPU has a very high L1 and L2 cache along with a very large VRAM. This results in some algorithms (double buffering) being rendered obsolete.

## HARDWARE OVERVIEW

With that said, lets first touch into GPU hardware basics, the GPU, has way more computational cores compared to a CPU which is essential for doing way more parallel computations when compared to a CPU. The GPU computational hierarchy is simple, the computational cores are divided into Grids, which are sub-divided into Blocks. Each Block is then again further subdivided into threads, these threads are the smallest unit for computations.  A group of 32 threads is classified by hardware as a warp (a crucial concept for memory coalescing). The memory hierarchy is classified as follows, there is the global memory (aka the VRAM). This unit is responsible for receiving data from host (the CPU). It is very large in size (in terms of the no. of bytes) but has very high latency when transferring data. The next memory is the L2 cache which is like a backup cache for data dumped by the L1 cache. The next is shared memory (L1 cache), essentially each block is managed by an SM (streaming microprocessor), this SM includes shared memory which is very fast when compared to the VRAM. This shared memory is called “shared” because it is basically accessible to any thread of a particular block. This memory is key for some algorithmic optimizations like Tiling. The smallest and the fastest unit of memory is the registers which are only accessible to a single thread, we will be making use of this for an algorithm called Register Blocking. The entire idea behind hardware acceleration is that since a GPU has more cores than a CPU, it can do faster calculations than a CPU when the data is very large, however, to do calculations each computational unit of the GPU should have access to that data first and more often than not, the real challenge is getting faster access to data to these computational units. All the algorithms practiced in this report are about the fastest way threads can get access to data and then do the calculations.
 

Now with the hardware overview out of the way, let us get to the algorithms.

## NAÏVE IMPLEMENTATION

The first implementation is the naïve one. This implementation doesn’t use any crazy algorithms, just plain simple matrix multiplication. We do it by simple multiplying each element. The process involves generating 2 matrices on the CPU, allocating sufficient data on GPU VRAM and transferring them, doing basic matrix multiplication on them (where 1 thread does calculation for one element of the resultant matrix) and copying back the resultant matrix on the CPU. This process is quite cumbersome since each time the threads have to gather their data from the VRAM, which is quite slow (high latency). This results in this process being memory bound.

## MEMORY COALESCING

Now, one of the first problems we can solve is the way in which memory access is made parallel. Essentially, after block assignment, a group of 32 threads is assigned to a warp which because of the SM perform a single instruction. Now, when matrix multiplication takes place, we often map each row of a thread to each column of the output matrix, this results in the resultant matrix being able to access consecutive elements of one matrix but unable to access consecutive elements of other matrix, leading to loss of parallelism. One way to solve this is to map the row of a thread to row of the resultant matrix, using this technique the elements of the resultant matrix are able to access consecutive elements of both matrices. This results in what is called a coalesced read.

## TILING

Now that memory access has been solved, we can take advantage of another component of our hardware, the shared memory (L1 cache). The SM allows memory access to L1 cache to all threads of the same block. We can copy our large matrices on the VRAM, now we can make smaller matrices called tiles and allocate memory the size of tiles on this shared memory. We now copy some elements of both matrices onto these tiles, these tiles then compute the partial dot product and store it in the resultant matrix, now these tiles copy other portions of the matrices till the elements of the resultant matrix the size of the tiles is computed. This process is repeated for rest of the portions of the resultant matrix till the entire matrix is computed. Now instead of resultant matrix accessing the VRAM multiple times, which is slow, it accesses the shared memory which is faster. This improves our GEMM speed and makes it a little less memory bound.
 

## REGISTER BLOCKING

Now, we can move on in trying to access the fastest memory unit in our hardware, the registers. Basically, what we do is that we copy some of the elements of the input matrices on the shared memory, but now instead of directly computing while accessing memory from the L1 cache, we copy some of the elements of the tiles to registers of each thread and then start computations (again partial dot product). This process now enables a thread to access memory directly from its own registers for computations and enables a thread to do more than one computation which significantly increases the speed of the GEMM. There are two types of Register blocking, one where registers are used for only one dimension of the matrix (1D Register Blocking) and one where both the dimensions are given access to registers (2D Register Blocking).
 

## DOUBLE BUFFERING

This is the last optimization that I tried to apply, though with this implementation I gain partial success. Essentially this optimization is based on the fact that during GEMM, you constantly have to access and re-access old elements from the input matrices, this causes some unnecessary latency in the element calculations. To solve this issue, essentially what can be done is to make two buffers. Essentially one buffer will be busy doing the computations, while the other will be busy loading up data for the next set of calculations. These buffers use shared memory for fast memory access.

This method however resulted in a faster GEMM on Google Collab’s T4 GPU but a slower GEMM on my personal GPU (RTX 4060) (Ada Lovelace Architecture). The reason is because of my personal GPU’s L2 cache (the cache responsible for storing dumped elements form the L1 cache) which is far superior to T4’s L2 cache. The T4 has a 6MB L2 cache whereas my GPU has a 24MB cache. Essentially my GPU is able to store more data, and it is strictly more efficient to access my L2 cache data as compared to the L1 cache data (as it most likely has been dumped before) resulting in faster GEMM without double buffering on my GPU. Though its still important to note that this technique still works on lower L2 cache GPUs like the T4.

---

## Benchmark Results

### 📊 Performance Data Table

Here is the combined performance data from both the RTX 4060 (Ada Lovelace) and the T4 (Turing) GPUs. All tests were for a 1024x1024 SGEMM.

| Implementation | RTX 4060 (Time) | RTX 4060 (GFLOPS) | RTX 4060 (% of cuBLAS) | T4 (Time) | T4 (GFLOPS) | T4 (% of cuBLAS) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Naive** | 2.8969 ms | 741.3 | **12.4%** | 5.802080 ms | 370.1 | **14.4%** |
| **Tiling** | 2.4010 ms | 894.4 | **14.9%** | 3.605190 ms | 595.7 | **23.2%** |
| **Register Blocking** | 1.3681 ms | 1569.7 | **26.2%** | 3.487198 ms | 615.8 | **24.0%** |
| **Double Buffering** | 1.1581 ms | 1854.4 | **30.9%** | 2.929112 ms | 733.2 | **28.5%** |
| **Double Buffering v2** | 0.8754 ms | 2453.2 | **40.9%** | **2.009183 ms** | **1068.8** | **41.6%** |
| **v3 Optimised** | **0.8193 ms** | **2621.1** | **43.7%** | 2.339632 ms | 917.9 | **35.7%** |
| **cuBLAS** | **0.3579 ms** | **6000.1** | **100.0%** | **0.836040 ms** | **2568.6** | **100.0%** |

### 📈 Performance Graph

![Benchmark GFLOPS Graph](https://raw.githubusercontent.com/IVPITER080306/Dash_Lab_repository_f20241403/8562bc3dd41377ecddf1d39bd2c49c520350297d/Part_1/GPU%20plot.png)
