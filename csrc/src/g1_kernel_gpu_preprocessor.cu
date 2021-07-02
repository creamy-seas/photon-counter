/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/*
  Parallel reduction kernels
*/
#include <stdio.h>
#include <cufft.h>

#include <iostream>

#include "logging.hpp"
#include "g1_kernel.hpp"

#ifndef MIN
#define MIN(x,y) ((x < y) ? x : y)
#endif

#define WARPSIZE 32

unsigned int nextPow2(unsigned int x)
{
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return ++x;
}

/**
 * The 32 threads within a single warp are reduced to the first lane (thread0), by succesively copying from threads at a given offset
 * See https://developer.nvidia.com/blog/faster-parallel-reductions-kepler/ for diagram of procedure.
 *
 * Here `val` was the original value held by the thread.
 * - Threads with higher indicies have their values successively added onto this thread's value.
 * - By supplying `val` in the `__shfl_down_sync` command, this threads value is exposed for shuffling to other threads as well.
 */
__inline__ __device__
int warp_reduce_sum(int val) {
    for (int offset = warpSize/2; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

/**
 * Whereas `warp_reduce_sum` reduces for warpSize (32 threads) within a single warp
 * this function recues all 1024 (maximum) threads within a single block
 */
__inline__ __device__
int block_reduce_sum(int val) {
    // TODO: Scale depending on number of threads launched
    extern __shared__ int shared[];

    unsigned int wid = threadIdx.x / warpSize; // Warp index
    unsigned int lane = threadIdx.x % warpSize; // Lane that the current thread occupies in that warp

    // Reduce the 32 threads in each warp (into lane 0) and store it in shared memory under the relevant warp index.
    val = warp_reduce_sum(val);
    if (lane == 0) shared[wid] = val;
    // Await all threads to complete reduction
    __syncthreads();

    // At this stage the shared array is `[warp0sum, warp1sum, warp2sum, ..., (Number of threads / warpsize)sum]`
    // which need to be reduced again (into lane 0).
    // First check if the thread will be involved in this reduction (assign 0 otherwise)
    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;

    // Only need to utilise threads from the first warp for this final summation
    // ! This assumes that one warp (32 threads) will be larger enough for the shared array size.
    if (wid == 0) val = warp_reduce_sum(val);
    return val;
}

/**
 * Entrypoint to `block_reduce_sum` for reducing for all threads (max 1024) in a single block,
 */
__global__
void reduce_kernel(int *gpu_in, int *gpu_out, unsigned int N) {
    int sum;

    // Copy over input data. Thread is reutilised multiple times
    for (int tidx = blockIdx.x * blockDim.x + threadIdx.x;
         tidx < N;
         tidx += blockDim.x * gridDim.x) {
        sum += gpu_in[tidx];
    }

    // Perform reduction
    sum = block_reduce_sum(sum);
    if (threadIdx.x == 0)
        gpu_out[blockIdx.x]=sum;
}


/* This reduction interleaves which threads are active by using the modulo
   operator.  This operator is very expensive on GPUs, and the interleaved
   inactivity means that no whole warps are active, which is also very
   inefficient */
__global__ void reduce0(
    // short *chA_data, short *chB_data,
    // int *chA_normalised, int *chB_normalised, int *sq_normalised
    ) {
    // Given N points for processing, they are split into 1024 THREADS and (N + 1024) / 1024 BLOCKS

    // 1024 threads will be launched for each block. This means 1024/32 = 32 warps (each with )


    // // Populate shared memory
    // unsigned int tid = threadIdx.x;
    // unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // unsigned int i = threadIdx.x % 32;
    // unsigned int j = __shfl_down_sync(i, 2);


    // chA_shared[tid] = (i < G1_DIGITISER_POINTS) ? (int)chA_data[i] : 0;
    // chB_shared[tid] = (i < G1_DIGITISER_POINTS) ? (int)chA_data[i] : 0;
    // sq_shared[tid] = (i < G1_DIGITISER_POINTS) ? (int)chA_data[i] * chA_data[i] + chB_data[i] * chB_data[i] : 0;
    // __syncthreads();

    // // Perform reduction
    // for (unsigned int s=1; s < blockDim.x; s *= 2)
    // {
    //     if ((tid % (2*s)) == 0) {
    //         chA_shared[tid] += chA_shared[tid + s];
    //         chB_shared[tid] += chB_shared[tid + s];
    //         sq_shared[tid] += sq_shared[tid + s];
    //     }
    //     __syncthreads();
    // }

    // // Write result to global memory
    // if (tid == 0)
    // {
    //     chA_normalised[blockIdx.x] = chA_shared[0] / G1_DIGITISER_POINTS;
    //     chB_normalised[blockIdx.x] = chA_shared[0] / G1_DIGITISER_POINTS;
    //     sq_normalised[blockIdx.x] = sq_shared[0] / G1_DIGITISER_POINTS;
    // }
}

////////////////////////////////////////////////////////////////////////////////
// Compute the number of threads and blocks to use for the given reduction kernel
// For the kernels >= 3, we set threads / block to the minimum of maxThreads and
// n/2. For kernels < 3, we set to the minimum of maxThreads and n.  For kernel
// 6, we observe the maximum specified number of blocks, because each thread in
// that kernel can process a variable number of elements.
////////////////////////////////////////////////////////////////////////////////
void getNumBlocksAndThreads(int whichKernel, int n, int maxBlocks, int maxThreads, int &blocks, int &threads) {

    //get device capability, to avoid block/grid size exceed the upper bound
    cudaDeviceProp prop;
    int device;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);

    if (whichKernel < 3)
    {
        threads = (n < maxThreads) ? nextPow2(n) : maxThreads;
        blocks = (n + threads - 1) / threads;
    }
    else
    {
        threads = (n < maxThreads*2) ? nextPow2((n + 1)/ 2) : maxThreads;
        blocks = (n + (threads * 2 - 1)) / (threads * 2);
    }

    if ((int)threads*blocks > (int)prop.maxGridSize[0] * prop.maxThreadsPerBlock)
    {
        printf("n is too large, please choose a smaller number!\n");
    }

    if (blocks > prop.maxGridSize[0])
    {
        printf("Grid size <%d> exceeds the device capability <%d>, set block size as %d (original %d)\n",
               blocks, prop.maxGridSize[0], threads*2, threads);

        blocks /= 2;
        threads *= 2;
    }

    if (whichKernel == 6)
    {
        blocks = MIN(maxBlocks, blocks);
    }
}

void G1::GPU::preprocessor(
    short *chA, short *chB,
    double *mean_list, double *variance_list,
    double **normalised_data){

    int *gpu_in, *gpu_out;
    cudaMalloc(reinterpret_cast<void**>(&gpu_in), G1_DIGITISER_POINTS * sizeof(int));
    cudaMalloc(reinterpret_cast<void**>(&gpu_out), G1_DIGITISER_POINTS * sizeof(int));

    int threads = 1024;
    int blocks = min((G1_DIGITISER_POINTS + threads - 1) / threads, 1024);

    reduce_kernel<<<blocks, threads, threads/32>>>(gpu_in, gpu_out, G1_DIGITISER_POINTS);
    reduce_kernel<<<1, 1024, threads/32>>>(gpu_out, gpu_out, blocks);

    // int threads, blocks;
    // getNumBlocksAndThreads(0, 262144, 2147483647, 1024, blocks, threads);
    // printf("threads %i, blocks %i\n", threads, blocks);
    // getNumBlocksAndThreads(1, 262144, 2147483647, 1024, blocks, threads);
    // printf("threads %i, blocks %i\n", threads, blocks);
    // getNumBlocksAndThreads(2, 262144, 2147483647, 1024, blocks, threads);
    // printf("threads %i, blocks %i\n", threads, blocks);
    // getNumBlocksAndThreads(3, 262144, 2147483647, 1024, blocks, threads);
    // printf("threads %i, blocks %i\n", threads, blocks);
    // getNumBlocksAndThreads(4, 262144, 2147483647, 1024, blocks, threads);
    // printf("threads %i, blocks %i\n", threads, blocks);
    // getNumBlocksAndThreads(5, 262144, 2147483647, 1024, blocks, threads);
    // printf("threads %i, blocks %i\n", threads, blocks);
    // getNumBlocksAndThreads(6, 262144, 2147483647, 1024, blocks, threads);
    // printf("threads %i, blocks %i\n", threads, blocks);


    // getNumBlocksAndThreads(1, N, 2147483647, 1024, blocks, threads);
    // printf("threads %i, blocks %i\n", threads, blocks);

    // getNumBlocksAndThreads(2, N, 2147483647, 1024, blocks, threads);
    // printf("threads %i, blocks %i\n", threads, blocks);

    // getNumBlocksAndThreads(3, N, 2147483647, 1024, blocks, threads);
    // printf("threads %i, blocks %i\n", threads, blocks);

    // getNumBlocksAndThreads(4, N, 2147483647, 1024, blocks, threads);
    // printf("threads %i, blocks %i\n", threads, blocks);

    // getNumBlocksAndThreads(5, N, 2147483647, 1024, blocks, threads);
    // printf("threads %i, blocks %i\n", threads, blocks);

    // getNumBlocksAndThreads(6, N, 2147483647, 1024, blocks, threads);
    // printf("threads %i, blocks %i\n", threads, blocks);

    // dim3 dimBlock(threads, 1, 1);
    // dim3 dimGrid(blocks, 1, 1);

    // when there is only one warp per block, we need to allocate two warps
    // worth of shared memory so that we don't index shared memory out of bounds
    // int smemSize = (threads <= 32) ? 2 * threads * sizeof(T) : threads * sizeof(T);

    // Copy data to GPU

    // Run kernel
    // reduce0<<< dimGrid, dimBlock, smemSize >>>(, d_odata, size);
    // reduce0<<<1, 1>>>();


    // Normalise output
}
