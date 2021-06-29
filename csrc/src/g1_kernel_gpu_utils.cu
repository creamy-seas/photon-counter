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

#include "g1_kernel.hpp"

#ifndef MIN
#define MIN(x,y) ((x < y) ? x : y)
#endif

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

// Utility class used to avoid linker errors with extern
// unsized shared memory arrays with templated type
template<class T>
struct SharedMemory
{
    __device__ inline operator       T *()
        {
            extern __shared__ int __smem[];
            return (T *)__smem;
        }

    __device__ inline operator const T *() const
        {
            extern __shared__ int __smem[];
            return (T *)__smem;
        }
};

// specialize for double to avoid unaligned memory
// access compile errors
template<>
struct SharedMemory<double>
{
    __device__ inline operator       double *()
        {
            extern __shared__ double __smem_d[];
            return (double *)__smem_d;
        }

    __device__ inline operator const double *() const
        {
            extern __shared__ double __smem_d[];
            return (double *)__smem_d;
        }
};

/* This reduction interleaves which threads are active by using the modulo
   operator.  This operator is very expensive on GPUs, and the interleaved
   inactivity means that no whole warps are active, which is also very
   inefficient */
template <class TIN, class TOUT>
__global__ void
reduce0(TIN *data_in, TOUT *data_out, unsigned int n)
{
    TOUT *data_s = SharedMemory<TOUT>();

    // load shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    data_s[tid] = (i < n) ? (long)data_in[i] : 0;

    __syncthreads();

    // do reduction in shared mem
    for (unsigned int s=1; s < blockDim.x; s *= 2)
    {
        // modulo arithmetic is slow!
        if ((tid % (2*s)) == 0)
            data_s[tid] += data_s[tid + s];

        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0)
        data_out[blockIdx.x] = data_s[0];
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

    if ((float)threads*blocks > (float)prop.maxGridSize[0] * prop.maxThreadsPerBlock)
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

void G1::GPU::preprocessor(short *chA, short *chB, int N,
                           double *mean_list, double *variance_list,
                           double **normalised_data){

    int threads, blocks;
    getNumBlocksAndThreads(0, N, 2147483647, 1024, blocks, threads);
    printf("threads %i, blocks %i\n", threads, blocks);

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

    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid(blocks, 1, 1);

    // when there is only one warp per block, we need to allocate two warps
    // worth of shared memory so that we don't index shared memory out of bounds
    // int smemSize = (threads <= 32) ? 2 * threads * sizeof(T) : threads * sizeof(T);

    // Copy data to GPU

    // Run kernel
    // reduce0<T><<< dimGrid, dimBlock, smemSize >>>(, d_odata, size);

    // Normalise output
}
