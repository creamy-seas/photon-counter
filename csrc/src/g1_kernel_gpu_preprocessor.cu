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
#include <helper_cuda.h> // for CUDA_CHECK

// TODO: remove
#include <iostream>

#include "logging.hpp"
#include "g1_kernel.hpp"
#include "utils.hpp"

#ifndef MIN
#define MIN(x,y) ((x < y) ? x : y)
#endif

#define WARPSIZE 32
#define FULL_MASK 0xffffffff

/**
 * The 32 threads within a single warp are reduced to the first lane (thread0), by succesively copying from threads at a given offset
 * See https://developer.nvidia.com/blog/faster-parallel-reductions-kepler/ for diagram of procedure.
 *
 * Here `val` was the original value held by the thread.
 * - Threads with higher indicies have their values successively added onto this thread's value.
 * - By supplying `val` in the `__shfl_down_sync` command, this threads value is exposed for shuffling to other threads as well.
 */
template <typename T>
__inline__ __device__
T warp_reduce_sum(T val) {
    for (int offset = warpSize/2; offset > 0; offset /= 2)
        val += __shfl_down_sync(FULL_MASK, val, offset);
    return val;
}

/**
 * Whereas `warp_reduce_sum` reduces for warpSize (32 threads) within a single warp
 * this function recues all 1024 (maximum) threads within a single block
 */
template <typename T>
__inline__ __device__
T block_reduce_sum(T val) {
    // TODO: Scale depending on number of threads launched
    extern __shared__ T shared[]; // Size of shared array depends on parameter passed in at kernel launch.

    // unsigned int tidx = threadIdx.x;
    // unsigned int i = (2 * blockDim.x) * blockIdx.x + threadIdx.x; // Double the block size is used

    unsigned int wid = threadIdx.x / warpSize; // Warp index
    unsigned int lane = threadIdx.x % warpSize; // Lane that the current thread occupies in that warp

    // Reduce the 32 threads in each warp (into lane 0) and store it in shared memory under the relevant warp index.
    val = warp_reduce_sum<T>(val);
    if (lane == 0) shared[wid] = val;
    // Await all threads to complete reduction
    __syncthreads();

    // At this stage the shared array is `[warp0sum, warp1sum, warp2sum, ..., (Number of threads / warpsize)sum]`
    // which need to be reduced again (into lane 0).
    // First check if the thread will be involved in this reduction (assign 0 otherwise)
    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;

    // Only need to utilise threads from the first warp for this final summation
    // ! This assumes that one warp (32 threads) will be larger enough for the shared array size.
    if (wid == 0) val = warp_reduce_sum<T>(val);
    return val;
}

/**
 * Entrypoint to `block_reduce_sum` for reducing for all threads (max 1024) in a single block,
 */
template <typename Tin, typename Tout>
__global__ void reduce_kernel(
    Tin *gpu_in, Tout *gpu_out,
    unsigned int N,
    Tout normalisation
    ) {
    Tout sum(0);

    // Copy over input data. Thread is reutilised multiple times
    for (int tidx = blockIdx.x * blockDim.x + threadIdx.x;
         tidx < N;
         tidx += blockDim.x * gridDim.x) {
        sum += (Tout)gpu_in[tidx];
    }

    // Perform reduction
    sum = block_reduce_sum<Tout>(sum);
    if (threadIdx.x == 0)
        gpu_out[blockIdx.x] = sum / normalisation;
}

void G1::GPU::preprocessor(int N,
                           short *chA_data, short *chB_data,
                           float *mean_list, float *variance_list,
                           float **normalised_data){
    int success = 0;

    short *gpu_in;
    float *gpu_out;
    // float *gpu_mean_list;
    success += cudaMalloc(reinterpret_cast<void**>(&gpu_in), N * sizeof(short));
    success += cudaMalloc(reinterpret_cast<void**>(&gpu_out), N * sizeof(float));
    // success += cudaMalloc(reinterpret_cast<void**>(&gpu_mean_list), G1::no_outputs * sizeof(float));
    if (success != 0) FAIL("Failed memory allocation");

    // Copy input data
    cudaMemcpy(gpu_in, chA_data, N * sizeof(short), cudaMemcpyHostToDevice);

    const unsigned long blocks = (N + G1::GPU::pp_threads - 1) / G1::GPU::pp_threads;
    // Evaluation of mean
    reduce_kernel<<<blocks, G1::GPU::pp_threads, G1::GPU::pp_shared_memory>>>(gpu_in, gpu_out, N, (float)1);
    reduce_kernel<<<1, G1::GPU::pp_threads, G1::GPU::pp_shared_memory>>>(gpu_out, gpu_out, blocks, (float)N);

    cudaMemcpy(&mean_list[CHAG1], gpu_out, sizeof(float), cudaMemcpyDeviceToHost);
}
