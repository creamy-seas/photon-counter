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
 * Instead of delcaring shared memory like this:
 *
 *     extern __shared__ T shared[]; // Size of shared array depends on parameter passed in at kernel launch.
 *
 * this is done with an auxillary class used to avoid linker errors with extern
 * unsized shared memory arrays with templated type.
 */
template<typename T>
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

/**
 * Specialize for double to avoid unaligned memory access compile errors
 */
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
 * Whereas `warp_reduce_sum` reduces values in threads running in a single warp (32 threads)
 * this function recues all 1024 threads (32 warps) within a single block
 */
template <typename T>
__inline__ __device__
T block_reduce_sum(T val) {
    T *shared = SharedMemory<T>();

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
 * Summation of `gpu_in` into `gpu_out` with optional normalisation.
 *
 * `gpu_in` is subdivided into blocks:
 * [b1, b1, b1, b1, ..., b2, b2, b2, b2, ...]
 *
 * The length of `gpu_out` must be at least as large as the number of blocks
 * the kernel was called with in order to retain the values reduced in each of the blocks:
 * [block1sum, block2sum, block3sum, ...]
 *
 * **This will normally be used as a second invocation after `mean_kernel` or `variance_and_normalisation_kernel`
 *  in order to reduce their temporary outputs.**
 */
template <typename Tin, typename Tout>
__global__ void reduction_kernel(
    Tin *gpu_in_chA, Tin *gpu_in_chB,
    Tout *gpu_out,
    unsigned int N,
    Tout normalisation
    ) {

    Tout chA_sum(0), chB_sum(0);//, sq_sum(0);
    // Copy over input data to `sum` that is local to this thread.
    for (int tidx = blockIdx.x * blockDim.x + threadIdx.x;
         tidx < N;
         tidx += blockDim.x * gridDim.x) {
        chA_sum += (Tout)gpu_in_chA[tidx];
        chB_sum += (Tout)gpu_in_chB[tidx];
    }

    // Perform reduction
    chA_sum = block_reduce_sum<Tout>(chA_sum);
    chB_sum = block_reduce_sum<Tout>(chB_sum);

    // With the value reduced in the first thread, normalise and output
    if (threadIdx.x == 0) {
        gpu_out[CHAG1] = chA_sum / normalisation;
        gpu_out[CHBG1] = chB_sum / normalisation;
    }
}

/**
 * Based on the reduction kernel, but with:
 * - no normalisation
 * - unique summation for the square channel.
 * Returns sum in the individual blocks: [block1sum, block2sum, block3sum, ...] that need to be reduced one more time.
 */
template <typename Tin, typename Tout>
__global__ void mean_kernel(
    Tin *gpu_in_chA, Tin *gpu_in_chB,
    Tout *gpu_out_chA, Tout *gpu_out_chB,
    unsigned int N
    ) {

    Tout chA_sum(0), chB_sum(0);//, sq_sum(0);
    // Copy over input data to `sum` that is local to this thread.
    for (int tidx = blockIdx.x * blockDim.x + threadIdx.x;
         tidx < N;
         tidx += blockDim.x * gridDim.x) {
        chA_sum += (Tout)gpu_in_chA[tidx];
        chB_sum += (Tout)gpu_in_chB[tidx];
    }

    // Perform reduction
    chA_sum = block_reduce_sum<Tout>(chA_sum);
    chB_sum = block_reduce_sum<Tout>(chB_sum);

    // With the value reduced in the first thread, normalise and output
    if (threadIdx.x == 0) {
        gpu_out_chA[blockIdx.x] = chA_sum;
        gpu_out_chB[blockIdx.x] = chB_sum;
    }
}

/**
 * Alternative copying of data compared to the reduction_kernel with:
 * - Normalisation of input data by the mean
 * - Storage of the mean square deviation in order to compute the variance.
 * Returns sum in the individual blocks: [block1sum, block2sum, block3sum, ...] that need to be reduced one more time.
 */
template <typename Tin, typename Tout>
__global__ void variance_and_normalisation_kernel(
    Tin *gpu_in_chA, Tin *gpu_in_chB,
    Tout *gpu_out_chA, Tout *gpu_out_chB,
    Tout *gpu_normalised_chA, Tout *gpu_normalised_chB,
    Tout *gpu_mean_list,
    unsigned int N
    ) {

    Tout chA_sum(0), chB_sum(0);//, sq_sum(0);

    // Normalise by the mean and seed the `sum` for this thread
    // in order to sum up the square differences for evaluation of the variance.
    for (int tidx = blockIdx.x * blockDim.x + threadIdx.x;
         tidx < N;
         tidx += blockDim.x * gridDim.x) {
        gpu_normalised_chA[tidx] = (Tout)gpu_in_chA[tidx] - gpu_mean_list[CHAG1];
        gpu_normalised_chB[tidx] = (Tout)gpu_in_chB[tidx] - gpu_mean_list[CHBG1];

        chA_sum += gpu_normalised_chA[tidx] * gpu_normalised_chA[tidx];
        chB_sum += gpu_normalised_chB[tidx] * gpu_normalised_chB[tidx];
    }

    // Perform reduction
    chA_sum = block_reduce_sum<Tout>(chA_sum);
    chB_sum = block_reduce_sum<Tout>(chB_sum);

    // With the value reduced in the first thread, normalise and output
    if (threadIdx.x == 0){
        gpu_out_chA[blockIdx.x] = chA_sum;
        gpu_out_chB[blockIdx.x] = chB_sum;
    }
}

template <typename T>
void G1::GPU::preprocessor(int N,
                           short *chA_data, short *chB_data,
                           T *mean_list, T *variance_list,
                           T **normalised_data){
    const unsigned long blocks = (N + G1::GPU::pp_threads - 1) / G1::GPU::pp_threads;

    int success = 0;
    short *gpu_in_chA, *gpu_in_chB;
    T *gpu_out_chA, *gpu_out_chB;
    T *gpu_normalised_chA, *gpu_normalised_chB;
    T *gpu_mean_list; T *gpu_variance_list;

    success += cudaMalloc(reinterpret_cast<void**>(&gpu_in_chA), N * sizeof(short));
    success += cudaMalloc(reinterpret_cast<void**>(&gpu_in_chB), N * sizeof(short));
    success += cudaMalloc(reinterpret_cast<void**>(&gpu_out_chA), blocks * sizeof(T));
    success += cudaMalloc(reinterpret_cast<void**>(&gpu_out_chB), blocks * sizeof(T));
    success += cudaMalloc(reinterpret_cast<void**>(&gpu_normalised_chA), N * sizeof(T));
    success += cudaMalloc(reinterpret_cast<void**>(&gpu_normalised_chB), N * sizeof(T));
    success += cudaMalloc(reinterpret_cast<void**>(&gpu_mean_list), G1::no_outputs * sizeof(T));
    success += cudaMalloc(reinterpret_cast<void**>(&gpu_variance_list), G1::no_outputs * sizeof(T));
    if (success != 0) FAIL("Failed memory allocation");

    // Copy input data
    cudaMemcpy(gpu_in_chA, chA_data, N * sizeof(short), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_in_chB, chB_data, N * sizeof(short), cudaMemcpyHostToDevice);

    // Evaluation of mean
    mean_kernel
        <<<blocks, G1::GPU::pp_threads, G1::GPU::pp_shared_memory>>>
        (gpu_in_chA, gpu_in_chB, gpu_out_chA, gpu_out_chB, N);
    reduction_kernel
        <<<1, G1::GPU::pp_threads, G1::GPU::pp_shared_memory>>>
        (gpu_out_chA, gpu_out_chB, gpu_mean_list, blocks, (T)N);

    // Evaluation of variance and normalisation
    variance_and_normalisation_kernel
        <<<blocks, G1::GPU::pp_threads, G1::GPU::pp_shared_memory>>>
        (gpu_in_chA, gpu_in_chB, gpu_out_chA, gpu_out_chB, gpu_normalised_chA, gpu_normalised_chB, gpu_mean_list, N);
    reduction_kernel
        <<<1, G1::GPU::pp_threads, G1::GPU::pp_shared_memory>>>
        (gpu_out_chA, gpu_out_chB, gpu_variance_list, blocks, (T)N);

    cudaMemcpy(mean_list, gpu_mean_list, G1::no_outputs * sizeof(T), cudaMemcpyDeviceToHost);
    cudaMemcpy(variance_list, gpu_variance_list, G1::no_outputs * sizeof(T), cudaMemcpyDeviceToHost);
    cudaMemcpy(normalised_data[CHAG1], gpu_normalised_chA, N * sizeof(T), cudaMemcpyDeviceToHost);
    cudaMemcpy(normalised_data[CHBG1], gpu_normalised_chB, N * sizeof(T), cudaMemcpyDeviceToHost);
}

template void G1::GPU::preprocessor<float>(
    int N,
    short *chA_data, short *chB_data,
    float *mean_list, float *variance_list,
    float **normalised_data);
template void G1::GPU::preprocessor<double>(
    int N,
    short *chA_data, short *chB_data,
    double *mean_list, double *variance_list,
    double **normalised_data);
