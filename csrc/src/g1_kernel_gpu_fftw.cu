/*
 * Copyright 1993-2017 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/* Example showing the use of CUFFT for fast 1D-convolution using FFT. */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sstream>

// includes, project
#include <cuda_runtime.h>
#include <cufft.h>
#include <helper_cuda.h>

#include "g1_kernel.hpp"
#include "logging.hpp"

/**
 * As an intermediate step, perform squaring of the fourier transform.
 * @param gpu_variance array `[CHAG1, CHBG1, SQG1]` for normalisation
 * @param index selects the variance to use
 **/
static __global__ void fftw_square(cufftComplex *fourier_transform,
                                   float *gpu_variance,
                                   unsigned int index) {

    const int step = blockDim.x * gridDim.x;
    const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    int mid_point = (G1_DIGITISER_POINTS / 2) + 1;
    for (int i = thread_id; i < mid_point; i += step) {
        fourier_transform[i].x = (
            fourier_transform[i].x * fourier_transform[i].x +
            fourier_transform[i].y * fourier_transform[i].y) / (
                gpu_variance[index] * G1_DIGITISER_POINTS * G1_DIGITISER_POINTS);
        fourier_transform[i].y = 0;
    }
}

void G1::GPU::g1_kernel(
    short *chA_data, short *chB_data,
    cufftReal **gpu_inout, float **cpu_inout,
    short **gpu_raw_data, float **gpu_pp_aux, cufftComplex **gpu_fftw_aux, float *gpu_mean, float *gpu_variance,
    cufftHandle *plans_forward, cufftHandle *plans_backward){

    // Normalise input arrays
    G1::GPU::preprocessor(
        G1_DIGITISER_POINTS, chA_data, chB_data,
        gpu_raw_data, reinterpret_cast<float**>(gpu_inout),
        gpu_pp_aux, gpu_mean, gpu_variance);

    for (int i(0); i < G1::no_outputs; i++) {
        CUDA_CHECK(
            cufftExecR2C(
                plans_forward[i],
                gpu_inout[i], gpu_fftw_aux[i]),
            "G1 Kernel: Failed forward transform.");
        fftw_square<<<(unsigned long long)G1_DIGITISER_POINTS / 1024 + 1,1024>>>(
            gpu_fftw_aux[i],
            gpu_variance,
            i
            );
        CUDA_CHECK(
            cufftExecC2R(
                plans_backward[i], gpu_fftw_aux[i], gpu_inout[i]),
            "G1 Kernel: Failed backward transform."
            );
        CUDA_CHECK(
            cudaMemcpy(cpu_inout[i], gpu_inout[i],
                       sizeof(cufftReal) * G1_DIGITISER_POINTS,
                       cudaMemcpyDeviceToHost),
            "G1 Kernel: Failed to copy data to CPU.");
    }
}
