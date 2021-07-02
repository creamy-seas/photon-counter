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

static __global__ void fftw_square(cufftComplex *fourier_transform, float normalisation) {

    const int step = blockDim.x * gridDim.x;
    const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    int mid_point = (G1_DIGITISER_POINTS / 2) + 1;
    for (int i = thread_id; i < mid_point; i += step) {
        fourier_transform[i].x = (
            fourier_transform[i].x * fourier_transform[i].x +
            fourier_transform[i].y * fourier_transform[i].y) / normalisation;
        fourier_transform[i].y = 0;
    }
}

void G1::GPU::g1_kernel(
    short *chA_data, short *chB_data,
    cufftReal **gpu_inout, cufftComplex **gpu_aux, float **cpu_inout,
    cufftHandle *plans_forward, cufftHandle *plans_backward){

    // Normalise input arrays
    float mean_list[G1::no_outputs]; float variance_list[G1::no_outputs];
    // G1::CPU::preprocessor(chA_data, chB_data, G1_DIGITISER_POINTS, mean_list, variance_list, cpu_inout);

    // cudaStream_t *stream_list = new cudaStream_t[G1::no_outputs];
    // for (int s(0); s < G1::no_outputs; s++){
    //     stream_list[s] = cudaStream_t();
    //     cudaStreamCreate(&stream_list[s]);
    // }
    // for (int i(0); i < G1::no_outputs; i++) {
    //     CUDA_CHECK(
    //         cufftSetStream(plans_forward[i], stream_list[i]), "Failed to bind plan to stream");
    //     CUDA_CHECK(
    //         cufftSetStream(plans_backward[i], stream_list[i]), "Failed to bind plan to stream");
    // }

    for (int i(0); i < G1::no_outputs; i++) {
        CUDA_CHECK(
            cudaMemcpy(gpu_inout[i], cpu_inout[i],
                       sizeof(float) * G1_DIGITISER_POINTS, cudaMemcpyHostToDevice),
            "G1 Kernel: Failed to copy data to GPU.");
        CUDA_CHECK(
            cufftExecR2C(
                plans_forward[i],
                gpu_inout[i], gpu_aux[i]),
            "G1 Kernel: Failed forward transform.");
        fftw_square<<<(unsigned long long)G1_DIGITISER_POINTS / 1024 + 1,1024>>>(
            gpu_aux[i],
            (long)G1_DIGITISER_POINTS * G1_DIGITISER_POINTS * variance_list[i]);
        CUDA_CHECK(
            cufftExecC2R(
                plans_backward[i], gpu_aux[i], gpu_inout[i]),
            "G1 Kernel: Failed backward transform."
            );
        CUDA_CHECK(
            cudaMemcpy(cpu_inout[i], gpu_inout[i],
                       sizeof(cufftReal) * G1_DIGITISER_POINTS,
                       cudaMemcpyDeviceToHost),
            "G1 Kernel: Failed to copy data to CPU.");
    }
}
