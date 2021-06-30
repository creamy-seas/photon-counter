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

#include <iostream>

// includes, project
#include <cuda_runtime.h>
#include <cufft.h>
#include <helper_cuda.h>
// #include <helper_functions.h>

#include "g1_kernel.hpp"
#include "logging.hpp"

// The filter size is assumed to be a number smaller than the signal size
#define SIGNAL_SIZE 50
#define FILTER_KERNEL_SIZE 11

// Complex data type
typedef float2 Complex;

static __device__ __host__ inline Complex ComplexAdd(Complex a, Complex b) {
    Complex c;
    c.x = a.x + b.x;
    c.y = a.y + b.y;
    return c;
}
static __device__ __host__ inline Complex ComplexScale(Complex a, float s) {
    Complex c;
    c.x = s * a.x;
    c.y = s * a.y;
    return c;
}
static __device__ __host__ inline Complex ComplexMul(Complex a, Complex b) {
    Complex c;
    c.x = a.x * b.x - a.y * b.y;
    c.y = a.x * b.y + a.y * b.x;
    return c;
}
static __global__ void ComplexPointwiseMulAndScale(
    Complex *a, const Complex *b,
    int size, float scale) {

    const int step = blockDim.x * gridDim.x;
    const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = thread_id; i < size; i += step)
        a[i] = ComplexScale(ComplexMul(a[i], b[i]), scale);
}

// static __device__ __host__ inline cufftComplex complex_multiplication(cufftComplex a, cufftComplex b) {
//     Complex c;
//     c.x = a.x * b.x - a.y * b.y;
//     c.y = a.x * b.y + a.y * b.x;
//     return c;
// }
static __global__ void fftw_square(cufftComplex *fourier_transform) {

    // const int step = blockDim.x * gridDim.x;
    // const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    // for (int i = thread_id; i < G1_DIGITISER_POINTS; i += step) {
    //     fourier_transform[i].x = (
    //         fourier_transform[i].x * fourier_transform[i].x +
    //         fourier_transform[i].y * fourier_transform[i].y) + 10;
    //     fourier_transform[i].y = -20;
    // }
    fourier_transform[0].x = 100;
}

// Computes convolution on the host
void Convolve(const Complex *signal, int signal_size,
              const Complex *filter_kernel, int filter_kernel_size,
              Complex *filtered_signal) {
    int minRadius = filter_kernel_size / 2;
    int maxRadius = filter_kernel_size - minRadius;

    // Loop over output element indices
    for (int i = 0; i < signal_size; ++i) {
        filtered_signal[i].x = filtered_signal[i].y = 0;

        // Loop over convolution indices
        for (int j = -maxRadius + 1; j <= minRadius; ++j) {
            int k = i + j;

            if (k >= 0 && k < signal_size) {
                filtered_signal[i] =
                    ComplexAdd(filtered_signal[i],
                               ComplexMul(signal[k], filter_kernel[minRadius - j]));
            }
        }
    }
}

int PadData(
    const Complex *signal, Complex **padded_signal, int signal_size,
    const Complex *filter_kernel, Complex **padded_filter_kernel,
    int filter_kernel_size) {

    int minRadius = filter_kernel_size / 2;
    int maxRadius = filter_kernel_size - minRadius;
    int new_size = signal_size + maxRadius;

    // Pad signal
    Complex *new_data =
        reinterpret_cast<Complex *>(malloc(sizeof(Complex) * new_size));
    memcpy(new_data + 0, signal, signal_size * sizeof(Complex));
    memset(new_data + signal_size, 0, (new_size - signal_size) * sizeof(Complex));
    *padded_signal = new_data;

    // Pad filter
    new_data = reinterpret_cast<Complex *>(malloc(sizeof(Complex) * new_size));
    memcpy(new_data + 0, filter_kernel + minRadius, maxRadius * sizeof(Complex));
    memset(new_data + maxRadius, 0,
           (new_size - filter_kernel_size) * sizeof(Complex));
    memcpy(new_data + new_size - minRadius, filter_kernel,
           minRadius * sizeof(Complex));
    *padded_filter_kernel = new_data;

    return new_size;
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
void runTest() {
    printf("[simpleCUFFT] is starting...\n");

    // Allocate host memory for the signal
    Complex *h_signal = reinterpret_cast<Complex *>(malloc(sizeof(Complex) * SIGNAL_SIZE));
    for (unsigned int i = 0; i < SIGNAL_SIZE; ++i){
        h_signal[i].x = rand() / static_cast<float>(RAND_MAX); h_signal[i].y = 0;
    }


    // Allocate host memory for the filter
    Complex *h_filter_kernel = reinterpret_cast<Complex *>(malloc(sizeof(Complex) * FILTER_KERNEL_SIZE));
    for (unsigned int i = 0; i < FILTER_KERNEL_SIZE; ++i){
        h_filter_kernel[i].x = rand() / static_cast<float>(RAND_MAX); h_filter_kernel[i].y = 0;
    }

    // Pad signal and filter kernel
    Complex *h_padded_signal;
    Complex *h_padded_filter_kernel;
    int new_size =
        PadData(h_signal, &h_padded_signal, SIGNAL_SIZE, h_filter_kernel,
                &h_padded_filter_kernel, FILTER_KERNEL_SIZE);
    int mem_size = sizeof(Complex) * new_size;

    // Allocate device memory for signal
    Complex *d_signal;
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_signal), mem_size));
    checkCudaErrors(
        cudaMemcpy(d_signal, h_padded_signal, mem_size, cudaMemcpyHostToDevice));

    // Allocate device memory for filter kernel
    Complex *d_filter_kernel;
    checkCudaErrors(
        cudaMalloc(reinterpret_cast<void **>(&d_filter_kernel), mem_size));
    checkCudaErrors(cudaMemcpy(d_filter_kernel, h_padded_filter_kernel, mem_size,
                               cudaMemcpyHostToDevice));

    // CUFFT plan simple API
    cufftHandle plan;
    checkCudaErrors(cufftPlan1d(&plan, new_size, CUFFT_C2C, 1));

    // Transform signal and kernel
    printf("Transforming signal cufftExecC2C\n");
    checkCudaErrors(cufftExecC2C(plan, reinterpret_cast<cufftComplex *>(d_signal),
                                 reinterpret_cast<cufftComplex *>(d_signal),
                                 CUFFT_FORWARD));

    // Multiply the coefficients together and normalize the result
    printf("Launching ComplexPointwiseMulAndScale<<< >>>\n");
    ComplexPointwiseMulAndScale<<<32, 256>>>(d_signal, d_filter_kernel, new_size,
                                             1.0f / new_size);

    // Transform signal back
    printf("Transforming signal back cufftExecC2C\n");
    checkCudaErrors(cufftExecC2C(plan, reinterpret_cast<cufftComplex *>(d_signal),
                                 reinterpret_cast<cufftComplex *>(d_signal),
                                 CUFFT_INVERSE));

    // Copy device memory to host
    Complex *h_convolved_signal = h_padded_signal;
    checkCudaErrors(cudaMemcpy(h_convolved_signal, d_signal, mem_size,
                               cudaMemcpyDeviceToHost));

    // Convolve on the host
    Complex *h_convolved_signal_ref =
        reinterpret_cast<Complex *>(malloc(sizeof(Complex) * SIGNAL_SIZE));
    Convolve(h_signal, SIGNAL_SIZE, h_filter_kernel, FILTER_KERNEL_SIZE,
             h_convolved_signal_ref);

    // Destroy CUFFT context
    checkCudaErrors(cufftDestroy(plan));

    free(h_signal);
    free(h_filter_kernel);
    free(h_padded_signal);
    free(h_padded_filter_kernel);
    free(h_convolved_signal_ref);
    checkCudaErrors(cudaFree(d_signal));
    checkCudaErrors(cudaFree(d_filter_kernel));

    OKGREEN("Done");

    // cufftExecR2C
    // Square and pivot
    // cufftExecC2R
}

int G1::GPU::g1_prepare_fftw_plan(cufftHandle *&plans_forward, cufftHandle *&plans_backward) {

    OKBLUE("Generating optimised forward and backward plans");
    plans_forward = new cufftHandle[G1::no_outputs]; plans_backward = new cufftHandle[G1::no_outputs];
    for (int i(0); i < G1::no_outputs; i++) {
        if (
            cufftPlan1d(&plans_forward[i], G1_DIGITISER_POINTS, CUFFT_R2C, 1) != CUFFT_SUCCESS)
            FAIL("Failed to create FFTW Forward Plan on GPU");
        if (
            cufftPlan1d(&plans_backward[i], G1_DIGITISER_POINTS, CUFFT_C2R, 1) != CUFFT_SUCCESS)
            FAIL("Failed to create FFTW Backward Plan on GPU");
    }
    return 0;
}

void G1::GPU::allocate_memory(short **chA_data, short **chB_data){
    int success = 0;
    success += cudaHostAlloc((void**)chA_data,
                             SP_POINTS * R_POINTS * sizeof(short),
                             cudaHostAllocDefault);
    success += cudaHostAlloc((void**)chB_data,
                                 SP_POINTS * R_POINTS * sizeof(short),
                                 cudaHostAllocDefault);
    if (success != 0) FAIL("Power Kernel: Failed to allocate locked input memory on CPU.");
}

void handle_error(cufftResult result, std::string error_message){
    if (result != CUFFT_SUCCESS) {
        std::stringstream ss;
        ss << error_message << ": Error code " << result << "\nCheck https://docs.nvidia.com/cuda/cufft/index.html#cufftresult";
        FAIL(ss.str());
    }
}
void handle_error(cudaError_t result, std::string error_message){
    std::cout << result << std::endl;

    if (result != 0) {
        std::stringstream ss;
        ss << error_message << ": Error code " << result << "\nCheck  https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g3f51e3575c2178246db0a94a430e0038";
        FAIL(ss.str());
    }
}

void G1::GPU::g1_kernel(
    short *chA_data, short *chB_data,
    double **data_out,
    cufftHandle *plans_forward, cufftHandle *plans_backward){

    // TODO: move sections below to separate method
    // Allocate memory for the GPU
    // Complex *intermediate_array = reinterpret_cast<Complex *>(malloc(sizeof(Complex) * G1_DIGITISER_POINTS));

    // Normalise input arrays
    // double mean_list[G1::no_outputs];
    // double variance_list[G1::no_outputs];
    // G1::CPU::preprocessor(chA_data, chB_data, G1_DIGITISER_POINTS, mean_list, variance_list, data_out);

    // Memory allocation
    // gpu_inout
    cufftReal *gpu_chA;
    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&gpu_chA), sizeof(cufftReal) * G1_DIGITISER_POINTS));
    // gpu_aux
    cufftComplex *aux_array;
    checkCudaErrors(
        cudaMalloc(
            reinterpret_cast<void **>(&aux_array), sizeof(cufftComplex) * (int(G1_DIGITISER_POINTS / 2) + 1)));
    // cpu_out
    // Complex *chA_outc = new Complex[G1_DIGITISER_POINTS];

    float *chA_out;
    checkCudaErrors(
        cudaHostAlloc(
            (void**)&chA_out,
            sizeof(float) * G1_DIGITISER_POINTS,
            cudaHostAllocDefault
            ));

    // Forward transform
    checkCudaErrors(cufftExecR2C(
                        plans_forward[0],
                        gpu_chA, aux_array));
    // Square
    fftw_square<<<G1_DIGITISER_POINTS / 1024 + 1,1024>>>(aux_array);
    // cudaMemcpy(aux_array, chA_out,
    //            sizeof(cufftReal) * G1_DIGITISER_POINTS,
    //            cudaMemcpyDeviceToHost);

    // Backward transform
    checkCudaErrors(cufftExecC2R(
                        plans_backward[0],
                        aux_array, gpu_chA));
// Copy back to CPU
    checkCudaErrors(
        cudaMemcpy(chA_out, gpu_chA,
                   sizeof(cufftReal) * G1_DIGITISER_POINTS,
                   cudaMemcpyDeviceToHost));

    for (int i(0); i < G1_DIGITISER_POINTS; i+=1)
        std::cout << chA_out[i] << std::endl;
}
