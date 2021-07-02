#include <cuda_runtime.h> // cudaMalloc cudaFree

#include "logging.hpp"
#include "g1_kernel.hpp"

int G1::GPU::g1_prepare_fftw_plan(cufftHandle *&plans_forward, cufftHandle *&plans_backward) {

    OKBLUE("Generating optimised forward and backward plans");
    plans_forward = new cufftHandle[G1::no_outputs];
    plans_backward = new cufftHandle[G1::no_outputs];

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

void G1::GPU::allocate_memory(cufftReal **&gpu_inout, cufftComplex **&gpu_aux, float **&cpu_inout) {
    int success = 0;
    gpu_inout = new cufftReal*[G1::no_outputs];
    gpu_aux = new cufftComplex*[G1::no_outputs];
    for (int i(0); i < G1::no_outputs; i++) {
        success += cudaMalloc(reinterpret_cast<void**>(&gpu_inout[i]), G1_DIGITISER_POINTS * sizeof(cufftReal));
        success += cudaMalloc(reinterpret_cast<void**>(&gpu_aux[i]), (int(G1_DIGITISER_POINTS / 2) + 1) * sizeof(cufftComplex));
    }
    if (success != 0) FAIL("G1 Kernel: Failed to allocate memory on GPU.");

    cpu_inout = new float*[G1::no_outputs];
    for (int i(0); i < G1::no_outputs; i++)
        success += cudaHostAlloc(reinterpret_cast<void**>(&cpu_inout[i]), G1_DIGITISER_POINTS * sizeof(float), cudaHostAllocDefault);
    if (success != 0) FAIL("G1 Kernel: Failed to allocate locked memory on CPU.");
}

void G1::GPU::free_memory(cufftReal **gpu_inout, cufftComplex **gpu_aux, float **cpu_inout) {
    OKBLUE("G1 Kernel: Deallocating memory on GPU and CPU.");
    int success = 0;

    for (int i(0); i < G1::no_outputs; i++) {
        success += cudaFree(gpu_inout[i]);
        success += cudaFree(gpu_aux[i]);
    }
    delete[] gpu_inout;
    delete[] gpu_aux;
    if (success != 0) FAIL("Power Kernel: Failed to free memory on GPU.");

    for (int i(0); i < G1::no_outputs; i++) {
        success += cudaFreeHost(cpu_inout[i]);
    }
    delete[] cpu_inout;
    if (success != 0) FAIL("Power Kernel: Failed to free memory on CPU.");
}
