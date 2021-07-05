#include <cuda_runtime.h> // cudaMalloc cudaFree

#include "logging.hpp"
#include "g1_kernel.hpp"

int G1::check_g1_kernel_parameters(bool display){
    PYTHON_START;

    cudaDeviceProp prop = fetch_gpu_parameters(display);

    if (G1::GPU::pp_threads > prop.maxThreadsPerBlock)
        FAIL("pp_threads ("
             + std::to_string(G1::GPU::pp_threads)
             + ") declared in g1_kernel.hpp > maxThreadsPerBlock ("
             + std::to_string(prop.maxThreadsPerBlock)
             + ") of "
             + std::string(prop.name)
            );

    if (G1::GPU::pp_shared_memory * 3 > prop.sharedMemPerBlock)
        FAIL("pp_shared_memory x 3 for CHAG1, CHBG1, SQG1 preprocessing ("
             + std::to_string(3 * G1::GPU::pp_shared_memory)
             + " bytes) > sharedMemPerBlock ("
             + std::to_string(prop.sharedMemPerBlock)
             + " bytes) of "
             + std::string(prop.name)
            );

    if (
        (G1_DIGITISER_POINTS + G1::GPU::pp_threads - 1) / G1::GPU::pp_threads
        >
        G1::GPU::pp_threads)
        FAIL(
            "G1 Kernel will not be able to preprocess the G1_DIGITISER_POINTS ("
            + std::to_string(G1_DIGITISER_POINTS)
            + ") using a reduction summation."
            );

    PYTHON_END;
    return 0;
}

unsigned long G1::GPU::get_number_of_blocks(int N) {
    return (unsigned long)(N + G1::GPU::pp_threads - 1) / G1::GPU::pp_threads;
}

void G1::GPU::allocate_memory(
    short **&gpu_raw_data, cufftReal **&gpu_inout, float **&cpu_inout,
    float **&gpu_pp_aux, cufftComplex **&gpu_fftw_aux, float *&gpu_mean, float *&gpu_variance,
    int N) {
    int success = 0;
    unsigned long blocks = G1::GPU::get_number_of_blocks(N);

    gpu_raw_data = new short*[2];
    success += cudaMalloc(reinterpret_cast<void**>(&gpu_raw_data[CHAG1]), G1_DIGITISER_POINTS * sizeof(short));
    success += cudaMalloc(reinterpret_cast<void**>(&gpu_raw_data[CHBG1]), G1_DIGITISER_POINTS * sizeof(short));

    gpu_pp_aux = new float*[G1::no_outputs];
    gpu_inout = new cufftReal*[G1::no_outputs];
    gpu_fftw_aux = new cufftComplex*[G1::no_outputs];
    for (int i(0); i < G1::no_outputs; i++) {
        success += cudaMalloc(reinterpret_cast<void**>(&gpu_pp_aux[i]), blocks * sizeof(float));
        success += cudaMalloc(reinterpret_cast<void**>(&gpu_inout[i]), G1_DIGITISER_POINTS * sizeof(cufftReal));
        success += cudaMalloc(reinterpret_cast<void**>(&gpu_fftw_aux[i]), (int(G1_DIGITISER_POINTS / 2) + 1) * sizeof(cufftComplex));
    }
    success += cudaMalloc(reinterpret_cast<void**>(&gpu_mean), G1::no_outputs * sizeof(float));
    success += cudaMalloc(reinterpret_cast<void**>(&gpu_variance), G1::no_outputs * sizeof(float));
    if (success != 0) FAIL("G1 Kernel: Failed to allocate memory on GPU.");

    cpu_inout = new float*[G1::no_outputs];
    for (int i(0); i < G1::no_outputs; i++)
        success += cudaHostAlloc(reinterpret_cast<void**>(&cpu_inout[i]), G1_DIGITISER_POINTS * sizeof(float), cudaHostAllocDefault);
    if (success != 0) FAIL("G1 Kernel: Failed to allocate locked memory on CPU.");
}

void G1::GPU::free_memory(
    short **gpu_raw_data, cufftReal **gpu_inout, float **cpu_inout,
    float **gpu_pp_aux, cufftComplex **gpu_fftw_aux, float *gpu_mean, float *gpu_variance) {
    OKBLUE("G1 Kernel: Deallocating memory on GPU and CPU.");
    int success = 0;

    cudaFree(gpu_raw_data[CHAG1]); cudaFree(gpu_raw_data[CHBG1]);
    delete[] gpu_raw_data;

    for (int i(0); i < G1::no_outputs; i++) {
        success += cudaFree(gpu_pp_aux[i]);
        success += cudaFree(gpu_inout[i]);
        success += cudaFree(gpu_fftw_aux[i]);
    }
    cudaFree(gpu_mean);
    cudaFree(gpu_variance);
    delete[] gpu_pp_aux;
    delete[] gpu_inout;
    delete[] gpu_fftw_aux;
    if (success != 0) FAIL("Power Kernel: Failed to free memory on GPU.");

    for (int i(0); i < G1::no_outputs; i++) {
        success += cudaFreeHost(cpu_inout[i]);
    }
    delete[] cpu_inout;
    if (success != 0) FAIL("Power Kernel: Failed to free memory on CPU.");
}

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
