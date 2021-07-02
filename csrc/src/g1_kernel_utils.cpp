#include <cuda_runtime_api.h> //for cudaDeviceProp

#include "logging.hpp"
#include "g1_kernel.hpp"
#include "utils_gpu.hpp"

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

template <typename T> void G1::CPU::preprocessor(short *chA_data, short *chB_data,
                                                 int N,
                                                 T *mean_list,
                                                 T *variance_list,
                                                 T **normalised_data) {
    T chA_mean(0), chB_mean(0), sq_mean(0);
    for (int i = 0; i < N; i++) {
        chA_mean += chA_data[i];
        chB_mean += chB_data[i];
        normalised_data[SQG1][i] = (T)chA_data[i] * (T)chA_data[i] + (T)chB_data[i] * (T)chB_data[i];
        sq_mean += normalised_data[SQG1][i];
    }
    chA_mean /= N;
    chB_mean /= N;
    sq_mean /= N;
    mean_list[CHAG1] = chA_mean;
    mean_list[CHBG1] = chB_mean;
    mean_list[SQG1] = sq_mean;

    // Evaluation of variance and normalisation
    T chA_sqDiff(0), chB_sqDiff(0), sq_sqDiff(0);
    for (int i = 0; i < N; i++) {
        normalised_data[CHAG1][i] = chA_data[i] - chA_mean;
        normalised_data[CHBG1][i] = chB_data[i] - chB_mean;
        normalised_data[SQG1][i] = normalised_data[SQG1][i] - sq_mean;

        chA_sqDiff += normalised_data[CHAG1][i] * normalised_data[CHAG1][i];
        chB_sqDiff += normalised_data[CHBG1][i] * normalised_data[CHBG1][i];
        sq_sqDiff += normalised_data[SQG1][i] * normalised_data[SQG1][i];
    }

    variance_list[CHAG1] = chA_sqDiff / N;
    variance_list[CHBG1] = chB_sqDiff / N;
    variance_list[SQG1] = sq_sqDiff / N;
}

template void G1::CPU::preprocessor<float>(short *chA_data, short *chB_data, int N, float *mean_list, float *variance_list, float **normalised_data);
template void G1::CPU::preprocessor<double>(short *chA_data, short *chB_data, int N, double *mean_list, double *variance_list, double **normalised_data);
