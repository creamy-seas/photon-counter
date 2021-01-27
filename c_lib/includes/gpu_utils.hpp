#include <cuda_runtime.h> //for cudaDeviceProp

#define CUDA_CHECK(i, message)                  \
        if (i != 0) {                           \
                FAIL(message);}

#ifndef UTILS_GPU_HPP
#define UTILS_GPU_HPP

cudaDeviceProp fetch_gpu_parameters();

#endif
