#include <cuda_runtime.h> //for cudaDeviceProp

#ifndef UTILS_GPU_HPP
#define UTILS_GPU_HPP

#define WARP_SIZE 32 ///< Warp size for Nvidia Quadro P6000 GPU. Unlikely to change.

/*
 * Fetch and display parameters of the GPU (already done in python, this is just proof of concept)
 */
cudaDeviceProp fetch_gpu_parameters(bool display=false);

#endif
