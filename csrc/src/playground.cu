#include <iostream>
#include "playground.hpp"
#include "colours.hpp"

__global__ void example_gpu_func_kernel(int a, int b, float *c){
        *c = (float)(a * a + b * b);
}

float GPU::example_gpu_func(short a, short b) {

        // cudaDeviceProp prop = fetch_gpu_parameters();

        /* cudaDeviceProp program_prop; */
        /* memset(&program_prop, 0, sizeof(cudaDeviceProp)); */
        /* program_prop.maxGridSize[0] = 100; */

        // Allocate device and host variables
        float c;
        float *dev_c;

        // Memory allocation on device
        cudaMalloc((void**) &dev_c, sizeof(float));

        // Kernel invoction
        example_gpu_func_kernel<<<1,1>>>(a, b, dev_c);

        // Copy back to device
        cudaMemcpy(
                &c,
                dev_c,
                sizeof(float),
                cudaMemcpyDeviceToHost
                );


        cudaFree(dev_c);
        OKGREEN("GPU KERNEL Complete!");

        std::cout << c << std::endl;

        return c;
}
