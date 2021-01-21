#include <stdio.h>
#include <string>

#include "colours.hpp"
#include "utils_gpu.hpp"
#include "power.hpp"

__global__ void magnitude_squared(int a, int b, float *c){
        /*
         * Used for computing magnitude squared:
         * A^2 + B^2

         Since data will be fed in as a block of R * (N * P)
         R: Number of repititions
         N: Number of pulses
         P: Period of a pulse

         We will transform it
        */

        *c = (float)(a * a + b * b);
}



int verify_gpu_allocation(
        cudaDeviceProp gpu_properties,
        cudaDeviceProp desired_properties){
        return 1;
}


float GPU::power_kernel(short a, short b) {

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
        magnitude_squared<<<1,1>>>(1,2, dev_c);

        // Copy back to device
        cudaMemcpy(
                &c,
                dev_c,
                sizeof(float),
                cudaMemcpyDeviceToHost
                );


        cudaFree(dev_c);
        OKGREEN("GPU KERNEL Complete!");

        return c;
}
