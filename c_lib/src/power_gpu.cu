#include <stdio.h>
#include <string>

#include "colours.hpp"
#include "utils_gpu.hpp"
#include "power.hpp"


#ifndef PROCESSING_ARRAY_TYPE_POWER_KERNEL
#define PROCESSING_ARRAY_TYPE_POWER_KERNEL int
#endif /* PROCESSING_ARRAY_TYPE_POWER_KERNEL */

#ifndef R_POINTS
#define R_POINTS 1000
#endif /* R_POINTS */

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

__device__ void reduction_sum(PROCESSING_ARRAY_TYPE* cached_array[R_POINTS]){
        /*
         * Reduce the array by summing up the total into the first cell
         */

        int idx = R_POINTS / 2;
        int r_coodinate;

        while (idx != 0) {
                r_coordinate = threadIdx.x;
                while (r_coodinate < R_POINTS){
                        if (r_coordinate < idx)
                                cached_array[r_coordinate] += cached_array[r_coordinate + i];
                        r_coordinate += blockDim.x;
                }
                __syncthreads();
                idx /= 2;
        }
}



int verify_gpu_allocation(
        cudaDeviceProp gpu_properties,
        cudaDeviceProp desired_properties){
        // Check that processing array type can hold all parameters
        // Check that
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

/* void GPU::power_kernel( */
/*         short *chA_data_in, */
/*         short *chB_data_in, */
/*         short *chA_data_out, */
/*         short *chB_data_out, */
/*         unsigned int * sq_data_out, */
/*         int samples_per_record, */
/*         int number_of_records */
/*         ){ */


/*         // Allocate space on GPU */
/* } */
