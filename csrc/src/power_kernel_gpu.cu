/*
 * Used for computing magnitude squared:
 * A^2 + B^2

 Data will be fed in as a block of R * (N * P)
 R: Number of repititions
 N: Number of pulses
 P: Period of a pulse
*/

#include <stdexcept>
#include <stdio.h>
#include <string>
#include "colours.hpp"
#include "gpu_utils.hpp"
#include "power_kernel.hpp"

__device__ void reduction_sum(
        PROCESSING_ARRAY_TYPE cache_array[R_POINTS]){
        /*
         * Reduce the array by summing up the total into the first cell.

            __ Logic ___
            a1 a2 a3 a4 ... b1 b2 b3 b4 ... c1 c2 c3 c4 ...

            will be mapped to a 2D array

            a1 a2 a3 -> main_axis (np_coordinate)
            b1 b2 b3 ...
            c1 c2 c3 ...
            d1 d2 d3 ...
            e1 e2 e3 ...
            f1 f2 f3 ...
            g1 g2 g3 ...

            |
            repetition-axis (r_coordinate)

            And reduced to the following by summing up over the repetition axis
            <1> <2> <3> ...
         */

        int idx = R_POINTS / 2;
        int r_coordinate;

        while (idx != 0) {
                r_coordinate = threadIdx.x;
                while (r_coordinate < R_POINTS){
                        if (r_coordinate < idx)
                                cache_array[r_coordinate] += cache_array[r_coordinate + idx];
                        r_coordinate += blockDim.x;
                }
                __syncthreads();
                idx /= 2;
        }
}

__global__ void power_kernel_v1_no_background_runner(
        short *chA_data, short *chB_data, float *sq_data){

        /*
          chA
          * 1  2   3   4    -> main axis (NP_POINTS=4)
          * 5  6   7   8
          * 9  10  11  12
          * |
          * repetition axis (R_POINTS=3)

          * chB
          * 0  1   0   1
          * 1  0   1   0
          * 2  2   2   2
         */
        __shared__ PROCESSING_ARRAY_TYPE cache_array[R_POINTS];

        int np_coordinate = blockIdx.x;
        int r_coordinate, coordinate;

        while (np_coordinate < NP_POINTS) {
                r_coordinate = threadIdx.x;

                while (r_coordinate < R_POINTS) {
                        coordinate = r_coordinate * NP_POINTS + np_coordinate;

                        cache_array[r_coordinate] = (
                                chA_data[coordinate] * chA_data[coordinate]
                                + chB_data[coordinate] * chB_data[coordinate]
                                );
                        // Once thread has completed, shift the
                        // row index by the number of allocated
                        // threads and continue summation
                        r_coordinate += blockDim.x;
                }

                // Ensure that all threads have completed execution
                __syncthreads();

                // Summation
                reduction_sum(cache_array);
                sq_data[np_coordinate] = (float)cache_array[0] / R_POINTS;

                // Shift by number of allocated blocks along main-axis
                np_coordinate += gridDim.x;
        }
}

void GPU::power_kernel(
        short *chA_data,
        short *chB_data,
        float *sq_data,
        short **dev_chA_data,
        short **dev_chB_data,
        float **dev_sq_data
        ){
        /*
         * chA and chB arrays:
         a1 a2 a3 a4 ... b1 b2 b3 b4 ... c1 c2 c3 c4 ...
        */

        // ==> Ensure that allocate_memory_on_gpu has been called

#if R_POINTS %2 != 0
        throw std::runtime_error("R_POINTS needs to be a even number");
#endif

        // Copy input data over to GPU.
        // Dereference the dev_ch? (which is the address where the GPU memory location is kept)
        // in order to get the actual memory location
        int success = 0;
        success += cudaMemcpy(*dev_chA_data, chA_data,
                              TOTAL_POINTS*sizeof(short),
                              cudaMemcpyHostToDevice);
        success += cudaMemcpy(*dev_chB_data, chB_data,
                              TOTAL_POINTS*sizeof(short),
                              cudaMemcpyHostToDevice);
        if (success != 0)
                FAIL("Failed to copy data TO the GPU!");

        // Run kernel
        power_kernel_v1_no_background_runner<<<BLOCKS, THREADS_PER_BLOCK>>>(
                *dev_chA_data, *dev_chB_data, *dev_sq_data);

        // Copy from device
        success += cudaMemcpy(
                sq_data,
                *dev_sq_data,
                NP_POINTS * sizeof(float),
                cudaMemcpyDeviceToHost);
        if (success != 0)
                FAIL("Failed to copy data FROM the GPU!");

        // Ensure that free_memory_on_gpu is called ==>
}
