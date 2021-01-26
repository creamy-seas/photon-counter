/*
 * Used for computing magnitude squared:
 * A^2 + B^2

 Data will be fed in as a block of R * (N * P)
 R: Number of repititions
 N: Number of pulses
 P: Period of a pulse
*/

#include <stdio.h>
#include <string>
#include "colours.hpp"
#include "gpu_utils.hpp"
#include "power_kernel.hpp"

#define xstr(s) str(s)
#define str(s) #s

#ifndef PROCESSING_ARRAY_TYPE
#define PROCESSING_ARRAY_TYPE int
#endif

#ifndef R_POINTS
#define R_POINTS 1000
#endif

#ifndef NP_POINTS
#define NP_POINTS 1000
#endif

#ifndef THREADS_PER_BLOCK
#define THREADS_PER_BLOCK 1024
#endif

#define BLOCKS NP_POINTS

#define TOTAL_POINTS NP_POINTS*R_POINTS

__global__ void magnitude_squared(int a, int b, float *c){
        *c = (float)(a * a + b * b);
}

/*
  Suite for referencing the parametesr that the GPU kernel was compiled with
*/
GPU::PowerKernelParameters::PowerKernelParameters(
        int r_points,
        int np_points,
        std::string processing_array_type,
        int blocks,
        int threads_per_block
        ){
        this->r_points = r_points;
        this->np_points = np_points;
        this->processing_array_type = processing_array_type;
        this->blocks = blocks;
        this->threads_per_block = threads_per_block;

        this->print();
}

void GPU::PowerKernelParameters::print(){
        OKBLUE("===========================================");
        RED("          **POWER KERNEL**");

        OKBLUE("Data Parameters");
        printf("R_POINTS: %i\n", this->r_points );
        printf("NP_POINTS: %i\n", this->np_points );

        OKBLUE("Processing Parameters");
        printf("CACHED ARRAY TYPE: %s\n", this->processing_array_type.c_str() );
        printf("BLOCKS: %i\n", this->blocks );
        printf("THREADS_PER_BLOCK: %i\n", this->threads_per_block );

        OKBLUE("===========================================");
}

GPU::PowerKernelParameters GPU::fetch_kernel_parameters(){
        // GPU::PowerKernelParameters kp =
        return GPU::PowerKernelParameters(
                R_POINTS,
                NP_POINTS,
                xstr(PROCESSING_ARRAY_TYPE),
                BLOCKS,
                THREADS_PER_BLOCK
                );
}

// Pass in empty pointers -> this will allocate memory on GPU
void GPU::allocate_memory_on_gpu(
        short *dev_chA_data, short *dev_chB_data, float *dev_sq_data
        ){

        OKBLUE("Allocating memory for power kernel on GPU");

        cudaMalloc((void**)&dev_chA_data,
                   TOTAL_POINTS * sizeof(short));

        cudaMalloc((void**)&dev_chB_data,
                   TOTAL_POINTS * sizeof(short));

        cudaMalloc((void**)&dev_sq_data,
                   NP_POINTS * sizeof(float));
        // cudaFree(dev_sq_data);

        OKGREEN("Allocation done!");
}

// Call to deallocated memory on GPU after run is complete
void GPU::free_memory_on_gpu(
        short *dev_chA_data, short *dev_chB_data, float *dev_sq_data
        ){

        OKBLUE("Deallocating memory from GPU");

        // cudaFree(dev_chA_data);
        // cudaFree(dev_chB_data);
        // cudaFree((void*) dev_sq_data);

        OKGREEN("Memory freed!");
}

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

        // __shared__ PROCESSING_ARRAY_TYPE cache_array[R_POINTS];

        // int np_coordinate = blockIdx.x;
        // int r_coordinate, coordinate;

        sq_data[0] = 67;//(float)cache_array[0] / R_POINTS;

        // while (np_coordinate < NP_POINTS) {
        //         r_coordinate = threadIdx.x;

        //         while (r_coordinate < R_POINTS) {
        //                 coordinate = r_coordinate * NP_POINTS + np_coordinate;

        //                 cache_array[r_coordinate] = (
        //                         chA_data[coordinate] * chA_data[coordinate]
        //                         + chB_data[coordinate] * chB_data[coordinate]
        //                         );
        //                 // Once thread has completed, shift the
        //                 // row index by the number of allocated
        //                 // threads and continue summation
        //                 r_coordinate += blockDim.x;
        //         }

        //         // Ensure that all threads have completed execution
        //         __syncthreads();

        //         // Summation
        //         reduction_sum(cache_array);
        //         sq_data[np_coordinate] = 67;//(float)cache_array[0] / R_POINTS;

        //         // Shift by number of allocated blocks along main-axis
        //         np_coordinate += gridDim.x;
        // }
}

void GPU::power_kernel(
        short *chA_data,
        short *chB_data,
        float *sq_data,
        short *dev_chA_data,
        short *dev_chB_data,
        float *dev_sq_data
        ){
        /*
         * chA and chB arrays:
         a1 a2 a3 a4 ... b1 b2 b3 b4 ... c1 c2 c3 c4 ...
        */

        // Ensure that allocate_memory_on_gpu has been called

        // Copy input data over to GPU
        // cudaMemcpy(dev_chA_data, chA_data, TOTAL_POINTS*sizeof(short),
        // cudaMemcpyHostToDevice);
        // cudaMemcpy(dev_chB_data, chB_data, TOTAL_POINTS*sizeof(short),
        // cudaMemcpyHostToDevice);

        // Run kernel
        power_kernel_v1_no_background_runner<<<1, 1>>>(
                dev_chA_data, dev_chB_data, dev_sq_data);

        // Copy from device
        cudaMemcpy(
                sq_data,
                dev_sq_data,
                NP_POINTS * sizeof(float),
                cudaMemcpyDeviceToHost);
        // sq_data[0]=6;

        // Ensure that free_memory_on_gpu is called
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
        // OKGREEN("GPU KERNEL Complete!");

        return c;
}
