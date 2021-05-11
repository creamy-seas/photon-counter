/*
 * Used for computing magnitude squared:
 * A^2 + B^2 and individual quadratures
 * Inefficient, as it will always compute all values

 Data will be fed in as a block of R * (N * P=SP)
 SP: samples
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
    PROCESSING_ARRAY_TYPE chA_cumulative_array[R_POINTS],
    PROCESSING_ARRAY_TYPE chB_cumulative_array[R_POINTS],
    PROCESSING_ARRAY_TYPE sq_cumulative_array[R_POINTS]){
    /*
     * Reduce the array by summing up the total into the first cell.

     __ Logic ___
     a1 a2 a3 a4 ... b1 b2 b3 b4 ... c1 c2 c3 c4 ...

     will be mapped to a 2D array

     a1 a2 a3 -> main_axis (sp_coordinate)
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
            if (r_coordinate < idx) {
                chA_cumulative_array[r_coordinate] += chA_cumulative_array[r_coordinate + idx];
                chB_cumulative_array[r_coordinate] += chB_cumulative_array[r_coordinate + idx];
                sq_cumulative_array[r_coordinate] += sq_cumulative_array[r_coordinate + idx];
            }
            r_coordinate += blockDim.x;
        }
        __syncthreads();
        idx /= 2;
    }
}

// Background signal copied once to GPU
__constant__ short dev_chA_background[SP_POINTS];
__constant__ short dev_chB_background[SP_POINTS];

/*
  Call function to copy background for each channel a single time into
  * - dev_chA_background
  * - dev_chB_background
  */
void GPU::copy_background_arrays_to_gpu(short *chA_background, short *chB_background) {

    int success = 0;
    success += cudaMemcpyToSymbol(dev_chA_background, chA_background,
                                  SP_POINTS*sizeof(short));
    success += cudaMemcpyToSymbol(dev_chB_background, chB_background,
                                  SP_POINTS*sizeof(short));
    if (success != 0)
        FAIL("Failed to copy background data TO the GPU - check that the arrays have SP_POINTS in them!");
}



__global__ void power_kernel_v1_no_background_runner(
    short *chA_data, short *chB_data, double* chA_out, double* chB_out, double *sq_out){

    /*
      chA
      * 1  2   3   4    -> main axis (SP_POINTS=4)
      * 5  6   7   8
      * 9  10  11  12
      * |
      * repetition axis (R_POINTS=3)

      * chB
      * 0  1   0   1
      * 1  0   1   0
      * 2  2   2   2
      */
    __shared__ PROCESSING_ARRAY_TYPE chA_cumulative_array[R_POINTS];
    __shared__ PROCESSING_ARRAY_TYPE chB_cumulative_array[R_POINTS];
    __shared__ PROCESSING_ARRAY_TYPE sq_cumulative_array[R_POINTS];

    int sp_coordinate = blockIdx.x;
    int r_coordinate, coordinate;

    while (sp_coordinate < SP_POINTS) {
        r_coordinate = threadIdx.x;

        while (r_coordinate < R_POINTS) {
            coordinate = r_coordinate * SP_POINTS + sp_coordinate;

            chA_cumulative_array[r_coordinate] = chA_data[coordinate] * chA_data[coordinate];
            chB_cumulative_array[r_coordinate] = chB_data[coordinate] * chB_data[coordinate];
            sq_cumulative_array[r_coordinate] = chA_cumulative_array[r_coordinate] + chB_cumulative_array[r_coordinate];

            // Once thread has completed, shift the
            // row index by the number of allocated
            // threads and continue summation
            r_coordinate += blockDim.x;
        }

        // Ensure that all threads have completed execution
        __syncthreads();

        // Summation
        reduction_sum(chA_cumulative_array, chB_cumulative_array, sq_cumulative_array);
        chA_out[sp_coordinate] = (double)chA_cumulative_array[0] / R_POINTS;
        chB_out[sp_coordinate] = (double)chB_cumulative_array[0] / R_POINTS;
        sq_out[sp_coordinate] = (double)sq_cumulative_array[0] / R_POINTS;

        // Shift by number of allocated blocks along main-axis
        sp_coordinate += gridDim.x;
    }
}

// __global__ void power_kernel_v2_const_background_runner(
//     short *chA_data, short *chB_data, double *data_out, short chA_back, short chB_back){

//     __shared__ PROCESSING_ARRAY_TYPE cache_array[R_POINTS];

//     int sp_coordinate = blockIdx.x;
//     int r_coordinate, coordinate;
//     int _chA, _chB;

//     while (sp_coordinate < SP_POINTS) {
//         r_coordinate = threadIdx.x;

//         while (r_coordinate < R_POINTS) {
//             coordinate = r_coordinate * SP_POINTS + sp_coordinate;

//             _chA = chA_data[coordinate] - chA_back;
//             _chB = chB_data[coordinate] - chB_back;

//             cache_array[r_coordinate] = _chA * _chA + _chB * _chB;
//             // Once thread has completed, shift the
//             // row index by the number of allocated
//             // threads and continue summation
//             r_coordinate += blockDim.x;
//         }

//         // Ensure that all threads have completed execution
//         __syncthreads();

//         // Summation
//         reduction_sum(cache_array);
//         data_out[sp_coordinate] = (double)cache_array[0] / R_POINTS;

//         // Shift by number of allocated blocks along main-axis
//         sp_coordinate += gridDim.x;
//     }
// }

// /*
//   Background arrays are written into constant memory once and reused.
// */
// __global__ void power_kernel_v3_background_runner(
//     short *chA_data, short *chB_data, double *data_out){

//     __shared__ PROCESSING_ARRAY_TYPE cache_array[R_POINTS];

//     int sp_coordinate = blockIdx.x;
//     int r_coordinate, coordinate;
//     int _chA, _chB;

//     while (sp_coordinate < SP_POINTS) {
//         r_coordinate = threadIdx.x;

//         while (r_coordinate < R_POINTS) {
//             coordinate = r_coordinate * SP_POINTS + sp_coordinate;

//             _chA = chA_data[coordinate] - dev_chA_background[sp_coordinate];
//             _chB = chB_data[coordinate] - dev_chB_background[sp_coordinate];

//             cache_array[r_coordinate] = _chA * _chA + _chB * _chB;
//             // Once thread has completed, shift the
//             // row index by the number of allocated
//             // threads and continue summation
//             r_coordinate += blockDim.x;
//         }

//         // Ensure that all threads have completed execution
//         __syncthreads();

//         // Summation
//         reduction_sum(cache_array);
//         data_out[sp_coordinate] = (double)cache_array[0] / R_POINTS;

//         // Shift by number of allocated blocks along main-axis
//         sp_coordinate += gridDim.x;
//     }
// }

void GPU::power_kernel_v1_no_background(
    short *chA_data,
    short *chB_data,
    double **result_out,
    short **dev_chA_data,
    short **dev_chB_data,
    double **dev_chA_out,
    double **dev_chB_out,
    double **dev_sq_out){
    /*
     * chA and chB arrays:
     a1 a2 a3 a4 ... b1 b2 b3 b4 ... c1 c2 c3 c4 ...
    */

    // ==> Ensure that allocate_memory_on_gpu has been called

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
    if (success != 0) FAIL("Failed to copy data TO the GPU!");

    // Run kernel
    power_kernel_v1_no_background_runner<<<BLOCKS, THREADS_PER_BLOCK>>>(
        *dev_chA_data, *dev_chB_data, *dev_chA_out, *dev_chB_out, *dev_sq_out);

    // Copy from device
    success += cudaMemcpy(result_out[0], *dev_chA_out,
                          SP_POINTS * sizeof(double),
                          cudaMemcpyDeviceToHost);
    success += cudaMemcpy(result_out[1], *dev_chB_out,
                          SP_POINTS * sizeof(double),
                          cudaMemcpyDeviceToHost);
    success += cudaMemcpy(result_out[2], *dev_sq_out,
                          SP_POINTS * sizeof(double),
                          cudaMemcpyDeviceToHost);
    if (success != 0) FAIL("Failed to copy data FROM the GPU!");

    // Ensure that free_memory_on_gpu is called ==>
}

// void GPU::power_kernel_v2_const_background(
//     short *chA_data,
//     short *chB_data,
//     double *data_out,
//     short chA_back,
//     short chB_back,
//     short **dev_chA_data,
//     short **dev_chB_data,
//     double **dev_sq_out
//     ){
//     /*
//      * chA and chB arrays:
//      a1 a2 a3 a4 ... b1 b2 b3 b4 ... c1 c2 c3 c4 ...
//     */

//     // ==> Ensure that allocate_memory_on_gpu has been called

//     // Copy input data over to GPU.
//     // Dereference the dev_ch? (which is the address where the GPU memory location is kept)
//     // in order to get the actual memory location
//     int success = 0;
//     success += cudaMemcpy(*dev_chA_data, chA_data,
//                           TOTAL_POINTS*sizeof(short),
//                           cudaMemcpyHostToDevice);
//     success += cudaMemcpy(*dev_chB_data, chB_data,
//                           TOTAL_POINTS*sizeof(short),
//                           cudaMemcpyHostToDevice);
//     if (success != 0) FAIL("Failed to copy data TO the GPU!");

//     // Run kernel
//     power_kernel_v2_const_background_runner<<<BLOCKS, THREADS_PER_BLOCK>>>(
//             *dev_chA_data, *dev_chB_data, *dev_sq_out, chA_back, chB_back);

//     // Copy from device
//     success += cudaMemcpy(
//         data_out,
//         *dev_sq_out,
//         SP_POINTS * sizeof(double),
//         cudaMemcpyDeviceToHost);
//     if (success != 0) FAIL("Failed to copy data FROM the GPU!");

//     // Ensure that free_memory_on_gpu is called ==>
// }

// void GPU::power_kernel_v3_background(
//     short *chA_data,
//     short *chB_data,
//     double *data_out,
//     short **dev_chA_data,
//     short **dev_chB_data,
//     double **dev_sq_out
//     ){
//     /*
//      * chA and chB arrays:
//      a1 a2 a3 a4 ... b1 b2 b3 b4 ... c1 c2 c3 c4 ...
//     */

//     // ==> Ensure that allocate_memory_on_gpu has been called

//     // Copy input data over to GPU.
//     // Dereference the dev_ch? (which is the address where the GPU memory location is kept)
//     // in order to get the actual memory location
//     int success = 0;
//     success += cudaMemcpy(*dev_chA_data, chA_data,
//                           TOTAL_POINTS*sizeof(short),
//                           cudaMemcpyHostToDevice);
//     success += cudaMemcpy(*dev_chB_data, chB_data,
//                           TOTAL_POINTS*sizeof(short),
//                           cudaMemcpyHostToDevice);
//     if (success != 0) FAIL("Failed to copy data TO the GPU!");

//     // Run kernel
//     power_kernel_v3_background_runner<<<BLOCKS, THREADS_PER_BLOCK>>>(
//         *dev_chA_data, *dev_chB_data, *dev_sq_out);

//     // Copy from device
//     success += cudaMemcpy(
//         data_out,
//         *dev_sq_out,
//         SP_POINTS * sizeof(double),
//         cudaMemcpyDeviceToHost);
//     if (success != 0) FAIL("Failed to copy data FROM the GPU!");

//     // Ensure that free_memory_on_gpu is called ==>
// }
