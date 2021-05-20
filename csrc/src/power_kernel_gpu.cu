/*
 * Data from the digitiser will be fed in as a block of R * (N * P=SP)
 *  - SP: samples
 *  - R: Number of repititions
 *  - N: Number of pulses
 *  - P: Period of a pulse
 *
 *  e.g.
 *  a1 a2 a3 a4 ... b1 b2 b3 b4 ... c1 c2 c3 c4 ...
 *
 *  and after evaluating at each point, will be mapped to a 2D array
 *
 *  a1 a2 a3 -> main_axis (sp_coordinate)
 *  b1 b2 b3 ...
 *  c1 c2 c3 ...
 *  d1 d2 d3 ...
 *  e1 e2 e3 ...
 *  f1 f2 f3 ...
 *  g1 g2 g3 ...
 *
 *  |
 *  repetition-axis (r_coordinate)
 *
 *  And reduced to the following by summing up over the repetition axis
 *  <1> <2> <3> ...
 *
 * __ Kernel __
 * This kernel uses shared memory, so only a single copy of data to the GPU is required
 */

#include <stdexcept>
#include <stdio.h>
#include <string>

#include "colours.hpp"
#include "power_kernel.hpp"

// Background signal copied once to GPU
__constant__ short gpu_chA_background[SP_POINTS];
__constant__ short gpu_chB_background[SP_POINTS];

/*
  Call function to copy background for each channel a single time into
  * - gpu_chA_background
  * - gpu_chB_background
  */
void GPU::copy_background_arrays_to_gpu(short *chA_background, short *chB_background) {

    int success = 0;
    success += cudaMemcpyToSymbol(gpu_chA_background, chA_background,
                                  SP_POINTS*sizeof(short));
    success += cudaMemcpyToSymbol(gpu_chB_background, chB_background,
                                  SP_POINTS*sizeof(short));
    if (success != 0)
        FAIL("Failed to copy background data TO the GPU - check that the arrays have SP_POINTS in them!");
}

__device__ void reduction_sum(
    unsigned long int* chA_cumulative_array,
    unsigned long int* chB_cumulative_array,
    unsigned long int* chAsq_cumulative_array,
    unsigned long int* chBsq_cumulative_array,
    int length
    ){
    /*
     * Reduce the array by summing up the total of each array into the first cell.
     */

    int idx = length / 2;
    int r_coordinate;

    while (idx != 0) {
        r_coordinate = threadIdx.x;
        while (r_coordinate < length){
            if (r_coordinate < idx) {
                chA_cumulative_array[r_coordinate] += chA_cumulative_array[r_coordinate + idx];
                chB_cumulative_array[r_coordinate] += chB_cumulative_array[r_coordinate + idx];
                chAsq_cumulative_array[r_coordinate] += chAsq_cumulative_array[r_coordinate + idx];
                chBsq_cumulative_array[r_coordinate] += chBsq_cumulative_array[r_coordinate + idx];
            }
            r_coordinate += blockDim.x;
        }
        __syncthreads();
        idx /= 2;
    }
}

///////////////////////////////////////////////////////////////////////////////
//          Power Kernel: Single copy of data to GPU + shared memory         //
///////////////////////////////////////////////////////////////////////////////
__global__ void power_kernel_runner(short **gpu_in, double **gpu_out){

    // An unsigned long int will be able to hold 2^32/(2^14) = 2^18 = 262144 data points from the 14bit digitiser
    // Since the SP digitizer will have a maximum of 254200 record (using the GetMaxNofRecordsFromNofSamples(adq_cu_ptr, 1))
    // This will be able to contain everything
    __shared__ unsigned long int chA_cumulative_array[R_POINTS];
    __shared__ unsigned long int chB_cumulative_array[R_POINTS];
    __shared__ unsigned long int chAsq_cumulative_array[R_POINTS];
    __shared__ unsigned long int chBsq_cumulative_array[R_POINTS];

    int sp_coordinate = blockIdx.x;
    int r_coordinate, coordinate;

    // Each block deals with a specific SP_POINT
    // Each thread iterates over R_POINTS for each SP_POINT
    while (sp_coordinate < SP_POINTS) {
        r_coordinate = threadIdx.x;

        while (r_coordinate < R_POINTS) {
            coordinate = r_coordinate * SP_POINTS + sp_coordinate;

            chA_cumulative_array[r_coordinate] = gpu_in[CHA][coordinate] - gpu_chA_background[sp_coordinate];
            chB_cumulative_array[r_coordinate] = gpu_in[CHB][coordinate] - gpu_chB_background[sp_coordinate];

            chAsq_cumulative_array[r_coordinate] = chA_cumulative_array[r_coordinate] * chA_cumulative_array[r_coordinate];
            chBsq_cumulative_array[r_coordinate] = chB_cumulative_array[r_coordinate] * chB_cumulative_array[r_coordinate];

            // Once thread has completed, shift the
            // row index by the number of allocated
            // threads and continue summation
            r_coordinate += blockDim.x;
        }

        // Ensure that all threads have completed execution
        __syncthreads();

        // Summation
        reduction_sum(chA_cumulative_array,
                      chB_cumulative_array,
                      chAsq_cumulative_array,
                      chBsq_cumulative_array,
                      R_POINTS);
        gpu_out[CHA][sp_coordinate] = (double)chA_cumulative_array[0] / R_POINTS;
        gpu_out[CHB][sp_coordinate] = (double)chB_cumulative_array[0] / R_POINTS;
        gpu_out[CHASQ][sp_coordinate] = (double)chAsq_cumulative_array[0] / R_POINTS;
        gpu_out[CHBSQ][sp_coordinate] = (double)chBsq_cumulative_array[0] / R_POINTS;

        // Shift by number of allocated blocks along main-axis
        sp_coordinate += gridDim.x;
    }
}

void GPU::V1::power_kernel(
    short *chA_data,
    short *chB_data,
    double **data_out,
    short ***gpu_in, double ***gpu_out){
    // ==> Ensure that allocate_memory has been called
    // ==> Ensure that background arrays (set to 0 for no correction) have been copied over

    // Copy input data over to GPU.
    // Dereference the gpu_ch? (which is the address where the GPU memory location is kept)
    // in order to get the actual memory location
    int success = 0;
    success += cudaMemcpy(*gpu_in[CHA], chA_data,
                          R_POINTS * SP_POINTS * sizeof(short),
                          cudaMemcpyHostToDevice);
    success += cudaMemcpy(*gpu_in[CHB], chB_data,
                          R_POINTS * SP_POINTS * sizeof(short),
                          cudaMemcpyHostToDevice);
    if (success != 0) FAIL("Failed to copy data TO the GPU!");

    // Run kernel
    power_kernel_runner<<<BLOCKS, THREADS_PER_BLOCK>>>(*gpu_in, *gpu_out);

    // Copy from device
    // success += cudaMemcpy(data_out[CHA], *gpu_out[CHA],
    //                       SP_POINTS * sizeof(double),
    //                       cudaMemcpyDeviceToHost);
    // success += cudaMemcpy(data_out[CHB], *gpu_out[CHB],
    //                       SP_POINTS * sizeof(double),
    //                       cudaMemcpyDeviceToHost);
    // success += cudaMemcpy(data_out[CHASQ],*gpu_out[CHASQ],
    //                       SP_POINTS * sizeof(double),
    //                       cudaMemcpyDeviceToHost);
    // success += cudaMemcpy(data_out[CHBSQ], *gpu_out[CHBSQ],
    //                       SP_POINTS * sizeof(double),
    //                       cudaMemcpyDeviceToHost);
    if (success != 0) FAIL("Failed to copy data FROM the GPU!");

    // Manually evaluate sq = chAsq + chBsq
    for (int i(0); i < SP_POINTS; i++)
        data_out[SQ][i] = data_out[CHASQ][i] + data_out[CHBSQ][i];

    // Ensure that free_memory is called ==>
}


// void GPU::V2::power_kernel(
//     short *chA_data,
//     short *chB_data,
//     double **data_out,
//     short **gpu_chA_data0,
//     short **gpu_chB_data0,
//     double **gpu_chA_out0,
//     double **gpu_chB_out0,
//     double **gpu_chAsq_out0,
//     double **gpu_chBsq_out0,
//     short **gpu_chA_data1,
//     short **gpu_chB_data1,
//     double **gpu_chA_out1,
//     double **gpu_chB_out1,
//     double **gpu_chAsq_out1,
//     double **gpu_chBsq_out1
//     ){
//     // ==> Ensure that allocate_memory has been called
//     // ==> Ensure that background arrays (set to 0 for no correction) have been copied over

//     // Allocate auxillary arrays into which data will be copied in between the stream runs
//     double *host_chB_data, *host_chB_out, *host_chAsq_out, *host_cbBsq_out;

//     cudaHostAlloc()

//         // Launch two streams, dealing with alternating chunks
//         // steam0       stream1     stream0      stream1
//         // a1a2a3a4.... b1b2b3b4... c1c2c3c4.... d1d2d3d4...
//         //
//         // - Each chunk has length SP_POINTS
//         // - There are R_POINTS/R_POINTS_CHUNK total chunks to iterate through
//         cudaStream_t stream0, stream1;
//     cudaStreamCreate(&stream0); cudaStreamCreate(&stream1);
//     for (int i(0); i < R_POINTS; i+=2*R_POINTS_CHUNK){
//         // Memcpy, Kernel, Memcpy must be in the order below (breadth first), where copy
//         // operations are batched to prevent blocking of kernel execution

//         // Copy input data over to GPU asynchornously in chunks
//         // Dereference the gpu_ch? (which is the address where the GPU memory location is kept)
//         // in order to get the actual memory location
//         cudaMemcpyAsync(*gpu_chA_data, chA_data + i*SP_POINTS,
//                         R_POINTS_CHUNK * SP_POINTS * sizeof(short),
//                         cudaMemcpyHostToDevice,
//                         stream0);
//         cudaMemcpyAsync(*gpu_chA_data, chA_data + (i+R_POINTS_CHUNK)*SP_POINTS,
//                         R_POINTS_CHUNK * SP_POINTS * sizeof(short),
//                         cudaMemcpyHostToDevice,
//                         stream1);
//         cudaMemcpyAsync(*gpu_chB_data, chB_data+i,
//                         R_POINTS_CHUNK * SP_POINTS * sizeof(short),
//                         cudaMemcpyHostToDevice,
//                         stream0);
//         cudaMemcpyAsync(*gpu_chB_data, chB_data + (i+R_POINTS_CHUNK)*SP_POINTS,
//                         R_POINTS_CHUNK * SP_POINTS * sizeof(short),
//                         cudaMemcpyHostToDevice,
//                         stream1);

//         // Run kernel(s)
//         power_kernel_runner<<<BLOCKS, THREADS_PER_BLOCK, 0, stream0>>>(
//             *gpu_chA_data, *gpu_chB_data,
//             *gpu_chA_out, *gpu_chB_out,
//             *gpu_chAsq_out, *gpu_chBsq_out);

//         power_kernel_runner<<<BLOCKS, THREADS_PER_BLOCK, 0, stream1>>>(
//             *gpu_chA_data, *gpu_chB_data,
//             *gpu_chA_out, *gpu_chB_out,
//             *gpu_chAsq_out, *gpu_chBsq_out);

//         // Copy from device
//         cudaMemcpyAsync(data_out[CHA], *gpu_chA_out,
//                         SP_POINTS * sizeof(double),
//                         cudaMemcpyDeviceToHost,
//                         stream0);
//         cudaMemcpyAsync(data_out[CHB], *gpu_chB_out,
//                         SP_POINTS * sizeof(double),
//                         cudaMemcpyDeviceToHost,
//                         stream0);
//         cudaMemcpyAsync(data_out[CHASQ],*gpu_chAsq_out,
//                         SP_POINTS * sizeof(double),
//                         cudaMemcpyDeviceToHost,
//                         stream0);
//         cudaMemcpyAsync(data_out[CHBSQ], *gpu_chBsq_out,
//                         SP_POINTS * sizeof(double),
//                         cudaMemcpyDeviceToHost,
//                         stream0);

//         // Manually evaluate sq = chAsq + chBsq
//         for (int i(0); i < SP_POINTS; i++)
//             data_out[SQ][i] = data_out[CHASQ][i] + data_out[CHBSQ][i];

//     }
//     cudaStreamSynchronize(stream0); cudaStreamSynchronize(stream1);
//     cudaStreamDestroy(stream0); cudaStreamDestroy(stream1);

//     // Ensure that free_memory is called ==>
// }
