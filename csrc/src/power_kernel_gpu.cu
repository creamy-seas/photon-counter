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
#include "power_kernel.hpp"

__device__ void reduction_sum(
    unsigned long int chA_cumulative_array[R_POINTS],
    unsigned long int chB_cumulative_array[R_POINTS],
    unsigned long int chAsq_cumulative_array[R_POINTS],
    unsigned long int chBsq_cumulative_array[R_POINTS]
    ){
    /*
     * Reduce the array by summing up the total into the first cell.
     CUDA: Threads along the R_POINTS axis

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
                chAsq_cumulative_array[r_coordinate] += chAsq_cumulative_array[r_coordinate + idx];
                chBsq_cumulative_array[r_coordinate] += chBsq_cumulative_array[r_coordinate + idx];
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

/*
  Reduce the chA_data and chB_data into processed arrays.
  CUDA: Each block deals with a specific SP_POINT
  CUDA: Each thread iterates over R_POINTS for each SP_POINT
*/
__global__ void power_kernel_runner(
    short *chA_data, short *chB_data,
    double *chA_out, double *chB_out,
    double *chAsq_out, double *chBsq_out
    ){

    // An unsigned long int will be able to hold 2^32/(2^14) = 2^18 = 262144 data points from the 14bit digitiser
    // Since the SP digitizer will have a maximum of 254200 record (using the GetMaxNofRecordsFromNofSamples(adq_cu_ptr, 1))
    // This will be able to contain everything
    __shared__ unsigned long int chA_cumulative_array[R_POINTS];
    __shared__ unsigned long int chB_cumulative_array[R_POINTS];
    __shared__ unsigned long int chAsq_cumulative_array[R_POINTS];
    __shared__ unsigned long int chBsq_cumulative_array[R_POINTS];

    int sp_coordinate = blockIdx.x;
    int r_coordinate, coordinate;

    while (sp_coordinate < SP_POINTS) {
        r_coordinate = threadIdx.x;

        while (r_coordinate < R_POINTS) {
            coordinate = r_coordinate * SP_POINTS + sp_coordinate;

            chA_cumulative_array[r_coordinate] = chA_data[coordinate] - dev_chA_background[sp_coordinate];
            chB_cumulative_array[r_coordinate] = chB_data[coordinate] - dev_chB_background[sp_coordinate];

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
                      chBsq_cumulative_array);
        chA_out[sp_coordinate] = (double)chA_cumulative_array[0] / R_POINTS;
        chB_out[sp_coordinate] = (double)chB_cumulative_array[0] / R_POINTS;
        chAsq_out[sp_coordinate] = (double)chAsq_cumulative_array[0] / R_POINTS;
        chBsq_out[sp_coordinate] = (double)chBsq_cumulative_array[0] / R_POINTS;

        // Shift by number of allocated blocks along main-axis
        sp_coordinate += gridDim.x;
    }
}

void GPU::power_kernel(
    short *chA_data,
    short *chB_data,
    double **data_out,
    short **dev_chA_data,
    short **dev_chB_data,
    double **dev_chA_out,
    double **dev_chB_out,
    double **dev_chAsq_out,
    double **dev_chBsq_out
    ){
    /*
     * chA and chB arrays:
     a1 a2 a3 a4 ... b1 b2 b3 b4 ... c1 c2 c3 c4 ...
    */

    // ==> Ensure that allocate_memory_on_gpu has been called
    // ==> Ensure that background arrays (set to 0 for no correction) have been copied over

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
    power_kernel_runner<<<BLOCKS, THREADS_PER_BLOCK>>>(
        *dev_chA_data, *dev_chB_data,
        *dev_chA_out, *dev_chB_out,
        *dev_chAsq_out, *dev_chBsq_out);

    // Copy from device
    success += cudaMemcpy(data_out[CHA], *dev_chA_out,
                          SP_POINTS * sizeof(double),
                          cudaMemcpyDeviceToHost);
    success += cudaMemcpy(data_out[CHB], *dev_chB_out,
                          SP_POINTS * sizeof(double),
                          cudaMemcpyDeviceToHost);
    success += cudaMemcpy(data_out[CHASQ],*dev_chAsq_out,
                          SP_POINTS * sizeof(double),
                          cudaMemcpyDeviceToHost);
    success += cudaMemcpy(data_out[CHBSQ], *dev_chBsq_out,
                          SP_POINTS * sizeof(double),
                          cudaMemcpyDeviceToHost);
    if (success != 0) FAIL("Failed to copy data FROM the GPU!");

    // Manually evaluate sq = chAsq + chBsq
    for (int i(0); i < SP_POINTS; i++)
        data_out[SQ][i] = data_out[CHASQ][i] + data_out[CHBSQ][i];

    // Ensure that free_memory_on_gpu is called ==>
}
