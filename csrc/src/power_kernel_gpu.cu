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
 * Note that we need to dereference the addresses of the GPU memory pointer e.g. *gpu_in0[CJA]
 * In order to get the memory location on the gpu
 */

#include "colours.hpp"
#include "power_kernel.hpp"

// Background signal copied once to GPU
__constant__ short gpu_chA_background[SP_POINTS];
__constant__ short gpu_chB_background[SP_POINTS];

int odx; ///< Output inDeX for CHA, CHB, CHASQ ...

/*
  Call function to copy background for each channel a single time:
  * - gpu_chA_background
  * - gpu_chB_background
  */
void GPU::copy_background_arrays_to_gpu(short *chA_background, short *chB_background) {

    int success = 0;
    success += cudaMemcpyToSymbol(gpu_chA_background, chA_background,
                                  SP_POINTS*sizeof(short));
    success += cudaMemcpyToSymbol(gpu_chB_background, chB_background,
                                  SP_POINTS*sizeof(short));
    if (success != 0) FAIL(
        "Failed to copy background data TO the GPU - check that the arrays have SP_POINTS in them!");

}

/**
 * @copydoc ./power_kernel.cpp reduction_average
 */
__device__ void reduction_sum(
    long *chA_cumulative_array,
    long *chB_cumulative_array,
    long *chAsq_cumulative_array,
    long *chBsq_cumulative_array,
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

__global__ void power_kernel_runner(short *chA_data, short *chB_data,
                                    long *chA_out, long *chB_out,
                                    long *chAsq_out, long *chBsq_out){
    // long int will be able to hold 2^62/(2^14) data points from the 14bit digitiser
    // Since the SP digitizer will have a maximum of 254200 record (using the GetMaxNofRecordsFromNofSamples(adq_cu_ptr, 1))
    // This will be able to contain everything
    __shared__ long chA_cumulative_array[R_POINTS_PER_CHUNK];
    __shared__ long chB_cumulative_array[R_POINTS_PER_CHUNK];
    __shared__ long chAsq_cumulative_array[R_POINTS_PER_CHUNK];
    __shared__ long chBsq_cumulative_array[R_POINTS_PER_CHUNK];

    // Each block deals with a specific SP_POINT
    int sp_coordinate = blockIdx.x;

    // Each thread iterates over the repetitions for each SP_POINT
    int r_coordinate, coordinate;
    while ((sp_coordinate < SP_POINTS) && (threadIdx.x < R_POINTS_PER_CHUNK) ) {
        r_coordinate = threadIdx.x;

        while (r_coordinate < R_POINTS_PER_CHUNK) {
            coordinate = r_coordinate * SP_POINTS + sp_coordinate;

            chA_cumulative_array[r_coordinate] = chA_data[coordinate] - gpu_chA_background[sp_coordinate];
            chB_cumulative_array[r_coordinate] = chB_data[coordinate] - gpu_chB_background[sp_coordinate];

            chAsq_cumulative_array[r_coordinate] = chA_cumulative_array[r_coordinate] * chA_cumulative_array[r_coordinate];
            chBsq_cumulative_array[r_coordinate] = chB_cumulative_array[r_coordinate] * chB_cumulative_array[r_coordinate];

            // Once thread has completed, shift the
            // repetition index by the number of allocated
            // threads and continue summation
            r_coordinate += blockDim.x;
        }

        // Ensure that all threads have completed execution
        __syncthreads();

        // Summation and storage
        reduction_sum(chA_cumulative_array, chB_cumulative_array,
                      chAsq_cumulative_array, chBsq_cumulative_array,
                      R_POINTS_PER_CHUNK);
        // chA_out[sp_coordinate] += 1;//chA_cumulative_array[0];
        // chB_out[sp_coordinate] += 10;//chB_cumulative_array[0];
        // chAsq_out[sp_coordinate] += 100;//chAsq_cumulative_array[0];
        // chBsq_out[sp_coordinate] += 1000;//chBsq_cumulative_array[0];
        chA_out[sp_coordinate] += chA_cumulative_array[0];
        chB_out[sp_coordinate] += chB_cumulative_array[0];
        chAsq_out[sp_coordinate] += chAsq_cumulative_array[0];
        chBsq_out[sp_coordinate] += chBsq_cumulative_array[0];

        // Shift by number of allocated blocks along main-axis
        sp_coordinate += gridDim.x;
    }
}

/**
 * Data from `cpu_out` is appended to `data_out`.
 */
void accumulate(long** data_out, long ***cpu_out, int no_streams, int no_chunks){
    int odx;

    for (int sp(0); sp < SP_POINTS; sp++) {
        for (int i(0); i < GPU::no_outputs_from_gpu; i++) {
            odx = GPU::outputs_from_gpu[i];
            for (int s(0); s < no_streams; s++)
                data_out[odx][sp] += cpu_out[s][odx][sp];
        }
        data_out[SQ][sp] = data_out[CHASQ][sp] + data_out[CHBSQ][sp];
    }
}

/**
 * Data from `cpu_out` is appended to `data_out` **and normalised**.
 */
void accumulate(double** data_out, long ***cpu_out, int no_streams, int no_chunks){
    int odx;

    for (int sp(0); sp < SP_POINTS; sp++) {
        for (int i(0); i < GPU::no_outputs_from_gpu; i++) {
            odx = GPU::outputs_from_gpu[i];
            for (int s(0); s < no_streams; s++)
                data_out[odx][sp] += (double)cpu_out[s][odx][sp];
            data_out[odx][sp] /= no_chunks;
        }
        data_out[SQ][sp] = data_out[CHASQ][sp] + data_out[CHBSQ][sp];
    }
}

template<typename T> int GPU::power_kernel(
    short *chA_data, short *chB_data,
    T **data_out,
    short ***gpu_in, long ***gpu_out, long ***cpu_out, int no_streams){
    /**
     * ==> Ensure that allocate_memory has been called
     * ==> Ensure that background arrays (set to 0 for no correction) have been copied over
     *
     * Launch streams, dealing with alternating no_chunks
     * steam0       stream1     stream0      stream1
     * a1a2a3a4.... b1b2b3b4... c1c2c3c4.... d1d2d3d4...
     *
     * - Each chunk has length SP_POINTS
     * - There are R_POINTS/R_POINTS_PER_CHUNK total no_chunks to iterate through, split evenly between the streams
     */

    int success = 0;
    int odx; // ouput index, CHA, CHB ...
    int no_chunks = R_POINTS / R_POINTS_PER_CHUNK;

    // Create streams
    cudaStream_t *stream_list = new cudaStream_t[no_streams];
    for (int s(0); s < no_streams; s++){
        stream_list[s] = cudaStream_t();
        cudaStreamCreate(&stream_list[s]);
    }

    // Reset cumulative arrays
    for (int s(0); s < no_streams; s++) {
        for (int i(0); i < GPU::no_outputs_from_gpu; i++) {
            odx = GPU::outputs_from_gpu[i];
            success += cudaMemsetAsync(gpu_out[s][odx], 0, SP_POINTS*sizeof(long), stream_list[s]);
            if (success != 0) FAIL("Power Kernel: Failed to reset arrays on GPU:\nError code %i.", success);
        }
    }

    for (int chunk0(0); chunk0 < no_chunks; chunk0+=no_streams){
        // Memcpy -> Kernel -> Memcpy must be in the order below (breadth first)
        // to prevent blocking of kernel execution by batching of similar commands
        // Therefore, do NOT try to combine the for loops: copying must be issued first, then the kernels, then the copying again

        // Copy over the chA and chB data in no_chunks to each stream
        // - (chunk0 + s) * R_POINTS_PER_CHUNK * SP_POINTS  provides the correct offset when copying input data
        // - R_POINTS_PER_CHUNK * SP_POINTS * sizeof(short)  specifies how many bytes to copy
        for (int s(0); s < no_streams; s++){
            cudaMemcpyAsync(
                gpu_in[s][CHA], chA_data + (chunk0 + s) * R_POINTS_PER_CHUNK * SP_POINTS,
                R_POINTS_PER_CHUNK * SP_POINTS * sizeof(short),
                cudaMemcpyHostToDevice,
                stream_list[s]);
            cudaMemcpyAsync(
                gpu_in[s][CHB], chB_data + (chunk0 + s) * R_POINTS_PER_CHUNK * SP_POINTS,
                R_POINTS_PER_CHUNK * SP_POINTS * sizeof(short),
                cudaMemcpyHostToDevice,
                stream_list[s]);
        }

        // Launch kernels
        for (int s(0); s < no_streams; s++){
            power_kernel_runner<<<BLOCKS, THREADS_PER_BLOCK, 0, stream_list[s]>>>(
                gpu_in[s][CHA], gpu_in[s][CHB],
                gpu_out[s][CHA], gpu_out[s][CHB],
                gpu_out[s][CHASQ], gpu_out[s][CHBSQ]);
        }
    }
    // Copy over accumulated data from GPU to CPU
    for (int s(0); s < no_streams; s++){
        for (int i(0); i < GPU::no_outputs_from_gpu; i++) {
            odx = GPU::outputs_from_gpu[i];
            cudaMemcpyAsync(
                cpu_out[s][odx],
                gpu_out[s][odx],
                SP_POINTS * sizeof(long),
                cudaMemcpyDeviceToHost,
                stream_list[s]);
        }
    }

    // Await and close each stream
    for (int s(0); s < no_streams; s++) {
        cudaStreamSynchronize(stream_list[s]);
        cudaStreamDestroy(stream_list[s]);
    }
    delete[] stream_list;

    // Sum up totals from the different streams:
    // - If data_out is of type double**, normalise by the number of chunks for a ready result.
    // - If data_out is of type long**, only accumulate the data to normalise later on.
    accumulate(data_out, cpu_out, no_streams, no_chunks);

    /** Ensure that free_memory is called ==> */
    return no_chunks;
}

template int GPU::power_kernel<double>(
    short *chA_data, short *chB_data,
    double **data_out,
    short ***gpu_in, long ***gpu_out, long ***cpu_out, int no_streams); ///< When data_out is passed in as `double**` it is infered that data should be normalised as this is a single run.

template int GPU::power_kernel<long>(
    short *chA_data, short *chB_data,
    long **data_out,
    short ***gpu_in, long ***gpu_out, long ***cpu_out, int no_streams); ///< When data_out is paseed in as `long**` it is infered that data should be accumulated, as there will be further repititions.
