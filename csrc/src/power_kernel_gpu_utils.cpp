/*
 * Everything that can be compiled without the nvcc for the GPU power kernel
 * - Allocating data on device
 * - Fetching kernel parameters
 */

#include <cuda_runtime.h> // cudaMalloc cudaFree
#include <cuda_runtime_api.h> //for cudaDeviceProp
#include <limits.h>

#include <string>
#include "colours.hpp" // RED, OKBLUE etc
#include "power_kernel.hpp" // for power kernel parameters
#include "ia_ADQAPI.hpp" // for digitiser parameters MAX_CODE and MAX_NUMBER_OF_RECORDS
#include "utils_gpu.hpp" // To fetch GPU parameters

GPU::PowerKernelParameters::PowerKernelParameters(
    int r_points,
    int np_points,
    int blocks,
    int threads_per_block
    ){
    this->r_points = r_points;
    this->np_points = np_points;
    this->blocks = blocks;
    this->threads_per_block = threads_per_block;

    this->print();
}

void GPU::PowerKernelParameters::print(){
    OKBLUE("===========================================");
    RED("          **POWER KERNEL**");

    OKBLUE("Data Parameters");
    WHITE("R_POINTS: %i\n", this->r_points );
    WHITE("SP_POINTS: %i\n", this->np_points );

    OKBLUE("Processing Parameters");
    WHITE("BLOCKS: %i\n", this->blocks );
    WHITE("THREADS_PER_BLOCK: %i\n", this->threads_per_block );

    OKBLUE("===========================================");
}

GPU::PowerKernelParameters GPU::fetch_kernel_parameters(){
    // Even number required for summation on GPU
    if (R_POINTS % 2 != 0)
        throw std::runtime_error(
            "R_POINTS="
            + std::to_string(R_POINTS)
            + " needs to be a even number.");

    // Check that "chunking" falls within the limits of shared memory on GPU
    cudaDeviceProp prop = fetch_gpu_parameters();
    const int number_of_cumulative_arrays = 4;
    const int shared_memory_required = (R_POINTS_PER_CHUNK
                                        * sizeof(long)
                                        * number_of_cumulative_arrays);
    if (prop.sharedMemPerBlock < shared_memory_required)
        throw std::runtime_error(
            "Not enough shared memory on GPU ("
            + std::to_string(shared_memory_required)
            + " > "
            + std::to_string(prop.sharedMemPerBlock)
            + " bytes) for using R_POINTS_PER_CHUNK="
            + std::to_string(R_POINTS_PER_CHUNK)
            + " in power mesurements."
            );

    // Check that chunking of R_POINTS is valid
    if (R_POINTS < R_POINTS_PER_CHUNK)
        throw std::runtime_error(
            "R_POINTS ("
            + std::to_string(R_POINTS)
            + ") < R_POINTS_PER_CHUNK ("
            + std::to_string(R_POINTS_PER_CHUNK)
            + "): Chunking is bigger than amount of repititions on digitiser."
            );
    if ((R_POINTS - (R_POINTS / R_POINTS_PER_CHUNK) * R_POINTS_PER_CHUNK) != 0)
        throw std::runtime_error(
            "R_POINTS_PER_CHUNK ("
            + std::to_string(R_POINTS_PER_CHUNK)
            + ") does not fit fully into R_POINTS ("
            + std::to_string(R_POINTS)
            + ").");
    if ((R_POINTS / R_POINTS_PER_CHUNK % 2) != 0)
            throw std::runtime_error(
                "R_POINTS_PER_CHUNK ("
                + std::to_string(R_POINTS_PER_CHUNK)
                + ") does not chunk R_POINTS ("
                + std::to_string(R_POINTS)
                + ") evenly across the 2 streams.");

    // Ensure that the cumulative arrays will not overflow.
    if ((MAX_CODE * MAX_NUMBER_OF_RECORDS > LONG_MAX) ||
        (MAX_CODE * MAX_CODE * MAX_NUMBER_OF_RECORDS > LONG_MAX) ||
        (2 * MAX_CODE * MAX_CODE * MAX_NUMBER_OF_RECORDS > LONG_MAX))
        throw std::runtime_error(
            "Cumulative arrays will not be able to hold all the intermediate processing data for power measurements");

    // Reset is checked in python
    return GPU::PowerKernelParameters(
        R_POINTS,
        SP_POINTS,
        BLOCKS,
        THREADS_PER_BLOCK
        );
}

void GPU::allocate_memory(
    short **chA_data, short **chB_data,
    short ****gpu_in, long****gpu_out, long ****cpu_out, int no_streams){

    OKBLUE("Power Kernel: Allocating memory on GPU and CPU.");
    int success = 0; int odx;

    // Digitiser will transfer data into memory-locked arrays, made with cudaHostAlloc
    success += cudaHostAlloc((void**)chA_data,
                             SP_POINTS * R_POINTS * sizeof(short),
                             cudaHostAllocDefault);
    success += cudaHostAlloc((void**)chB_data,
                             SP_POINTS * R_POINTS * sizeof(short),
                             cudaHostAllocDefault);
    if (success != 0) FAIL("Power Kernel: Failed to allocate locked input memory on CPU.");

    // Input data is fed in chunks of R_POINTS_PER_CHUNK * SP_POINTS to the streams
    for (int s(0); s < no_streams; s++){
        success += cudaMalloc((void**)gpu_in[s][CHA], SP_POINTS * R_POINTS_PER_CHUNK * sizeof(short));
        success += cudaMalloc((void**)gpu_in[s][CHB], SP_POINTS * R_POINTS_PER_CHUNK * sizeof(short));
    }
    if (success != 0) FAIL("Power Kernel: Failed to allocate input memory on GPU.");

    // Output data will be written separately by each stream into a dedicated cpu_out array.
    // (*cpu_out) is used here, because we pass in an empty pointer BY REFFERENCE
    // in order to assign it to an array in the follow steps.
    // (*cpu_out) = new long**[no_streams];
    for (int s(0); s < no_streams; s++) {
        // (*cpu_out)[s] = new long*[GPU::no_outputs_from_gpu];

        for (int i(0); i < GPU::no_outputs_from_gpu; i++) {
            odx = GPU::outputs_from_gpu[i];

            // Processed Output data will accumulate on the GPU for each stream
            success += cudaMalloc((void**)gpu_out[s][odx], SP_POINTS * sizeof(long));
            if (success != 0) FAIL("Power Kernel: Failed to allocate output memory on GPU.");

            // And will be copied and summed on the cpu
            // (*cpu_out)[s][odx] = new long[SP_POINTS];
            success += cudaHostAlloc((void**)cpu_out[s][odx], SP_POINTS * sizeof(long), cudaHostAllocDefault);
            if (success != 0) FAIL("Power Kernel: Failed to allocate locked output memory on CPU.");
        }
    }

    OKGREEN("Power Kernel: Allocation done!");
}

void GPU::free_memory(
    short **chA_data, short **chB_data,
    short ****gpu_in, long ****gpu_out, long ****cpu_out, int no_streams){

    OKBLUE("Power Kernel: Deallocating memory on GPU and CPU.");
    int success = 0; int odx;

    success += cudaFreeHost(*chA_data);
    success += cudaFreeHost(*chB_data);
    if (success != 0) FAIL("Power Kernel: Failed to free locked input memory on CPU.");

    for (int s(0); s < no_streams; s++) {
        success += cudaFree(*gpu_in[s][CHA]);
        success += cudaFree(*gpu_in[s][CHB]);
    }
    if (success != 0) FAIL("Power Kernel: Failed to free input memory on GPU.");

    for (int s(0); s < no_streams; s++) {
        for (int i(0); i < GPU::no_outputs_from_gpu; i++) {
            odx = GPU::outputs_from_gpu[i];

            success += cudaFree(*gpu_out[s][odx]);
            if (success != 0) FAIL("Power Kernel: Failed to free output memory on GPU.");

            success += cudaFreeHost(*cpu_out[s][odx]);
            if (success != 0) FAIL("Power Kernel: Failed to free locked outputa memory on CPU.");

            // delete[] cpu_out[s][odx];
        }
        // delete[] cpu_out[s];
    }
    // delete[] cpu_out;

    OKGREEN("Power Kernel: Memory freed!");
}
