/*
 * Everything that can be compiled without the nvcc for the GPU power kernel
 * - Allocating data on device
 * - Fetching kernel parameters
 */

#include <cuda_runtime.h> // cudaMalloc cudaFree
#include <cuda_runtime_api.h> //for cudaDeviceProp

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
    int sizeof_unsigned_int = 4;
    int number_of_cumulative_arrays = 4;
    int shared_memory_required = (R_POINTS_CHUNK
                                  * sizeof_unsigned_int
                                  * number_of_cumulative_arrays);
    if (prop.sharedMemPerBlock < shared_memory_required)
        throw std::runtime_error(
            "Not enough shared memory on GPU ("
            + std::to_string(shared_memory_required)
            + " > "
            + std::to_string(prop.sharedMemPerBlock)
            + " bytes) for using R_POINTS_CHUNK="
            + std::to_string(R_POINTS_CHUNK)
            + " in power mesurements."
            );

    // Check that chunking of R_POINTS is valid
    if (R_POINTS < R_POINTS_CHUNK)
        throw std::runtime_error(
            "R_POINTS ("
            + std::to_string(R_POINTS)
            + ") < R_POINTS_CHUNK ("
            + std::to_string(R_POINTS_CHUNK)
            + "): Chunking is bigger than amount of repititions on digitiser."
            );
    if ((R_POINTS - (R_POINTS / R_POINTS_CHUNK) * R_POINTS_CHUNK) != 0)
        throw std::runtime_error(
            "R_POINTS_CHUNK ("
            + std::to_string(R_POINTS_CHUNK)
            + ") does not fit fully into R_POINTS ("
            + std::to_string(R_POINTS)
            + ").");
    if ((R_POINTS / R_POINTS_CHUNK % 2) != 0)
            throw std::runtime_error(
                "R_POINTS_CHUNK ("
                + std::to_string(R_POINTS_CHUNK)
                + ") does not chunk R_POINTS ("
                + std::to_string(R_POINTS)
                + ") evenly across the 2 streams.");

    // Ensure that the cumulative arrays will not overflow.
    unsigned long int max = -1UL;
    if ((MAX_CODE * MAX_NUMBER_OF_RECORDS > max) ||
        (MAX_CODE * MAX_CODE * MAX_NUMBER_OF_RECORDS > max) ||
        (2 * MAX_CODE * MAX_CODE * MAX_NUMBER_OF_RECORDS > max))
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

void GPU::V1::allocate_memory(short ***gpu_in, double ***gpu_out){
    // Pass in ADDRESS of pointers (&POINTER) -> this will allocate memory on GPU
    int success = 0;
    OKBLUE("Power Kernel: Allocating memory on GPU and CPU.");

    // Input data is fed in chunks of R_POINTS_CHUNK * SP_POINTS
    success += cudaMalloc((void**)gpu_in[CHA], SP_POINTS * R_POINTS * sizeof(short));
    success += cudaMalloc((void**)gpu_in[CHB], SP_POINTS * R_POINTS * sizeof(short));

    // Output data is read out from GPU and stored in persistent memory on CPU
    for (int i(0); i < GPU::no_outputs_from_gpu; i++)
        success += cudaMalloc((void**)gpu_out[GPU::outputs_from_gpu[i]], SP_POINTS * sizeof(double));
    if (success != 0) FAIL("Power Kernel: Failed to allocate memory on GPU.");

    OKGREEN("Power Kernel: Allocation done!");
}

void GPU::V1::free_memory(short ***gpu_in, double ***gpu_out){
    // Call to deallocated memory on GPU after run is complete
    int success = 0;
    OKBLUE("Power Kernel: Deallocating memory on GPU.");
    success += cudaFree(*gpu_in[CHA]);
    success += cudaFree(*gpu_in[CHB]);

    for (int i(0); i < GPU::no_outputs_from_gpu; i++) {
        success += cudaFree(*gpu_out[GPU::outputs_from_gpu[i]]);
    }
    if (success != 0) FAIL("Power Kernel: Failed to free  memory on GPU.");

    OKGREEN("Power Kernel: Memory freed!");
}

void GPU::V2::allocate_memory(
    short ***gpu_in0,
    short ***gpu_in1,
    double ***gpu_out0,
    double ***gpu_out1,
    double ***cpu_out0,
    double ***cpu_out1
    ){
    int success = 0;

    OKBLUE("Power Kernel: Allocating memory on GPU and CPU.");

    // Input data is fed in chunks of R_POINTS_CHUNK * SP_POINTS
    success += cudaMalloc((void**)gpu_in0[CHA], SP_POINTS * R_POINTS_CHUNK * sizeof(short));
    success += cudaMalloc((void**)gpu_in0[CHB], SP_POINTS * R_POINTS_CHUNK * sizeof(short));
    success += cudaMalloc((void**)gpu_in1[CHA], SP_POINTS * R_POINTS_CHUNK * sizeof(short));
    success += cudaMalloc((void**)gpu_in1[CHB], SP_POINTS * R_POINTS_CHUNK * sizeof(short));

    // Output data is read out from GPU and stored in persistent memory on CPU
    for (int i(0); i < GPU::no_outputs_from_gpu; i++) {
        success += cudaMalloc((void**)gpu_out0[GPU::outputs_from_gpu[i]], SP_POINTS * sizeof(double));
        success += cudaMalloc((void**)gpu_out1[GPU::outputs_from_gpu[i]], SP_POINTS * sizeof(double));
        if (success != 0) FAIL("Power Kernel: Failed to allocate memory on GPU.");

        success += cudaHostAlloc((void**)cpu_out0[GPU::outputs_from_gpu[i]],
                                 SP_POINTS * R_POINTS/R_POINTS_CHUNK * sizeof(double),
                                 cudaHostAllocDefault);
        success += cudaHostAlloc((void**)cpu_out1[GPU::outputs_from_gpu[i]],
                                 SP_POINTS * R_POINTS/R_POINTS_CHUNK * sizeof(double),
                                 cudaHostAllocDefault);
        if (success != 0) FAIL("Power Kernel: Failed to allocate memory on CPU.");
    }

    OKGREEN("Power Kernel: Allocation done!");
}

void GPU::V2::free_memory(
    short ***gpu_in0, short ***gpu_in1,
    double ***gpu_out0, double ***gpu_out1,
    double ***cpu_out0, double ***cpu_out1){
    int success = 0;

    OKBLUE("Power Kernel: Deallocating memory on GPU.");
    success += cudaFree(*gpu_in0[CHA]);
    success += cudaFree(*gpu_in0[CHB]);
    success += cudaFree(*gpu_in1[CHA]);
    success += cudaFree(*gpu_in1[CHB]);

    for (int i(0); i < GPU::no_outputs_from_gpu; i++) {
        success += cudaFree(*gpu_out0[GPU::outputs_from_gpu[i]]);
        success += cudaFree(*gpu_out1[GPU::outputs_from_gpu[i]]);
        if (success != 0) FAIL("Power Kernel: Failed to free  memory on GPU.");

        success += cudaFreeHost(*cpu_out0[GPU::outputs_from_gpu[i]]);
        success += cudaFreeHost(*cpu_out1[GPU::outputs_from_gpu[i]]);
        if (success != 0) FAIL("Power Kernel: Failed to free memory on CPU.");
    }

    OKGREEN("Power Kernel: Memory freed!");
}
