/*
 * Everything that can be compiled without the nvcc for the GPU power kernel
 * - Allocating data on device
 * - Fetching kernel parameters
 */

#include <cuda_runtime.h> // cudaMalloc cudaFree

#include "colours.hpp" // RED, OKBLUE etc
#include "power_kernel.hpp" // for power kernel parameters
#include "ia_ADQAPI.hpp" // for digitiser parameters MAX_CODE and MAX_NUMBER_OF_RECORDS

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
        throw std::runtime_error("R_POINTS needs to be a even number");

    // Ensure that the cumulative arrays will not overflow.
    unsigned long int max = -1UL;
    if ((MAX_CODE * MAX_NUMBER_OF_RECORDS > max) ||
        (MAX_CODE * MAX_CODE * MAX_NUMBER_OF_RECORDS > max) ||
        (2 * MAX_CODE * MAX_CODE * MAX_NUMBER_OF_RECORDS > max))
        throw std::runtime_error(
            "Cumulative arrays will not be able to hold all the intermediate processing data for power measurements");

    return GPU::PowerKernelParameters(
            R_POINTS,
            SP_POINTS,
            BLOCKS,
            THREADS_PER_BLOCK
            );
}

void GPU::allocate_memory_on_gpu(
    short **dev_chA_data,
    short **dev_chB_data,
    double **dev_chA_out, double **dev_chB_out,
    double **dev_chAsq_out, double **dev_chBsq_out
    ){
    // Pass in ADDRESS of pointers (&POINTER) -> this will allocate memory on GPU

    OKBLUE("Allocating memory for power kernel on GPU");

    int success = 0;
    success += cudaMalloc((void**)dev_chA_data,
                          TOTAL_POINTS * sizeof(short));
    success += cudaMalloc((void**)dev_chB_data,
                          TOTAL_POINTS * sizeof(short));
    success += cudaMalloc((void**)dev_chA_out,
                          SP_POINTS * sizeof(double));
    success += cudaMalloc((void**)dev_chB_out,
                          SP_POINTS * sizeof(double));
    success += cudaMalloc((void**)dev_chAsq_out,
                          SP_POINTS * sizeof(double));
    success += cudaMalloc((void**)dev_chBsq_out,
                          SP_POINTS * sizeof(double));

    if (success != 0) FAIL("Failed to allocate memory on GPU");

    OKGREEN("Allocation done!");
}

void GPU::free_memory_on_gpu(
    short **dev_chA_data,
    short **dev_chB_data,
    double **dev_chA_out, double **dev_chB_out,
    double **dev_chAsq_out, double **dev_chBsq_out
    ){
    // Call to deallocated memory on GPU after run is complete

    OKBLUE("Deallocating memory from GPU");

    int success = 0;
    success += cudaFree(*dev_chA_data);
    success += cudaFree(*dev_chB_data);
    success += cudaFree(*dev_chA_out);
    success += cudaFree(*dev_chB_out);
    success += cudaFree(*dev_chAsq_out);
    success += cudaFree(*dev_chBsq_out);
    if (success != 0)
        FAIL("Failed to allocate memory on GPU");

    OKGREEN("Memory freed!");
}
