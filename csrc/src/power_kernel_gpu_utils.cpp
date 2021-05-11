#include <cuda_runtime.h> // cudaMalloc cudaFree

#include "colours.hpp"
#include "power_kernel.hpp"
#include "gpu_utils.hpp"

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
    WHITE("R_POINTS: %i\n", this->r_points );
    WHITE("SP_POINTS: %i\n", this->np_points );

    OKBLUE("Processing Parameters");
    WHITE("CACHED ARRAY TYPE: %s\n", this->processing_array_type.c_str() );
    WHITE("BLOCKS: %i\n", this->blocks );
    WHITE("THREADS_PER_BLOCK: %i\n", this->threads_per_block );

    OKBLUE("===========================================");
}

GPU::PowerKernelParameters GPU::fetch_kernel_parameters(){
    if (R_POINTS % 2 != 0)
        throw std::runtime_error("R_POINTS needs to be a even number");

    return GPU::PowerKernelParameters(
        R_POINTS,
        SP_POINTS,
        xstr(PROCESSING_ARRAY_TYPE),
        BLOCKS,
        THREADS_PER_BLOCK
        );
}

// Pass in ADDRESS of pointers (&POINTER) -> this will allocate memory on GPU
void GPU::allocate_memory_on_gpu(
    short **dev_chA_data,
    short **dev_chB_data,
    double **dev_chA_out, double **dev_chB_out,
    double **dev_chAsq_out, double **dev_chBsq_out,
    double **dev_sq_out
    ){

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
    success += cudaMalloc((void**)dev_sq_out,
                          SP_POINTS * sizeof(double));

    if (success != 0) FAIL("Failed to allocate memory on GPU");

    OKGREEN("Allocation done!");
}

// Call to deallocated memory on GPU after run is complete
void GPU::free_memory_on_gpu(
    short **dev_chA_data,
    short **dev_chB_data,
    double **dev_chA_out, double **dev_chB_out,
    double **dev_chAsq_out, double **dev_chBsq_out,
    double **dev_sq_out
    ){

    OKBLUE("Deallocating memory from GPU");

    int success = 0;
    success += cudaFree(*dev_chA_data);
    success += cudaFree(*dev_chB_data);
    success += cudaFree(*dev_chA_out);
    success += cudaFree(*dev_chB_out);
    success += cudaFree(*dev_chAsq_out);
    success += cudaFree(*dev_chBsq_out);
    success += cudaFree(*dev_sq_out);
    if (success != 0)
        FAIL("Failed to allocate memory on GPU");

    OKGREEN("Memory freed!");
}
