/*
 * Everything that can be compiled without the nvcc for the GPU power kernel
 * - Allocating data on device
 * - Fetching kernel parameters
 */

#include <cuda_runtime.h> // cudaMalloc cudaFree
#include <cuda_runtime_api.h> //for cudaDeviceProp
#include <limits.h> // For LONG_MAX
#include <string> // For std::to_string

#include "logging.hpp" // RED, OKBLUE etc
#include "power_kernel.hpp" // for power kernel parameters
#include "sp_digitiser.hpp" // for digitiser parameters MAX_DIGITISER_CODE and MAX_NUMBER_OF_RECORDS
#include "utils_gpu.hpp" // To fetch GPU parameters

int GPU::fetch_power_kernel_threads() {
    return THREADS_PER_BLOCK;
};
int GPU::fetch_power_kernel_blocks() {
    return BLOCKS;
};

int GPU::check_power_kernel_parameters(bool display){
    PYTHON_START;

    // For reduction summation on GPU, this needs to be a power of 2
    if ((R_POINTS_PER_GPU_CHUNK & (R_POINTS_PER_GPU_CHUNK - 1)) != 0)
        FAIL(
            "R_POINTS_PER_GPU_CHUNK="
            + std::to_string(R_POINTS_PER_GPU_CHUNK)
            + " needs to be a power of 2 (in order to perform efficient reduction summation on GPU).");

    // Check that "chunking" is small enogh to not exceeed the shared memory available on GPU
    cudaDeviceProp prop = fetch_gpu_parameters();
    const int shared_memory_required = (R_POINTS_PER_GPU_CHUNK
                                        * sizeof(long)
                                        * GPU::no_outputs_from_gpu);
    if (prop.sharedMemPerBlock < shared_memory_required)
        FAIL(
            "Power Kernel: Not enough shared memory on GPU.\n\tR_POINTS_PER_GPU_CHUNK ("
            + std::to_string(R_POINTS_PER_GPU_CHUNK)
            + ") x LONG (8) x arrays used in GPU ("
            + std::to_string(GPU::no_outputs_from_gpu)
            + ") = "
            + std::to_string(shared_memory_required)
            + " > "
            + std::to_string(prop.sharedMemPerBlock)
            + " bytes availabel per GPU block.");

    // Check that chunking of R_POINTS for processing on the gpu is valid
    if (R_POINTS < R_POINTS_PER_GPU_CHUNK)
        FAIL(
            "R_POINTS ("
            + std::to_string(R_POINTS)
            + ") < R_POINTS_PER_GPU_CHUNK ("
            + std::to_string(R_POINTS_PER_GPU_CHUNK)
            + "): Chunking (for processing on gpu) is bigger than amount of repititions on digitiser. You will probably want to adjust R_POINTS"
            );
    if ((R_POINTS - (R_POINTS / R_POINTS_PER_GPU_CHUNK) * R_POINTS_PER_GPU_CHUNK) != 0)
        FAIL(
            "R_POINTS_PER_GPU_CHUNK ("
            + std::to_string(R_POINTS_PER_GPU_CHUNK)
            + ") does not fit fully into R_POINTS ("
            + std::to_string(R_POINTS)
            + ").");
    if ((R_POINTS / R_POINTS_PER_GPU_CHUNK % 2) != 0)
        FAIL(
            "R_POINTS_PER_GPU_CHUNK ("
            + std::to_string(R_POINTS_PER_GPU_CHUNK)
            + ") does not chunk R_POINTS ("
            + std::to_string(R_POINTS)
            + ") evenly across the 2 parallel streams that will run on GPU.");

    // Ensure that the cumulative arrays will not overflow.
    if ((MAX_DIGITISER_CODE * MAX_NUMBER_OF_RECORDS > LONG_MAX) ||
        (MAX_DIGITISER_CODE * MAX_DIGITISER_CODE * MAX_NUMBER_OF_RECORDS > LONG_MAX) ||
        (2 * MAX_DIGITISER_CODE * MAX_DIGITISER_CODE * MAX_NUMBER_OF_RECORDS > LONG_MAX))
        FAIL(
            "Cumulative arrays will not be able to hold all the intermediate processing data for power measurements");

    // Check that global memory is not exceeded
    long gpu_global_memory_allocation = (
        SP_POINTS * R_POINTS_PER_GPU_CHUNK * sizeof(short) // chA
        + SP_POINTS * R_POINTS_PER_GPU_CHUNK * sizeof(short) // chB
        + SP_POINTS * sizeof(long) * GPU::no_outputs_from_gpu // output
        );
    if (gpu_global_memory_allocation > (long)prop.totalGlobalMem)
        FAIL(
            "Input arrays for chA and chB of type" + std::string("(short) and ")
            + std::to_string(GPU::no_outputs_from_gpu) + "x output arrays of type (long)"
            + "allocated in the gpu (" + std::to_string(gpu_global_memory_allocation)
            + "bytes) bigger than logbal memory on GPU (" + std::to_string(prop.totalGlobalMem) + "bytes)."
            );


    if (display) {
        OKBLUE("===========================================");
        RED("          **POWER KERNEL**");

        OKBLUE("Data Parameters");
        WHITE("R_POINTS: %i\n", R_POINTS );
        WHITE("SP_POINTS: %i\n", SP_POINTS );

        OKBLUE("Processing Parameters");
        WHITE("BLOCKS: %i\n", BLOCKS );
        WHITE("THREADS_PER_BLOCK: %i\n", THREADS_PER_BLOCK );

        OKBLUE("===========================================");
    }

    PYTHON_END;

    return 0;
}

void GPU::allocate_memory(
    short **chA_data, short **chB_data,
    short ****gpu_in, long****gpu_out, long ****cpu_out, int no_streams){
    /** There is a lot of derefenecing in this function, since the arrays ara passed in by address & */

    OKBLUE("Power Kernel: Allocating memory on GPU and CPU.");
    int chunks = R_POINTS / R_POINTS_PER_GPU_CHUNK;
    if (chunks - (chunks / no_streams) * no_streams != 0)
        FAIL(
            "Power Kernel: no_streams ("
            + std::to_string(no_streams)
            + ") does no fit fully into R_POINTS/R_POINTS_PER_GPU_CHUNK ("
            + std::to_string(R_POINTS) + "/" + std::to_string(R_POINTS_PER_GPU_CHUNK)
            + ") = " + std::to_string(chunks));

        int success = 0; int odx;

        // Digitiser will transfer data into memory-locked arrays, made with cudaHostAlloc
        if (chA_data != 0)
            success += cudaHostAlloc((void**)chA_data,
                                     SP_POINTS * R_POINTS * sizeof(short),
                                     cudaHostAllocDefault);
        if (chB_data != 0)
            success += cudaHostAlloc((void**)chB_data,
                                     SP_POINTS * R_POINTS * sizeof(short),
                                     cudaHostAllocDefault);
        if (success != 0) FAIL("Power Kernel: Failed to allocate locked input memory on CPU.");


        // Input data is fed in chunks of R_POINTS_PER_GPU_CHUNK * SP_POINTS
        // (gpu_in is passed in by address, so dereference first, and assign it to a new 2D array of stream x channels)
        if (gpu_in != 0) {
            (*gpu_in) = new short**[no_streams];
            for (int s(0); s < no_streams; s++){
                // (each stream will have entries for chA and chB)
                (*gpu_in)[s] = new short*[2];

                // (chA and chB will store the address (short*) of the memory allocated on GPU)
                // (- they need to be passed in by address & in order for cudaMalloc to update their values)
                success += cudaMalloc((void**)&(*gpu_in)[s][CHA], SP_POINTS * R_POINTS_PER_GPU_CHUNK * sizeof(short));
                success += cudaMalloc((void**)&(*gpu_in)[s][CHB], SP_POINTS * R_POINTS_PER_GPU_CHUNK * sizeof(short));
            }
        }
        if (success != 0) FAIL("Power Kernel: Failed to allocate input memory on GPU.");

        if (gpu_out != 0 && cpu_out != 0) {
            // Each stream will have it's dedicated arrays for writting results to
            (*gpu_out) = new long**[no_streams];
            (*cpu_out) = new long**[no_streams];
            for (int s(0); s < no_streams; s++) {
                (*gpu_out)[s] = new long*[GPU::no_outputs_from_gpu];
                (*cpu_out)[s] = new long*[GPU::no_outputs_from_gpu];
                for (int i(0); i < GPU::no_outputs_from_gpu; i++) {
                    odx = GPU::outputs_from_gpu[i];

                    // Processed Output data will accumulate on the GPU for each stream
                    success += cudaMalloc((void**)&(*gpu_out)[s][odx], SP_POINTS * sizeof(long));
                    if (success != 0) FAIL("Power Kernel: Failed to allocate output memory on GPU.");

                    // And will be copied and summed on the cpu
                    success += cudaHostAlloc((void**)&(*cpu_out)[s][odx], SP_POINTS * sizeof(long), cudaHostAllocDefault);
                    if (success != 0) FAIL("Power Kernel: Failed to allocate locked output memory on CPU.");
                }
            }
        }

        OKGREEN("Power Kernel: Allocation done!");
    }

void GPU::free_memory(
    short *chA_data, short *chB_data,
    short ***gpu_in, long ***gpu_out, long ***cpu_out, int no_streams){

    OKBLUE("Power Kernel: Deallocating memory on GPU and CPU.");
    int success = 0; int odx;

    if (chA_data != 0)
        success += cudaFreeHost(chA_data);
    if (chB_data != 0)
        success += cudaFreeHost(chB_data);
    if (success != 0) FAIL("Power Kernel: Failed to free locked input memory on CPU.");

    if (gpu_in != 0) {
        for (int s(0); s < no_streams; s++) {
            success += cudaFree(gpu_in[s][CHA]);
            success += cudaFree(gpu_in[s][CHB]);
            delete[] gpu_in[s];
        }
        delete[] gpu_in;
    }
    if (success != 0) FAIL("Power Kernel: Failed to free input memory on GPU.");

    if (gpu_out != 0 && cpu_out != 0) {
        for (int s(0); s < no_streams; s++) {
            for (int i(0); i < GPU::no_outputs_from_gpu; i++) {
                odx = GPU::outputs_from_gpu[i];

                success += cudaFree(gpu_out[s][odx]);
                if (success != 0) FAIL("Power Kernel: Failed to free output memory on GPU.");

                success += cudaFreeHost(cpu_out[s][odx]);
                if (success != 0) FAIL("Power Kernel: Failed to free locked outputa memory on CPU.");
            }
            delete[] gpu_out[s];
            delete[] cpu_out[s];
        }
        delete[] gpu_out;
        delete[] cpu_out;
    }

    OKGREEN("Power Kernel: Memory freed!");
}
