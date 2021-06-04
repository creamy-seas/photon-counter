#include <stdio.h>
#include <sstream>
#include <string>

#include <cuda_runtime_api.h> //for cudaDeviceProp
#include "logging.hpp"
#include "utils_gpu.hpp"

cudaDeviceProp fetch_gpu_parameters(bool display){
    // Already implemented in python

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    if (display) {
        OKBLUE("===========================================");
        OKBLUE("Device parameters");
        WHITE("Name: %s\n", prop.name );
        WHITE("Compute capability: %d.%d\n", prop.major, prop.minor );
        WHITE("Clock rate: %d\n", prop.clockRate );
        WHITE("Device copy overlap: " );
        if (prop.deviceOverlap){
            OKGREEN("Enabled");
        }
        else{
            WARNING("Disabled");
        }
        WHITE( "Kernel execition timeout : " );
        if (prop.kernelExecTimeoutEnabled){
            OKGREEN("Enabled");
        }
        else{
            WARNING("Disabled");
        }

        OKBLUE("Memory Information for device");

        WHITE(
            "Total global mem (in bytes): %ld (int: %ld) (float, long int: %ld)\n",
            prop.totalGlobalMem,
            prop.totalGlobalMem / 2,
            prop.totalGlobalMem / 4
            );

        WHITE("Total constant Mem: %ld\n", prop.totalConstMem );
        WHITE("Max mem pitch: %ld\n", prop.memPitch );
        WHITE("Texture Alignment: %ld\n", prop.textureAlignment );

        OKBLUE("MP Information for device");
        if (prop.deviceOverlap){
            OKGREEN("Streams will speed computation up!");
        }
        else{
            WARNING("Streams will not speed computation up.");
        }
        WHITE("Multiprocessor count: %d\n",
              prop.multiProcessorCount );
        WHITE(
            "Shared mem per block: %ld (int: %ld) (float, long int: %ld)\n",
            prop.sharedMemPerBlock,
            prop.sharedMemPerBlock / 2,
            prop.sharedMemPerBlock / 4
            );
        WHITE("Registers per block: %d\n", prop.regsPerBlock );
        WHITE("Threads in warp: %d\n", prop.warpSize );
        WHITE("Max threads per block: %d\n",
              prop.maxThreadsPerBlock );
        WHITE("Max block dimensions: (%d, %d, %d)\n",
              prop.maxThreadsDim[0], prop.maxThreadsDim[1],
              prop.maxThreadsDim[2] );
        WHITE("Max grid dimensions: (%d, %d, %d)\n",
              prop.maxGridSize[0], prop.maxGridSize[1],
              prop.maxGridSize[2] );
        OKBLUE("===========================================");
    }

    return prop;
}
