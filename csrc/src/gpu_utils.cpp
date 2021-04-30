#include <stdio.h>
#include <sstream>
#include <string>

#include <cuda_runtime_api.h> //for cudaDeviceProp
#include "colours.hpp"
#include "gpu_utils.hpp"

/*
 * Fetch and display parameters of the GPU (already done in python)
 */
cudaDeviceProp fetch_gpu_parameters(){

        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);

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
                "Total global mem (in bytes): %ld (int16: %ld) (float32: %ld)\n",
                prop.totalGlobalMem,
                prop.totalGlobalMem / 16,
                prop.totalGlobalMem / 32
                );

        WHITE( "Total constant Mem: %ld\n", prop.totalConstMem );
        WHITE( "Max mem pitch: %ld\n", prop.memPitch );
        WHITE( "Texture Alignment: %ld\n", prop.textureAlignment );

        OKBLUE("MP Information for device");
        WHITE( "Multiprocessor count: %d\n",
                prop.multiProcessorCount );
        WHITE(
                "Shared mem per block: %ld (int16: %ld) (float32: %ld)\n",
                prop.sharedMemPerBlock,
                prop.sharedMemPerBlock / 16,
                prop.sharedMemPerBlock / 32
                );
        WHITE( "Registers per block: %d\n", prop.regsPerBlock );
        WHITE( "Threads in warp: %d\n", prop.warpSize );
        WHITE( "Max threads per block: %d\n",
                prop.maxThreadsPerBlock );
        WHITE( "Max block dimensions: (%d, %d, %d)\n",
                prop.maxThreadsDim[0], prop.maxThreadsDim[1],
                prop.maxThreadsDim[2] );
        WHITE( "Max grid dimensions: (%d, %d, %d)\n",
                prop.maxGridSize[0], prop.maxGridSize[1],
                prop.maxGridSize[2] );
        OKBLUE("===========================================");

        return prop;
}
