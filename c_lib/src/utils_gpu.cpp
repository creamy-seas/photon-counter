#include <stdio.h>
#include <cuda_runtime.h> //for cudaDeviceProp
#include "colours.hpp"
#include "utils_gpu.hpp"

/*
 * Fetch and display parameters of the GPU (already done in python)
 */
cudaDeviceProp fetch_gpu_parameters(){

        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);

        OKBLUE("===========================================");
        OKBLUE("Device parameters");
        printf("Name: %s\n", prop.name );
        printf("Compute capability: %d.%d\n", prop.major, prop.minor );
        printf("Clock rate: %d\n", prop.clockRate );
        printf("Device copy overlap: " );
        if (prop.deviceOverlap){
                OKGREEN("Enabled");
        }
        else{
                WARNING("Disabled");
        }
        printf( "Kernel execition timeout : " );
        if (prop.kernelExecTimeoutEnabled){
                OKGREEN("Enabled");
        }
        else{
                WARNING("Disabled");
        }

        OKBLUE("Memory Information for device");
        printf(
                "Total global mem (in bytes): %ld (int16: %ld) (float32: %ld)\n",
                prop.totalGlobalMem,
                prop.totalGlobalMem / 16,
                prop.totalGlobalMem / 32
                );

        printf( "Total constant Mem: %ld\n", prop.totalConstMem );
        printf( "Max mem pitch: %ld\n", prop.memPitch );
        printf( "Texture Alignment: %ld\n", prop.textureAlignment );

        OKBLUE("MP Information for device");
        printf( "Multiprocessor count: %d\n",
                prop.multiProcessorCount );
        printf(
                "Shared mem per block: %ld (int16: %ld) (float32: %ld)\n",
                prop.sharedMemPerBlock,
                prop.sharedMemPerBlock / 16,
                prop.sharedMemPerBlock / 32
                );
        printf( "Registers per block: %d\n", prop.regsPerBlock );
        printf( "Threads in warp: %d\n", prop.warpSize );
        printf( "Max threads per block: %d\n",
                prop.maxThreadsPerBlock );
        printf( "Max block dimensions: (%d, %d, %d)\n",
                prop.maxThreadsDim[0], prop.maxThreadsDim[1],
                prop.maxThreadsDim[2] );
        printf( "Max grid dimensions: (%d, %d, %d)\n",
                prop.maxGridSize[0], prop.maxGridSize[1],
                prop.maxGridSize[2] );
        OKBLUE("===========================================");

        return prop;
}
