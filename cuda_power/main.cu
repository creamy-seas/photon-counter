#include <stdio.h>
#include "colours.h"

// Return type is void
__global__ void magnitude_squared(int a, int b, float *c){
        /*
         * Used for computing magnitude squared:
         * A^2 + B^2

         Since data will be fed in as a block of R * (N * P)
         R: Number of repititions
         N: Number of pulses
         P: Period of a pulse

         We will transform it
         */

        *c = (float)(a * a + b * b) / 10;
}

int verify_gpu_allocation(
        cudaDeviceProp gpu_properties,
        cudaDeviceProp desired_properties){

        return 1;
}

// Fetch and display parameters of the GPU (redundant - already done in python)
cudaDeviceProp fetch_gpu_parameters(){

        // Read in device properties into address
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);

        printf("%s\n", OKBLUE("==========================================="));
        printf("%s\n", OKBLUE("Device parameters"));
        printf("Name: %s\n", prop.name );
        printf("Compute capability: %d.%d\n", prop.major, prop.minor );
        printf("Clock rate: %d\n", prop.clockRate );
        printf("Device copy overlap: " );
        if (prop.deviceOverlap)
                printf("%s\n", OKGREEN("Enabled") );
        else
                printf("%s\n", FAIL("Disabled"));
        printf( "Kernel execition timeout : " );
        if (prop.kernelExecTimeoutEnabled)
                printf( "%s\n", OKGREEN("Enabled") );
        else
                printf( "%s\n", FAIL("Disabled"));


        printf( "%s\n", OKBLUE("Memory Information for device"));
        printf(
                "Total global mem: %ld (int16: %ld) (float32: %ld)\n",
                prop.totalGlobalMem,
                prop.totalGlobalMem / 16,
                prop.totalGlobalMem / 32
                );

        printf( "Total constant Mem: %ld\n", prop.totalConstMem );
        printf( "Max mem pitch: %ld\n", prop.memPitch );
        printf( "Texture Alignment: %ld\n", prop.textureAlignment );

        printf( "%s\n", OKBLUE("MP Information for device"));
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
        printf("%s\n", OKBLUE("==========================================="));

        return prop;
}

int main(void){

        cudaDeviceProp prop = fetch_gpu_parameters();

        cudaDeviceProp program_prop;
        memset(&program_prop, 0, sizeof(cudaDeviceProp));

        program_prop.maxGridSize[0] = 100;

        // float c;
        // float *dev_c;

        // cudaMalloc((void**) &dev_c, sizeof(float));

        // magnitude_squared<<<1,1>>>(1,2, dev_c);

        // cudaMemcpy(
        //         &c,
        //         dev_c,
        //         sizeof(float),
        //         cudaMemcpyDeviceToHost
        //         );
        // std::cout << c << std::endl;

        // cudaFree(dev_c);
        return 0;
}
