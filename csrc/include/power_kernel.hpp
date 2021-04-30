///////////////////////////////////////////////////////////////////////////////
//             Functions for gpu processing of power measurements            //
///////////////////////////////////////////////////////////////////////////////
#include <string>

#define xstr(s) _str(s)
#define _str(s) #s

#ifndef PROCESSING_ARRAY_TYPE
#define PROCESSING_ARRAY_TYPE int
#endif

#ifndef R_POINTS
#define R_POINTS 1000
#endif

#ifndef NP_POINTS
#define NP_POINTS 1000
#endif

#ifndef THREADS_PER_BLOCK
#define THREADS_PER_BLOCK 1024
#endif

#define BLOCKS NP_POINTS

#define TOTAL_POINTS NP_POINTS*R_POINTS

#ifndef POWER_KERNEL_HPP
#define POWER_KERNEL_HPP

namespace CPU {

        /*
          short* chA_data, chB_data:              raw data from the digitiser
          unsigned int* sq_data:                  evaluated power
          int no_points = samples_per_record * number of records
          number_of_threads:                      number of threads to launch
        */
        void power_kernel(
                short* chA_data,
                short* chB_data,
                unsigned int* sq_data,
                int no_points,
                int number_of_threads
                );

        /*
          short* chA_data, chB_data:              raw data from the digitiser
          unsigned int* sq_data:                  evaluated power
          short chA_back, chB_back:               background average on both channels
          int no_points = samples_per_record * number of records
          int number_of_threads:                  number of threads to launch
        */
        void power_kernel(
                short *chA_data,
                short *chB_data,
                unsigned int *sq_data,
                short chA_back,
                short chB_back,
                int no_points,
                int number_of_threads
                );

        /*
          short* chA_data, chB_data:              raw data from the digitiser
          unsigned int* sq_data:                  evaluated power
          short* chA_back, chB_back:              background set of measurements for both channels
          int no_points = samples_per_record * number of records
          int number_of_threads:                  number of threads to launch
        */
        void power_kernel(
                short *chA_data,
                short *chB_data,
                unsigned int *sq_data,
                short *chA_back,
                short *chB_back,
                int no_points,
                int number_of_threads
                );
}

namespace GPU {
    /*
     * The following will need to be defined:
     * PROCESSING_ARRAY_TYPE
     * R_POINTS
     * NP_POINTS
     * THREADS_PER_BLOCK
     */

    // Communication of kernel parameters to python
    struct PowerKernelParameters {
        int r_points;
        int np_points;
        int blocks;
        int threads_per_block;
        std::string processing_array_type;

        PowerKernelParameters(
            int r_points,
            int np_points,
            std::string processing_array_type,
            int blocks,
            int threads_per_block);
        void print();
    };
    PowerKernelParameters fetch_kernel_parameters();

    // Preparation and completion of memory allocation on GPU
    void allocate_memory_on_gpu(short **dev_chA_data, short **dev_chB_data, float **dev_sq_data);
    void free_memory_on_gpu(short **dev_chA_data, short **dev_chB_data, float **dev_sq_data);
    void copy_background_arrays_to_gpu(short *chA_background, short *chB_background);

    void power_kernel_v1_no_background(
        short *chA_data,
        short *chB_data,
        float *sq_data,
        // for GPU memory pass in the address (&POINTER) of the memory locators
        short **dev_chA_data,
        short **dev_chB_data,
        float **dev_sq_data);
    /*
      short* chA_data, chB_data:              raw data from the digitiser
      unsigned int* sq_data:                  evaluated power
      short chA_back, chB_back:               background average on both channels
      int no_points = samples_per_record * number of records
      int number_of_threads:                  number of threads to launch
    */
    void power_kernel_v2_const_background(
        short *chA_data,
        short *chB_data,
        float *sq_data,
        short chA_back,
        short chB_back,
        short **dev_chA_data,
        short **dev_chB_data,
        float **dev_sq_data
        );

    /*
      short* chA_data, chB_data:              raw data from the digitiser
      unsigned int* sq_data:                  evaluated power
      short chA_back, chB_back:               background average on both channels
      int no_points = samples_per_record * number of records
      int number_of_threads:                  number of threads to launch
    */
    void power_kernel_v3_background(
        short *chA_data,
        short *chB_data,
        float *sq_data,
        short **dev_chA_data,
        short **dev_chB_data,
        float **dev_sq_data
        );
}

#endif
