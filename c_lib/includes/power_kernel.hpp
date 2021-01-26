#include <string>

#ifndef POWER_KERNEL_HPP
#define POWER_KERNEL_HPP

namespace CPU {

        /*
          short* chA_data, chB_data:              raw data from the digitiser
          unsigned int* sq_data:                  evaluate power
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
          unsigned int* sq_data:                  evaluate power
          short chA_back, chB_back:               background set of measurements
          int no_points = samples_per_record * number of records
          number_of_threads:                      number of threads to launch
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
          unsigned int* sq_data:                  evaluate power
          short* chA_back, chB_back:              background set of measurements
          int no_points = samples_per_record * number of records
          number_of_threads:                      number of threads to launch
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
        void allocate_memory_on_gpu(short *dev_chA_data, short *dev_chB_data, float *dev_sq_data);
        void free_memory_on_gpu(short *dev_chA_data, short *dev_chB_data, float *dev_sq_data);

        float power_kernel(
                short a,
                short b);

        void power_kernel(
                short *chA_data,
                short *chB_data,
                float *sq_data,
                short *dev_chA_data,
                short *dev_chB_data,
                float *dev_sq_data);

        /*
          short* chA_data, chB_data:              raw data from the digitiser
          unsigned int* sq_data:                  evaluate power
          short chA_back, chB_back:               background set of measurements
          int no_points = samples_per_record * number of records
          number_of_threads:                      number of threads to launch
        */
        // void power_kernel(
        //         short *chA_data,
        //         short *chB_data,
        //         unsigned int *sq_data,
        //         short chA_back,
        //         short chB_back,
        //         int no_points,
        //         int number_of_threads
        //         );

        /*
          short* chA_data, chB_data:              raw data from the digitiser
          unsigned int* sq_data:                  evaluate power
          short* chA_back, chB_back:              background set of measurements
          int no_points = samples_per_record * number of records
          number_of_threads:                      number of threads to launch
        */
        // void power_kernel(
        //         short *chA_data,
        //         short *chB_data,
        //         unsigned int *sq_data,
        //         short *chA_back,
        //         short *chB_back,
        //         int no_points,
        //         int number_of_threads
        //         );
}

#endif
