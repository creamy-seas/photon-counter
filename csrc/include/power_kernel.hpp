#ifndef POWER_KERNEL_HPP
#define POWER_KERNEL_HPP
// Kernels that evalaute power from readings from digitiser
// In general we exepct the digitiser to return SP_POINTS (samples per record) repeated R_POINTS (number of records).
// Therefore the chA and chB sizes are SP_POINTS * R_POINTS
//
// Note - GPU and CPU kernels will differ, as for the GPU kernel array sizes need to be known at compile time
#include <string>

#define xstr(s) _str(s)
#define _str(s) #s

#ifndef PROCESSING_ARRAY_TYPE
#define PROCESSING_ARRAY_TYPE int
#endif

#ifndef R_POINTS
#define R_POINTS 1000
#endif

#ifndef SP_POINTS
#define SP_POINTS 1000
#endif

#ifndef THREADS_PER_BLOCK
#define THREADS_PER_BLOCK 1024
#endif

#define BLOCKS SP_POINTS

#define TOTAL_POINTS SP_POINTS*R_POINTS

#define CHA_MASK 1
#define CHB_MASK 2
#define SQ_MASK 4

/*
 * Macro to compactly expand the processing cases into blocks that are run depending on what the input_mask
 * value is. E.g. if input_mask=5 (101 in binary), then chA_func and sq_func are run.
 *
 * input_mask: the parameter that will be checked when determining which block to execute.
 * x_func: lambda function, with no input or output arguments (use pointers to pass data in or out)
 *      read more about lambda functions here: https://www.cprogramming.com/c++11/c++11-lambda-closures.html
 *      e.g. auto lambda_func_example = [&capture_value] () { capture_value[0][0] = 1; };
 */
#define MASK_RESOLVER(input_mask, chA_func, chB_func, sq_func)          \
    if (input_mask == (CHA_MASK ^ CHB_MASK ^ SQ_MASK)) {                \
        chA_func();                                                     \
        chB_func();                                                     \
        sq_func();                                                      \
    }                                                                   \
    else if (input_mask == (CHA_MASK ^ CHB_MASK)){                 \
        chA_func();                                                     \
        chB_func();                                                     \
    }                                                                   \
    else if (input_mask == (CHA_MASK ^ SQ_MASK)){                  \
        chA_func();                                                     \
        sq_func();                                                      \
    }                                                                   \
    else if (input_mask == (CHB_MASK ^ SQ_MASK)){                  \
        chB_func();                                                     \
        sq_func();                                                      \
    }                                                                   \
    else if (input_mask == (CHA_MASK))                             \
        chA_func();                                                     \
    else if (input_mask == (CHB_MASK))                             \
        chB_func();                                                     \
    else if (input_mask == (SQ_MASK))                              \
        sq_func();

namespace CPU {

    /*
      short* chA_data, chB_data:              raw data from the digitiser
      double** processed_data:                holds arrays of the averaged chA², chB², SQ=chA² + chB² data
      unsigned int processing mask:                    SQ,chA²,chB² e.g. 100==4 will only process SQ
      int no_points = samples_per_record * number of records
      number_of_threads:                      number of threads to launch
    */
    void power_kernel_v1_no_background(
        short* chA_data,
        short* chB_data,
        double** processed_data,
        unsigned int processing_mask,
        int sp_points,
        int r_points,
        int number_of_threads
        );

    /*
      short* chA_data, chB_data:              raw data from the digitiser
      double** processed_data:                holds arrays of the averaged chA², chB², SQ=chA² + chB² data
      unsigned int processing mask:                    SQ,chA²,chB² e.g. 100==4 will only process SQ
      short chA_back, chB_back:               background average on both channels
      int no_points = samples_per_record * number of records
      int number_of_threads:                  number of threads to launch
    */
    void power_kernel_v2_const_background(
        short *chA_data,
        short *chB_data,
        double** processed_data,
        unsigned int processing_mask,
        short chA_back,
        short chB_back,
        int sp_points,
        int r_points,
        int number_of_threads
        );

    /*
      short* chA_data, chB_data:              raw data from the digitiser
      double** processed_data:                holds arrays of the averaged chA², chB², SQ=chA² + chB² data
      unsigned int processing mask:           chA²,chB²,SQ e.g. 100==4 will only process SQ
      short* chA_back, chB_back:              background set of measurements for both channels, OF THE SAME SIZE as the channel data!
      int no_points = samples_per_record * number of records
      int number_of_threads:                  number of threads to launch
    */
    void power_kernel_v3_background(
        short *chA_data,
        short *chB_data,
        double** processed_data,
        unsigned int processing_mask,
        short *chA_back,
        short *chB_back,
        int sp_points,
        int r_points,
        int number_of_threads
        );
    }

namespace GPU {
    /*
     * The following will need to be defined:
     * PROCESSING_ARRAY_TYPE
     * R_POINTS
     * SP_POINTS
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
    void allocate_memory_on_gpu(short **dev_chA_data, short **dev_chB_data, double **dev_sq_out);
    void free_memory_on_gpu(short **dev_chA_data, short **dev_chB_data, double **dev_sq_out);
    void copy_background_arrays_to_gpu(short *chA_background, short *chB_background);

    void power_kernel_v1_no_background(
        short *chA_data,
        short *chB_data,
        double *processed_data,
        // for GPU memory pass in the address (&POINTER) of the memory locators
        short **dev_chA_data,
        short **dev_chB_data,
        double **dev_sq_out);
    /*
      short* chA_data, chB_data:              raw data from the digitiser
      double* processed_data:                  evaluated power
      short chA_back, chB_back:               background average on both channels
      int no_points = samples_per_record * number of records
      int number_of_threads:                  number of threads to launch
    */
    void power_kernel_v2_const_background(
        short *chA_data,
        short *chB_data,
        double *processed_data,
        short chA_back,
        short chB_back,
        short **dev_chA_data,
        short **dev_chB_data,
        double **dev_sq_out
        );

    /*
      short* chA_data, chB_data:              raw data from the digitiser
      double* processed_data:                  evaluated power
      short chA_back, chB_back:               background average on both channels OF A SINGLE RUN!
      int no_points = samples_per_record * number of records
      int number_of_threads:                  number of threads to launch
    */
    void power_kernel_v3_background(
        short *chA_data,
        short *chB_data,
        double *processed_data,
        short **dev_chA_data,
        short **dev_chB_data,
        double **dev_sq_out
        );
}

#endif
