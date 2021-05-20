/*
 * In general we expect the digitiser to return SP_POINTS (samples per record) repeated R_POINTS (number of records).
 * Therefore the chA and chB sizes are SP_POINTS * R_POINTS
 *
 * The kernels will evaluate the average at each SP_POINT (averaged over R_POINTS) for:
 * - chA
 * - chB
 * - chAsq
 * - chBsq
 * - sq
 *
 * Note - GPU and CPU kernels will differ, as for the GPU kernel array sizes need to be known at compile time
 */
#ifndef POWER_KERNEL_HPP
#define POWER_KERNEL_HPP

#ifndef R_POINTS
#error "Need to specify R_POINTS (repetitions on the digitiser == number of records) for power measurements"
#endif

#ifndef SP_POINTS
#error "Need to specify SP_POINTS (sampler per record) for power measurements"
#endif

#ifndef THREADS_PER_BLOCK
#error "Need to specify THREADS_PER_BLOCK for power measurements"
#endif

#ifndef R_POINTS_CHUNK
#error "Need to specify R_POINTS_CHUNK (how to chunk the repetitions to stay within memory limits of GPU) for power measurements"
#endif

// Derived parameters
#define BLOCKS SP_POINTS

// Verbose Indexes used for accessing array elements in a human-readable way e.g. array[CHASQ]
#define NO_OF_POWER_KERNEL_OUTPUTS 5
#define CHA 0
#define CHB 1
#define CHASQ 2
#define CHBSQ 3
#define SQ 4

// Mask is used to select which output data to evaluate
#define CHA_MASK (1 << CHA)
#define CHB_MASK (1 << CHB)
#define CHASQ_MASK (1 << CHASQ)
#define CHBSQ_MASK (1 << CHBSQ)
#define SQ_MASK (1 << SQ)

namespace CPU {

    /*
      short* chA_data, chB_data:              raw data from the digitiser
      double** data_out:                holds arrays of the averaged chA², chB², SQ=chA² + chB² data
      unsigned int processing mask:                    SQ,chA²,chB² e.g. 100==4 will only process SQ
      int no_points = samples_per_record * number of records
      number_of_threads:                      number of threads to launch
    */
    void power_kernel_v1_no_background(
        short* chA_data,
        short* chB_data,
        double** data_out,
        unsigned int processing_mask,
        int sp_points,
        int r_points,
        int number_of_threads
        );

    /*
      short* chA_data, chB_data:              raw data from the digitiser
      double** data_out:                holds arrays of the averaged chA², chB², SQ=chA² + chB² data
      unsigned int processing mask:                    SQ,chA²,chB² e.g. 100==4 will only process SQ
      short chA_back, chB_back:               background average on both channels
      int no_points = samples_per_record * number of records
      int number_of_threads:                  number of threads to launch
    */
    void power_kernel_v2_const_background(
        short *chA_data,
        short *chB_data,
        double** data_out,
        unsigned int processing_mask,
        short chA_back,
        short chB_back,
        int sp_points,
        int r_points,
        int number_of_threads
        );

    /*
      short* chA_data, chB_data:              raw data from the digitiser
      double** data_out:                holds arrays of the averaged chA², chB², SQ=chA² + chB² data
      unsigned int processing mask:           chA²,chB²,SQ e.g. 100==4 will only process SQ
      short* chA_back, chB_back:              background set of measurements for both channels, OF THE SAME SIZE as the channel data!
      int no_points = samples_per_record * number of records
      int number_of_threads:                  number of threads to launch
    */
    void power_kernel_v3_background(
        short *chA_data,
        short *chB_data,
        double** data_out,
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

        PowerKernelParameters(
            int r_points,
            int np_points,
            int blocks,
            int threads_per_block);
        void print();
    };
    PowerKernelParameters fetch_kernel_parameters();

    /* Copy background data once into constant memory */
    void copy_background_arrays_to_gpu(short *chA_background, short *chB_background);

    const int outputs_from_gpu[4] = {CHA, CHB, CHASQ, CHBSQ};
    const int no_outputs_from_gpu = 4;

    namespace V1 {
        // Kernel copies digitiser data once to GPU.
        // !! Will not work for large number of R_POINTS as the shared memory on GPU runs out !!

        /**
         * Memory management on GPU:
         * - gpu_in/out should be arrays of ADDRESSES to POINTERS.
         * - These POINTERS will location of the arrays on GPU.
         * - Declare pointers in the following way: `short* gpu_chA_data`
         * - Then store the addresses of these pointers in the arrays: `short*** gpu_in[2] = {&gpu_chA_data, &gpu_chB_data}`
         */
        void allocate_memory(short ***gpu_in, double ***gpu_out);
        void free_memory(short ***gpu_in, double ***gpu_out);

        /*
          short* chA_data, chB_data:              raw data from the digitiser
          double** data_out:                      output of the kernel with chA, chB, chAsq, chBsq, sq data. Use the macro indicies to unpack them.
          gpu_in/gp_out:                          memory allocated on the GPU using the `allocate_memory` function
        */
        void power_kernel(
            short *chA_data, short *chB_data,
            double **data_out,
            short ***gpu_in, double ***gpu_out);
    }

    namespace V2 {
        /*
         * The input data for V1 is split into chunks, to avoid the limitation on shared memory
         * Streams are used to allow parallel copying and processing of these chunks
         */

        /* Memory is allocated for:
         * - input data on the GPU for both of the streams, for both of the digitiser channels
         * - output data on the GPU. Preapre it with double** gpu_outX[4];
         * - output data on the HOST, which needs to be memory locked for safe copying
         */
        void allocate_memory(short ***gpu_in0, short ***gpu_in1,
                             double ***gpu_out0, double ***gpu_out1,
                             double ***cpu_out0, double ***cpu_out1);
        void free_memory(short ***gpu_in0, short ***gpu_in1,
                         double ***gpu_out0, double ***gpu_out1,
                         double ***cpu_out0, double ***cpu_out1);
        /*
          short* chA_data, chB_data:              raw data from the digitiser
          double** data_out:                      kernel output - use indicies defined at start
          <T>** gpu_:                             pointers to memory allocated on GPU
        */
        // void power_kernel(
        //     short *chA_data,
        //     short *chB_data,
        //     double **data_out,
        //     short **gpu_chA_data, short **gpu_chB_data,
        //     double **gpu_chA_out, double **gpu_chB_out,
        //     double **gpu_chAsq_out, double **gpu_chBsq_out
        //     );
    }
}

#endif
