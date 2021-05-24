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

#ifndef R_POINTS_PER_CHUNK
#error "Need to specify R_POINTS_PER_CHUNK (how to chunk the repetitions to stay within memory limits of GPU) for power measurements"
#endif

// Derived parameters
#define BLOCKS SP_POINTS
#define THREADS_PER_BLOCK (R_POINTS_PER_CHUNK > 1024) ? 1024 : R_POINTS_PER_CHUNK

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
    // Communication of kernel parameters to python
    struct PowerKernelParameters {
        int r_points; int np_points; int blocks; int threads_per_block;

        PowerKernelParameters(int r_points,
                              int np_points,
                              int blocks,
                              int threads_per_block);
        void print();
    };
    PowerKernelParameters fetch_kernel_parameters();

    const int outputs_from_gpu[4] = {CHA, CHB, CHASQ, CHBSQ};
    const int no_outputs_from_gpu = 4;

    /* Copy background data once into constant memory */
    void copy_background_arrays_to_gpu(short *chA_background, short *chB_background);

    /*
     * The input data is split into chunks, to avoid the limitation on shared memory on GPU
     * Streams are used to process these chunks in parallel
     *
     * Memory is allocated for for each stream separately:
     * - input data on the GPU
     * - output data on the GPU
     * - output data on the CPU, which needs to be memory locked for safe copying from GPU->CPU using different streams
     */
    void allocate_memory(
        short **chA_data, short **chB_data,
        short ****gpu_in, long ****gpu_out, long ****cpu_out, int no_of_streams);
    void free_memory(
        short **chA_data, short **chB_data,
        short ****gpu_in, long ****gpu_out, long ****cpu_out, int no_of_streams);
    /*
     * short* chA_data, chB_data:              raw data from the digitiser
     * double** data_out:                      kernel output [CHA, CHB, CHASQ, CHBSQ]
     * <T>** gpu_:                             pointers to memory allocated on GPU
     */
    void power_kernel(
        short *chA_data, short *chB_data,
        double **data_out,
        // Auxillary memory allocation
        short ****gpu_in, long ****gpu_out, long ****cpu_out,
        int no_streams);
}

#endif
