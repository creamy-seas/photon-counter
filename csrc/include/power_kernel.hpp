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
#define BLOCKS SP_POINTS ///< Each `SP_POINT` will be processed by a separate GPU block.
#define THREADS_PER_BLOCK (R_POINTS_PER_CHUNK > 1024) ? 1024 : R_POINTS_PER_CHUNK ///< Allocationg up to 1024 GPU threads for dealing with repetitions. If there are more repeitions than this, threads will be reused.

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

    /**
     * @param processing_mask selection of computations to run. Should be a bitwise or of CHA_MASK, CHB_MASK, CHASQ_MASK, CHBSQ_MASK, SQ_MASK to determine what averages to run.
     * @param chA_back, chB_back background set of measurements for both channels
     */
    void power_kernel(
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
    /**
     * Communication of kernel parameters, for inspection and validation in python.
     */
    struct PowerKernelParameters {
        int r_points; int np_points; int blocks; int threads_per_block;

        PowerKernelParameters(int r_points, int np_points, int blocks, int threads_per_block);

        void print();
    };
    /**
     * As the GPU kernel will be compiled a single time, this will return the variables
     * that should be used when calling it.
     *
     * @returns PowerKernelParameters struct, with details on the compiled kernel parmeters.
     */
    PowerKernelParameters fetch_kernel_parameters();

    const int outputs_from_gpu[4] = {CHA, CHB, CHASQ, CHBSQ}; ///< Convenience array for iteration through the different computations running on GPU.
    const int no_outputs_from_gpu = 4; ///< GPU will be computing this number of results.

    /**
     * Copying of background data once into constant memory on the GPU - this will be subtracted from all input data before processing.
     *
     * @param chA_background, chB_background arrays of length `SP_POINTS` to copy over to GPU.
     */
    void copy_background_arrays_to_gpu(short *chA_background, short *chB_background);

    /*
     * The input data is split into chunks, to avoid the limitation on shared memory on GPU.
     * Streams are used to process these chunks in parallel.
     * Memory is allocated for for each stream separately to avoid race conditions
     *
     *
     * - input data on the CPU which needs to be memeory locked for safe copying CPU -> GPU using different streams
     * - input data on the GPU
     * - output data on the GPU
     * - output data on the CPU, which needs to be memory locked for safe copying from GPU->CPU using different streams
     *
     * Pass in the ADDRESSES of the pointers that will store these arrays i.e. `short ***gpu_in` will be paseed in as &gpu_in
     */
    void allocate_memory(
        short **chA_data, short **chB_data,
        short ****gpu_in, long ****gpu_out, long ****cpu_out, int no_of_streams);
    void free_memory(
        short **chA_data, short **chB_data,
        short ***gpu_in, long ***gpu_out, long ***cpu_out, int no_of_streams);

    /**
     * We expect the digitiser to return `SP_POINTS` (samples per record) repeated `R_POINTS` (number of records).
     * Therefore the chA and chB sizes are `SP_POINTS * R_POINTS`, which are processed to give:
     * - \f[ \left\langle{chA}\right\rangle \f]
     * - \f[ \left\langle{chB}\right\rangle \f]
     * - \f[ \left\langle{chA^2}\right\rangle \f]
     * - \f[ \left\langle{chB^2}\right\rangle \f]
     * - \f[ \left\langle{chA^2 + chB^2}\right\rangle \f]
     * each of `SP_POINTS` in length as the repetitions are averaged.
     *
     * **GPU kernels will need array sizes need to be known at compile time** so used GPU::fetch_kernel_parameters to
     * ensure that correct data is passed into it.
     *
     * @param chA_data, chB_data raw data from the digitiser
     * @param data_out kernel output in the following order: `[CHA, CHB, CHASQ, CHBSQ]`
     * @param gpui, gpu_out, cpu_out auxillary arrays allocated using `allocate_memory` function
     * @param no_streams to launch on GPU. Benchmarking indicates that 2 is the optimal choice.
     */
    void power_kernel(
        short *chA_data, short *chB_data,
        double **data_out,
        // Auxillary memory allocation
        short ***gpu_in, long ***gpu_out, long ***cpu_out, int no_streams);
}

#endif
