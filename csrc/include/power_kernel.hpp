#ifndef POWER_KERNEL_HPP
#define POWER_KERNEL_HPP

#ifndef R_POINTS
#error "Need to specify R_POINTS (repetitions on the digitiser == number of records) for power measurements"
#endif

#ifndef SP_POINTS
#error "Need to specify SP_POINTS (sampler per record) for power measurements"
#endif

#ifndef R_POINTS_PER_GPU_CHUNK
#define R_POINTS_PER_GPU_CHUNK 1000 ///< The GPU cannot allocate enough shared memory to process all the data. Therefore the total number of repetitions is broken up into chunks.
#endif

// Derived parameters
#define BLOCKS SP_POINTS ///< Each `SP_POINT` will be processed by a separate GPU block.
#define THREADS_PER_BLOCK (R_POINTS_PER_GPU_CHUNK > 1024) ? 1024 : R_POINTS_PER_GPU_CHUNK ///< Allocationg up to 1024 GPU threads for dealing with repetitions. If there are more repetitions than this, threads will be reused.

// Verbose Indexes used for accessing array elements in a human-readable way e.g. array[CHASQ]
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

#ifdef __cplusplus
extern "C" {
#endif
    namespace POWER {
        namespace GPU {
            int fetch_power_kernel_blocks();
            int fetch_power_kernel_threads();
            /**
             * Validation of kernel parameters before it's invocation.
             */
            int check_power_kernel_parameters(bool display=false);
        }
    }
#ifdef __cplusplus
}
#endif

/**
 * @brief \f$ \left\langle{chA}\right\rangle, \left\langle{chB}\right\rangle, \left\langle{chA^2}\right\rangle, \left\langle{chB^2}\right\rangle, \left\langle{chA^2 + chB^2}\right\rangle \f$ measurements.
 *
 * We expect the digitiser to return `SP_POINTS` (samples per record) repeated `R_POINTS` (number of records).
 * Therefore the chA and chB sizes are `SP_POINTS * R_POINTS`, which are processed to give an average value:
 * - \f[ \left\langle{chA}\right\rangle \f]
 * - \f[ \left\langle{chB}\right\rangle \f]
 * - \f[ \left\langle{chA^2}\right\rangle \f]
 * - \f[ \left\langle{chB^2}\right\rangle \f]
 * - \f[ \left\langle{chA^2 + chB^2}\right\rangle \f]
 */
namespace POWER {

    const int no_outputs = 5;

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

        const int outputs_from_gpu[4] = {CHA, CHB, CHASQ, CHBSQ}; ///< Convenience array for iteration through the different computations running on GPU.
        const int no_outputs_from_gpu = 4; ///< GPU will be computing this number of results.

        /**
         * Copying of background data once into constant memory on the GPU - this will be subtracted from all input data before processing.
         *
         * @param chA_background, chB_background arrays of length `SP_POINTS` to copy over to GPU.
         */
        void copy_background_arrays_to_gpu(short *chA_background, short *chB_background);

        /**
         * The input data is split into chunks, to avoid the limitation on shared memory on GPU.
         * - Streams are used to process these chunks in parallel.
         * - Memory is allocated for for each stream separately to avoid race conditions.
         * - Memory on the CPU needs to be pinned so that it is never paged and always accessible to streams.
         * - Memory on the GPU will be allocated and it's address (`short*`) stored in an array.
         *
         * Pass in the ADDRESSES of the pointers that will store these arrays e.g.
         *
         *     short ***gpu_in; short **gpu_out; long ***cpu_out; short *chA_data, short *chB_data;
         *     allocate_memory(&chA_data, &chB_data, &gpu_in, &gpu_out, &cpu_out, 2);
         *
         * Pass in 0 to skip allocation for certain inputs
         *
         *     short *chA_data, short *chB_data;
         *     allocate_memory(&chA_data, &chB_data, 0, 0, 0, 2);
         *
         * @param chA_data, chB_data arrays to be populated by the digitiser. Pinned
         * @param gpu_in, gpu_out arrays holding the addresses of the GPU arrays
         * @param cpu_out Pinned memory on CPU
         */
        void allocate_memory(
            short **chA_data, short **chB_data,
            short ****gpu_in, long ****gpu_out, long ****cpu_out, int no_of_streams);
        void free_memory(
            short *chA_data, short *chB_data,
            short ***gpu_in, long ***gpu_out, long ***cpu_out, int no_of_streams);

            /**
             * **Important**
             * - GPU kernels is compiled with fixed array sizes - use GPU::check_power_kernel_parameters to validate it.
             * - The actual data is processes in chunks of size `R_POINTS_PER_GPU_CHUNK` each. Depending on whether data is accumulated (`long **data_out`)
             *   or normalised (`double **data_out`) normalisation may need to take place outside the function.
             *
             * @param chA_data, chB_data raw data from the digitiser
             * @param data_out kernel output: `[CHA, CHB, CHASQ, CHBSQ]`.
             * @tparam T specifies whether data_out should be accumulated (long, for repetitive invocations) or normalised (double, for a single run)
             * @param gpui, gpu_out, cpu_out auxillary arrays allocated using `allocate_memory` function.
             * @param no_streams to launch on GPU. Benchmarking indicates that 2 is the optimal choice.
             * @param accumulate if true, data will be accumulated in `data_out` with no normalisation. Idea is to normalise it after many repetitive runs.
             *
             */
        template <typename T> void power_kernel(
            short *chA_data, short *chB_data,
            T **data_out,
            // Auxillary memory allocation
            short ***gpu_in, long ***gpu_out, long ***cpu_out, int no_streams);
    }
}

#endif
