#ifndef G1_KERNEL_HPP
#define G1_KERNEL_HPP

#include <fftw3.h>
#include <string>
#include <cufft.h>

#ifndef G1_DIGITISER_POINTS
#define G1_DIGITISER_POINTS 262144 ///< A single readout is requested from digitiser. Powers of 2 are executed especially fast
#endif

// Verbose Indexes used for accessing array elements in a human-readable way e.g. array[CHASQ]
#define CHAG1 0
#define CHBG1 1
#define SQG1 2

#define REAL 0
#define IMAG 1

#ifdef __cplusplus
extern "C" {
#endif
    /**
     * \f[ g^{(1)}(\tau) = \left\langle{X}(t)X(t+\tau)\right\rangle correlation measurements \f]
     */
    namespace G1 {
        namespace GPU {

            /**
             * Validation of kernel parameters before it's invocation.
             */
            int check_g1_kernel_parameters(bool display=false);
        }
    }

#ifdef __cplusplus
}
#endif

void runTest();

/**
 * @brief \f$ g^{(1)}(\tau) = \left\langle{X}(t)X(t+\tau)\right\rangle \f$ correlation measurements.
 */
namespace G1 {
    const int no_outputs = 3; ///< Kernels returns CHAG1 and CHBG1 and SQG1
    const int outputs[3] = {CHAG1, CHBG1, SQG1}; ///< Convenience array for iteration in loops/


    // TODO: GPU not implemented, as CPU performance is good enough
    /**
     * **Recomendations for FFTW**
     * - Use single precision transforms.
     * - Restrict the size along all dimensions to be representable as 2a×3b×5c×7d.
     * - Restrict the size along each dimension to use fewer distinct prime factors.
     * - Restrict the data to be contiguous in memory when performing a single transform. When performing multiple transforms make the individual datasets contiguous
     * Use out-of-place mode (copy data to new arrays)
     */
    namespace GPU {

        /**
         * Prepare optimised plans for the FFTW transform on the GP
         * Not all operations are supported on the GPU: https://docs.nvidia.com/cuda/cufft/index.html#fftw-supported-interface (such as dumping of wisdom files)
         * @param plan_name base name under which to save the plans
         * @param time_limit in seconds to run optimisation for
         * @param no_threads to use for execution. For an 8 core system, 8 is best
         *
         * @returns 0 for success
         */
        int g1_prepare_fftw_plan(cufftHandle *&plans_forward, cufftHandle *&plans_backward);

        void g1_kernel(
            short *chA_data, short *chB_data,
            double **data_out,
            cufftHandle *plans_forward, cufftHandle *plans_backward);

        /**
         * For evaluating correlation the data needs to undergo a normalisation by the average value.
         * This preprocessor:
         * - Apply normalisation
         * - Evaluate mean
         * - Evaluate variance
         *
         * Access the data using indicies CHAG1, CHBG1, SQG1.
         */
        void preprocessor(short *chA, short *chB, int N,
                          double *mean_list, double *variance_list,
                          double **normalised_data);
    }

    namespace CPU {

        /**
         * For evaluating correlation the data needs to undergo a normalisation by the average value.
         * This preprocessor:
         * - Apply normalisation
         * - Evaluate mean
         * - Evaluate variance
         *
         * Access the data using indicies CHAG1, CHBG1, SQG1.
         */
        void preprocessor(short *chA, short *chB, int N,
                          double *mean_list, double *variance_list,
                          double **normalised_data);

        /**
         * @brief Evaluation by blunt iteration.
         */
        namespace DIRECT {

            /**
             * Evaluation is performed using brute force cross-multiplication of arrays
             * @param data_out arrays for storing `CHAG1`, `CHBG1` and `SQG1` results.
             * @param tau_points autocorrelation \f$g^{(1)}(\tau)\f$ is evaluated from \f$\tau=0\f$ to \$\tau=\$`tau_points`
             * @param normalise_with_less_bias to normalise each \f$g^{(1)}(\tau)\f$ by \f$N-\tau\f$ (since there are less points for evaluating larger lags) or to use the standard \f$N\f$ normalisation
             * @param no_threads to use for parallel evaluation.
            */
            void g1_kernel(
                short *chA_data,
                short *chB_data,
                double** data_out,
                int tau_points,
                bool normalise_with_less_bias,
                int no_threads=1
                );
        }

        /**
         * @brief Faster evaluation using Wiener-Khinchin Theorem.
         */
        namespace FFTW {

            /**
             * Creates two **wisdom** files with parameters for optimised forward and backward FFTW transforms. Plans will work less well once reloaded (as they are optimised arrays in specific locations).
             * @param plan_name base name under which to save the plans
             * @param time_limit in seconds to run optimisation for
             * @param no_threads to use for execution. For an 8 core system, 8 is best
             *
             * @returns 0 for success
             */
            int g1_prepare_fftw_plan(std::string plan_name, int time_limit, int no_threads=8);

            /**
             * Arrays need to be allocated using `fftw_malloc` for processing by the FFTW library.
             * Each array is supplemented with an optimised plan.
             *
             * @param data_out array that will store the eventual CHAG1, CHBG1 and SQG1 values
             * @param aux_arrays used for intermediate computations. FFTW Plans are created between data_out[idx] <-> aux_arrays[idx].
             * @param plan_name base name of the wisdom files made using g1_prepare_fftw_plan. These optimised files are used as a baseline for generating plans specific to the allocated arrays
             * @param plans_forward, plans_backward arrays to store the optimised plans.
             *
             *     fftw_plan *plans_forward, *plans_backward;
             *     g1_allocate_memory(data_out, aux_arrays, plan_name, plans_forward, plans_backward);
             */
            void g1_allocate_memory(double **&data_out, fftw_complex **&aux_arrays,
                                    std::string plan_name,
                                    fftw_plan *&plans_forward, fftw_plan *&plans_backward);
            void g1_free_memory(double **data_out, fftw_complex **aux_arrays,
                                fftw_plan *plans_forward, fftw_plan *plans_backward);

            /**
             * Evaluation is performed using the fftw3 library and the Wiener–Khinchin theorem.
             * @param chA_data, chB_data data from digitiser
             *
             * The rest of the parameters must be generated using the g1_allocate_memory function:
             * @param data_out arrays for storing `CHAG1`, `CHBG1` and `SQG1` results.
             * @param aux_arrays used for intermediate computations. FFTW Plans are created between data_out[idx] <-> aux_arrays[idx].
             * @param plans_forward, plans_backward an optimised plans for thw FFTW.
             */
            void g1_kernel(
                short *chA_data,
                short *chB_data,
                double **data_out, fftw_complex **aux_arrays,
                fftw_plan *plans_forward, fftw_plan *plans_backward
                );
        }
    }
}

#endif
