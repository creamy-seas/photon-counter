#ifndef G1_KERNEL_HPP
#define G1_KERNEL_HPP

#include <fftw3.h>
#include <string>

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
    namespace G1 {
        namespace GPU {
            namespace DIRECT {
                int fetch_g1_kernel_blocks();
                int fetch_g1_kernel_threads();

                /**
                 * Validation of kernel parameters before it's invocation.
                 */
                int check_g1_kernel_parameters(bool display=false);
            }
        }
    }

#ifdef __cplusplus
}
#endif

/**
 * \f[ g^{(1)}(\tau) = \left\langle{X}(t)X(t+\tau)\right\rangle correlation measurements \f]
 */
namespace G1 {
    const int no_outputs = 3; ///< Kernels returns CHAG1 and CHBG1 and SQG1
    const int outputs[3] = {CHAG1, CHBG1, SQG1}; ///< Convenience array for iteration in loops/

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
             * @param aux_array used for intermediate computations. FFTW Plans are created between data_out[idx] <-> aux_array (thus aux_array is reused for multiple transforms).
             * @param plan_name base name of the wisdom files made using g1_prepare_fftw_plan. These optimised files are used as a baseline for generating plans specific to the allocated arrays
             * @param plans_forward, plans_backward arrays to store the optimised plans.
             *
             * Pass in the ADDRESSES of the pointers that will store these arrays e.g.
             *
             *     fftw_plan *plans_forward, *plans_backward;
             *     g1_allocate_memory(&data_out, &aux_array, plan_name, &plans_forward, &plans_backward);
             */
            void g1_allocate_memory(double ***data_out, fftw_complex **aux_array,
                                    std::string plan_name,
                                    fftw_plan **plans_forward, fftw_plan **plans_backward);
            void g1_free_memory(double **data_out, fftw_complex *aux_array,
                                fftw_plan *plans_forward, fftw_plan *plans_backward);

            /**
             * Evaluation is performed using the fftw3 library and the Wiener–Khinchin theorem.
             * @param chA_data, chB_data data from digitiser
             *
             * The rest of the parameters must be generated using the g1_allocate_memory function:
             * @param data_out arrays for storing `CHAG1`, `CHBG1` and `SQG1` results.
             * @param aux_array used for intermediate computations. FFTW Plans are created between data_out[idx] <-> aux_array (thus aux_array is reused for multiple transforms).
             * @param plans_forward, plans_backward an optimised plans for thw FFTW.
             */
            void g1_kernel(
                short *chA_data,
                short *chB_data,
                double **data_out, fftw_complex *aux_array,
                fftw_plan *plans_forward, fftw_plan *plans_backward
                );
        }
    }
}

#endif
