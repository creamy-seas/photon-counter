#ifndef G1_KERNEL_HPP
#define G1_KERNEL_HPP

// Verbose Indexes used for accessing array elements in a human-readable way e.g. array[CHASQ]
#define CHAG1 0
#define CHBG1 1
#define SQG1 2

#ifndef G1_DIGITISER_POINTS
#define G1_DIGITISER_POINTS 289262 ///< A single readout is requested from digitiser.
#endif

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

    namespace CPU {

        /**
         * For evaluating correlation the data needs to undergo a normalisation by the average value.
         * Access the data using indicies CHAG1, CHBG1, SQG1
         */
        void preprocessor(short *chA, short *chB, int N,
                          double *mean_list, double *variance_list,
                          double **normalised_data);

        namespace DIRECT {
/**
 * @param chA_back, chB_back background set of measurements for both channels
 * @param normalise_with_less_bias to normalise each \f$g^{(1)}(\tau)\f$ by \f$N-\tau\f$ (since there are less points for evaluating larger lags) or to use the standard \f$N\f$ normalisation.
 */
            void g1_kernel(
                short *chA_data,
                short *chB_data,
                double** data_out,
                int tau_points,
                bool normalise_with_less_bias,
                int no_threads
                );
        }
    }
}

#endif
