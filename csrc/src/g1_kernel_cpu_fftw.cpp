#include <fftw3.h> // for all fttw related items
#include <string> // for to_string

#include "logging.hpp"
#include "g1_kernel.hpp"

/**
 * The intermediate step of the Wiener-Khinchin involves finding the power spectrum (magnitude squared): https://mathworld.wolfram.com/Wiener-KhinchinTheorem.html
 */
void fftw_square(fftw_complex *fourier_transform) {
    int mid_point = G1_DIGITISER_POINTS / 2;

    for (int i(0); i < mid_point; i++) {
        fourier_transform[i][REAL] = fourier_transform[i][REAL] * fourier_transform[i][REAL] + fourier_transform[i][IMAG] * fourier_transform[i][IMAG];
        fourier_transform[i][IMAG] = 0;
    }
}


void G1::CPU::FFTW::g1_kernel(short *chA_data, short *chB_data,
                              double **data_out,
                              fftw_plan plan_forward, fftw_plan plan_backward,
                              fftw_complex *aux_transform_array
    ){

    // // Normalise input arrays
    // double chA_wip[G1_DIGITISER_POINTS]; double chB_wip[G1_DIGITISER_POINTS]; double sq_wip[G1_DIGITISER_POINTS];
    // double *wip_data[G1::no_outputs] = {chA_wip, chB_wip, sq_wip};
    // double mean_list[G1::no_outputs];
    // double variance_list[G1::no_outputs];
    // G1::CPU::preprocessor(chA_data, chB_data, G1_DIGITISER_POINTS, mean_list, variance_list, wip_data);

    // // Run the forward transform -> Square -> Backward transform
    // fftw_execute_dft_r2c(plan_forward, chA_wip, aux_transform_array);
    // fftw_square(aux_transform_array);
    // fftw_execute_dft_c2r(plan_backward, aux_transform_array, data_out[CHAG1]);

    // fftw_execute_dft_r2c(plan_forward, chB_wip, aux_transform_array);
    // fftw_square(aux_transform_array);
    // fftw_execute_dft_c2r(plan_backward, aux_transform_array, data_out[CHBG1]);

    // fftw_execute_dft_r2c(plan_forward, sq_wip, aux_transform_array);
    // fftw_square(aux_transform_array);
    // fftw_execute_dft_c2r(plan_backward, aux_transform_array, data_out[SQG1]);
}

int G1::CPU::FFTW::g1_prepare_fftw_plan(std::string plan_name, int time_limit, int no_threads){

    // Setup
    fftw_init_threads();
    fftw_plan_with_nthreads(no_threads);
    fftw_set_timelimit(time_limit);
    double* x = (double*)fftw_malloc(sizeof(double) * G1_DIGITISER_POINTS);
    fftw_complex* y = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * (int(G1_DIGITISER_POINTS / 2) + 1));

    // Optimisation
    fftw_plan plan_forward = fftw_plan_dft_r2c_1d(G1_DIGITISER_POINTS, x, y, FFTW_EXHAUSTIVE);
    const char *plan_forward_name = (plan_name + "_forward.wis").c_str();
    if (!fftw_export_wisdom_to_filename(plan_forward_name))
        FAIL("Failed to export to wisdom file " + std::string(plan_forward_name));
    OKGREEN("Created forward FFTW plan in file %s", plan_forward_name);
    fftw_forget_wisdom();

    fftw_plan plan_backward = fftw_plan_dft_c2r_1d(G1_DIGITISER_POINTS, y, x, FFTW_EXHAUSTIVE);
    const char *plan_backward_name = (plan_name + "_backward.wis").c_str();
    if (!fftw_export_wisdom_to_filename(plan_backward_name))
        FAIL("Failed to export to wisdom file " + std::string(plan_backward_name));
    OKGREEN("Created backward FFTW plan in file %s", plan_backward_name);
    fftw_forget_wisdom();

    // Cleanup
    fftw_destroy_plan(plan_forward); fftw_destroy_plan(plan_backward);
    fftw_cleanup();
    fftw_free(x); fftw_free(y);
    fftw_cleanup_threads();

    return 0;
}
