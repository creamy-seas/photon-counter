#include <fftw3.h> // for all fttw related items
#include <string> // for to_string
#include <thread> // for std::thread

#include <iostream> // TODO: remove

#include "logging.hpp"
#include "g1_kernel.hpp"
#include "utils.hpp"

/**
 * The intermediate step of the Wiener-Khinchin involves finding the power spectrum (magnitude squared): https://mathworld.wolfram.com/Wiener-KhinchinTheorem.html
 */
void fftw_square(fftw_complex *fourier_transform) {
    int mid_point = G1_DIGITISER_POINTS / 2;

    for (int i(0); i < mid_point + 1; i++) {
        fourier_transform[i][REAL] = fourier_transform[i][REAL] * fourier_transform[i][REAL] + fourier_transform[i][IMAG] * fourier_transform[i][IMAG];
        fourier_transform[i][IMAG] = 0;
    }
}

const char* derive_plan_forward_name(std::string plan_name) {
    return (plan_name + "_forward.wis").c_str();
}

const char* derive_plan_backward_name(std::string plan_name) {
    return (plan_name + "_backward.wis").c_str();
}


int G1::CPU::FFTW::g1_prepare_fftw_plan(std::string plan_name, int time_limit, int no_threads){

    // Setup
    if (!fftw_init_threads()) FAIL("Failed to init threads!");
    fftw_plan_with_nthreads(no_threads);
    fftw_set_timelimit(time_limit);
    double* x = (double*)fftw_malloc(sizeof(double) * G1_DIGITISER_POINTS);
    fftw_complex* y = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * (int(G1_DIGITISER_POINTS / 2) + 1));

    // Optimisation
    fftw_plan plan_forward = fftw_plan_dft_r2c_1d(G1_DIGITISER_POINTS, x, y, FFTW_EXHAUSTIVE);
    const char *plan_forward_name = derive_plan_forward_name(plan_name);
    if (!fftw_export_wisdom_to_filename(plan_forward_name))
        FAIL("Failed to export to wisdom file " + std::string(plan_forward_name));
    OKGREEN("Created forward FFTW plan in file '%s'", plan_forward_name);
    fftw_forget_wisdom();

    fftw_plan plan_backward = fftw_plan_dft_c2r_1d(G1_DIGITISER_POINTS, y, x, FFTW_EXHAUSTIVE);
    const char *plan_backward_name = derive_plan_backward_name(plan_name);
    if (!fftw_export_wisdom_to_filename(plan_backward_name))
        FAIL("Failed to export to wisdom file " + std::string(plan_backward_name));
    OKGREEN("Created backward FFTW plan in file '%s'", plan_backward_name);
    fftw_forget_wisdom();

    // Cleanup
    fftw_destroy_plan(plan_forward); fftw_destroy_plan(plan_backward);
    fftw_cleanup();
    fftw_free(x); fftw_free(y);
    fftw_cleanup_threads();

    return 0;
}

void G1::CPU::FFTW::g1_allocate_memory(double **&data_out, fftw_complex **&aux_arrays,
                                       std::string plan_name,
                                       fftw_plan *&plans_forward, fftw_plan *&plans_backward) {
    /** There is a lot of derefenecing in this function, since the arrays ara passed in by address & */

    // Allocate arrays for use with FFTW
    data_out = new double*[G1::no_outputs];
    aux_arrays = new fftw_complex*[G1::no_outputs];
    for (int i(0); i < G1::no_outputs; i++){
        data_out[i] = (double*)fftw_malloc(sizeof(double) * G1_DIGITISER_POINTS);
        aux_arrays[i] = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * (int(G1_DIGITISER_POINTS / 2) + 1));
    }

    // Create forward and backwards plans for each index. Since plans are loaded, set a time limit of 0
    if (!fftw_init_threads()) FAIL("Failed to init threads!");
    fftw_set_timelimit(0);

    OKBLUE("Generating optimised forward plans");
    plans_forward = new fftw_plan[G1::no_outputs];
    if (!fftw_import_wisdom_from_filename(derive_plan_forward_name(plan_name))) FAIL("Failed to load wisdom file " + std::string(derive_plan_forward_name(plan_name)));
    for (int i(0); i < G1::no_outputs; i++)
        plans_forward[i] = fftw_plan_dft_r2c_1d(
            G1_DIGITISER_POINTS, data_out[i], aux_arrays[i], FFTW_EXHAUSTIVE);
    fftw_forget_wisdom();

    OKBLUE("Generating optimised backward plans");
    plans_backward = new fftw_plan[G1::no_outputs];
    if (!fftw_import_wisdom_from_filename(derive_plan_backward_name(plan_name))) FAIL("Failed to load wisdom file " + std::string(derive_plan_backward_name(plan_name)));
    for (int i(0); i < G1::no_outputs; i++)
        plans_backward[i] = fftw_plan_dft_c2r_1d(
            G1_DIGITISER_POINTS, aux_arrays[i], data_out[i], FFTW_EXHAUSTIVE);
    fftw_forget_wisdom();
}

void G1::CPU::FFTW::g1_free_memory(double **data_out, fftw_complex **aux_arrays,
                                   fftw_plan *plans_forward, fftw_plan *plans_backward) {
    for (int i(0); i < G1::no_outputs; i++) {
        fftw_destroy_plan(plans_forward[i]);
        fftw_destroy_plan(plans_backward[i]);
        fftw_free(data_out[i]);
        fftw_free(aux_arrays[i]);
    }

    delete[] aux_arrays;
    delete[] data_out;
    fftw_cleanup_threads();
    fftw_cleanup();
}

void g1_kernel_runner(double *data_out,
                      fftw_complex *aux_array,
                      fftw_plan plan_forward, fftw_plan plan_backward,
                      double variance){
    // Run the forward transform -> Square -> Backward transform
    fftw_execute_dft_r2c(plan_forward, data_out, aux_array);
    fftw_square(aux_array);
    fftw_execute_dft_c2r(plan_backward, aux_array, data_out);

    double normalisation = (double)G1_DIGITISER_POINTS * G1_DIGITISER_POINTS * variance;
    for (int i(0); i < G1_DIGITISER_POINTS; i++)
        data_out[i] /= normalisation;
}

void G1::CPU::FFTW::g1_kernel(short *chA_data, short *chB_data,
                              double **data_out, fftw_complex **aux_arrays,
                              fftw_plan *plans_forward, fftw_plan *plans_backward) {
    // Normalise input arrays
    // double mean_list[G1::no_outputs];
    double variance_list[G1::no_outputs];
    // G1::CPU::preprocessor(chA_data, chB_data, G1_DIGITISER_POINTS, mean_list, variance_list, data_out);

    // Each thread will perform it's own transform
    std::thread thread_list[G1::no_outputs];
    for (int i(0); i < G1::no_outputs; i++) {
        thread_list[i] = std::thread(g1_kernel_runner,
                                     data_out[i],
                                     aux_arrays[i],
                                     plans_forward[i],
                                     plans_backward[i],
                                     variance_list[i]);
    }
    for (int i(0); i < G1::no_outputs; i++)
        thread_list[i].join();
}
