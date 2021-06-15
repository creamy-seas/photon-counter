#include <thread>
#include <math.h> // for floor

#include "g1_kernel.hpp"

/**
 * Evaluation is run from `tau_min` to `tau_max`. Array are multiplied by themselves at staggered offsets.
 */
void g1_kernel_runner(double **data_out, double **wip_data, int N,
                      double *variance_list,
                      bool normalised_with_less_bias,
                      int tau_min, int tau_max
    ){
    double chA_cumulative(0), chB_cumulative(0), sq_cumulative(0);
    double normalisation;

    for (int tau(tau_min); tau < tau_max; tau++) {
        for (int i(0); i < N - tau; i++) {
            chA_cumulative += wip_data[CHAG1][i] * wip_data[CHAG1][i + tau];
            chB_cumulative += wip_data[CHBG1][i] * wip_data[CHBG1][i + tau];
            sq_cumulative += wip_data[SQG1][i] * wip_data[SQG1][i + tau];
        }
        normalisation = normalised_with_less_bias ? (double)N - (double)tau : (double)N;

        data_out[CHAG1][tau] = chA_cumulative / variance_list[CHAG1] / normalisation;
        data_out[CHBG1][tau] = chB_cumulative / variance_list[CHBG1] / normalisation;
        data_out[SQG1][tau] = sq_cumulative / variance_list[SQG1] / normalisation;

        chA_cumulative = 0;
        chB_cumulative = 0;
        sq_cumulative = 0;
    }
}

void G1::CPU::DIRECT::g1_kernel(
    short *chA_data,
    short *chB_data,
    double** data_out,
    int tau_points,
    bool normalise_with_less_bias,
    int no_threads
    ){

    double chA_wip[G1_DIGITISER_POINTS]; double chB_wip[G1_DIGITISER_POINTS]; double sq_wip[G1_DIGITISER_POINTS];
    double *wip_data[G1::no_outputs] = {chA_wip, chB_wip, sq_wip};
    double mean_list[G1::no_outputs];
    double variance_list[G1::no_outputs];

    // Normalise arrays and evalaute variance
    G1::CPU::preprocessor(chA_data, chB_data, G1_DIGITISER_POINTS, mean_list, variance_list, wip_data);

    // Launch mutltiple threads for different tau section
    std::thread* t = new std::thread[no_threads];
    int increment = floor((tau_points) / no_threads);

    for (int i(0); i < no_threads - 1; i++) {
        t[i] = std::thread(g1_kernel_runner,
                           data_out, wip_data, G1_DIGITISER_POINTS,
                           variance_list,
                           normalise_with_less_bias,
                           i * increment, (i + 1) * increment);
    }
    t[no_threads - 1] = std::thread(g1_kernel_runner,
                                    data_out, wip_data, G1_DIGITISER_POINTS,
                                           variance_list,
                                           normalise_with_less_bias,
                                           increment * (no_threads - 1),
                                    tau_points);

    //collect the multiple threads together
    for (int i(0); i < no_threads; i++)
        t[i].join();
    delete[] t;
}
