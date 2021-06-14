#include "g1_kernel.hpp"

int G1::GPU::DIRECT::fetch_g1_kernel_blocks(){
    return 1;
}

int G1::GPU::DIRECT::fetch_g1_kernel_threads(){
    return 1;
}

int G1::GPU::DIRECT::check_g1_kernel_parameters(bool display){
    return 1;
}

void G1::CPU::preprocessor(short *chA_data, short *chB_data,
                           int N,
                           double *mean_list,
                           double *variance_list,
                           double **normalised_data)
{
    double chA_mean(0), chB_mean(0), sq_mean(0);
    for (int i = 0; i < N; i++) {
        chA_mean += chA_data[i];
        chB_mean += chB_data[i];
        normalised_data[SQG1][i] = (int)chA_data[i] * chA_data[i] + (int)chB_data[i] * chB_data[i];
        sq_mean += normalised_data[SQG1][i];
    }
    chA_mean /= N;
    chB_mean /= N;
    sq_mean /= N;

    // Evaluation of variance and normalisation
    double chA_sqDiff(0), chB_sqDiff(0), sq_sqDiff(0);
    for (int i = 0; i < N; i++) {
        normalised_data[CHAG1][i] = chA_data[i] - chA_mean;
        normalised_data[CHBG1][i] = chB_data[i] - chB_mean;
        normalised_data[SQG1][i] = normalised_data[SQG1][i] - sq_mean;

        chA_sqDiff += normalised_data[CHAG1][i] * normalised_data[CHAG1][i];
        chB_sqDiff += normalised_data[CHBG1][i] * normalised_data[CHBG1][i];
        sq_sqDiff += normalised_data[SQG1][i] * normalised_data[SQG1][i];
    }

    mean_list[CHAG1] = chA_mean;
    mean_list[CHBG1] = chB_mean;
    mean_list[SQG1] = sq_mean;

    variance_list[CHAG1] = chA_sqDiff / N;
    variance_list[CHBG1] = chB_sqDiff / N;
    variance_list[SQG1] = sq_sqDiff / N;
}
