#include <thread> // thread
#include <math.h> // floor

#include "power_kernel.hpp"
/* #include "ADQILYAAPI_x64.h" */
/* #include "DLL_Imperium.h" */
/* #include <fstream> */
/* #include "fftw3.h" */
/* #include <ctime> */

/**
 * Reduce the array by summing up the total into the first cell.
 *
 * __ Logic ___
 * a1 a2 a3 a4 ... b1 b2 b3 b4 ... c1 c2 c3 c4 ...
 *
 * will be mapped to a 2D array
 *
 * a1 a2 a3 -> main_axis (sp_coordinate)
 * b1 b2 b3 ...
 * c1 c2 c3 ...
 * d1 d2 d3 ...
 * e1 e2 e3 ...
 * f1 f2 f3 ...
 * g1 g2 g3 ...
 * |
 * repetition-axis (r_coordinate)
 *
 * And reduced to the following by summing up over the repetition axis and normalising by the size
 * <1> <2> <3> ...
 *
 * @param flat_cumulative_data 2D array of sequntial data to accumulate (or average)
 */
void reduction_average(unsigned int** flat_cumulative_data, double** data_out,
                       unsigned int processing_mask,
                       int sp_points, int r_points) {

    // At least it is clear what is going on
    for (int sp(0); sp < sp_points; sp++) {
        for (int r(1); r < r_points; r++) {
            if (processing_mask & CHA_MASK) flat_cumulative_data[CHA][sp] += flat_cumulative_data[CHA][sp + r * sp_points];
            if (processing_mask & CHB_MASK) flat_cumulative_data[CHB][sp] += flat_cumulative_data[CHB][sp + r * sp_points];
            if (processing_mask & CHASQ_MASK) flat_cumulative_data[CHASQ][sp] += flat_cumulative_data[CHASQ][sp + r * sp_points];
                if (processing_mask & CHBSQ_MASK) flat_cumulative_data[CHBSQ][sp] += flat_cumulative_data[CHBSQ][sp + r * sp_points];
                if (processing_mask & SQ_MASK) flat_cumulative_data[SQ][sp] += flat_cumulative_data[SQ][sp + r * sp_points];
}
        if (processing_mask & CHA_MASK) data_out[CHA][sp] = (double)flat_cumulative_data[CHA][sp] / r_points;
        if (processing_mask & CHB_MASK) data_out[CHB][sp] = (double)flat_cumulative_data[CHB][sp] / r_points;
        if (processing_mask & CHASQ_MASK) data_out[CHASQ][sp] = (double)flat_cumulative_data[CHASQ][sp] / r_points;
        if (processing_mask & CHBSQ_MASK) data_out[CHBSQ][sp] = (double)flat_cumulative_data[CHBSQ][sp] / r_points;
        if (processing_mask & SQ_MASK) data_out[SQ][sp] = (double)flat_cumulative_data[SQ][sp] / r_points;
    }
}

void power_kernel_runner(
    short* chA_data, short* chB_data, unsigned int** flat_cumulative_data,
    short *chA_back, short *chB_back,
    int start_idx, int stop_idx, int* cycle_array) {

    for (int i(start_idx); i < stop_idx; i++) {
        // As the background data only countains sp_points, a cycle wrap is required
        flat_cumulative_data[CHA][i] = chA_data[i] - chA_back[cycle_array[i]];
        flat_cumulative_data[CHB][i] = chB_data[i] - chB_back[cycle_array[i]];
        flat_cumulative_data[CHASQ][i] = flat_cumulative_data[CHA][i] * flat_cumulative_data[CHA][i];
        flat_cumulative_data[CHBSQ][i] = flat_cumulative_data[CHB][i] * flat_cumulative_data[CHB][i];
        flat_cumulative_data[SQ][i] = flat_cumulative_data[CHASQ][i] + flat_cumulative_data[CHBSQ][i];
    }
}

void POWER::CPU::power_kernel(
    short *chA_data, short *chB_data, double **data_out,
    unsigned int processing_mask,
    short *chA_back, short *chB_back,
    int sp_points,
    int r_points,
    int number_of_threads){

    int no_points  = sp_points * r_points;

    // 1. Prepare processing arrays
    unsigned int** flat_cumulative_data = new unsigned int*[NO_OF_POWER_KERNEL_OUTPUTS];
    for (int i(0); i < NO_OF_POWER_KERNEL_OUTPUTS; i++)
        flat_cumulative_data[i] = new unsigned int[no_points]();
    // As the background data only countains sp_points, an auxillary cycle array
    // will hold the valid indicies for accessing the bacgkound data
    int* cycle_array = new int[no_points];
    for (int sp(0); sp < sp_points; sp++) {
        for (int r(0); r < r_points; r++) {
            cycle_array[sp + r * sp_points] = sp;
        }
    }

    // 2. Preapare threads
    std::thread* t = new std::thread[number_of_threads];
    int idx = 0;
    int increment = floor((no_points) / number_of_threads);

    // 3. launch multiple parallel threads
    for (int i(0); i < number_of_threads - 1; i++) {
        t[i] = std::thread(
            power_kernel_runner,
            chA_data, chB_data, flat_cumulative_data,
            chA_back, chB_back,
            idx, idx + increment, cycle_array);
        idx += increment;
    }
    t[number_of_threads - 1] = std::thread(
        power_kernel_runner,
        chA_data, chB_data, flat_cumulative_data,
        chA_back, chB_back,
        idx, no_points, cycle_array);

     // 4. join the threads
    for (int i(0); i < number_of_threads; i++)
        t[i].join();

     // 5. Average the cumualtive arrays
    reduction_average(flat_cumulative_data, data_out,
                      processing_mask,
                      sp_points, r_points);

    // 6. Free processing arrays
    for (int i(0); i < NO_OF_POWER_KERNEL_OUTPUTS; i++)
        delete[] flat_cumulative_data[i];
    delete[] cycle_array;
    delete[] t;
}
