#include <thread> // thread
#include <iostream>
#include <string>
#include <math.h> // floor

#include "power_kernel.hpp"
/* #include "ADQILYAAPI_x64.h" */
/* #include "DLL_Imperium.h" */
/* #include <fstream> */
/* #include "fftw3.h" */
/* #include <ctime> */

void reduction_average(unsigned int* flat_sq_data, float* average_array, int sp_points, int r_points) {
    /*
     * Reduce the array by summing up the total into the first cell.

     __ Logic ___
     a1 a2 a3 a4 ... b1 b2 b3 b4 ... c1 c2 c3 c4 ...

     will be mapped to a 2D array

     a1 a2 a3 -> main_axis (sp_coordinate)
     b1 b2 b3 ...
     c1 c2 c3 ...
     d1 d2 d3 ...
     e1 e2 e3 ...
     f1 f2 f3 ...
     g1 g2 g3 ...
     |
     repetition-axis (r_coordinate)

     And reduced to the following by summing up over the repetition axis and normalising by the size
     <1> <2> <3> ...
    */

    for (int sp(0); sp < sp_points; sp++) {
        for (int r(1); r < r_points; r++) {
            flat_sq_data[sp] += flat_sq_data[sp + r * sp_points];
        }
        average_array[sp] = (float)flat_sq_data[sp] / r_points;
    }
}

void power_kernel_v1_no_background_runner(
    short* chA_data, short* chB_data, unsigned int* sq_data,
    int start_idx, int stop_idx) {

    for (int i(start_idx); i < stop_idx; i++) {
        sq_data[i] = chA_data[i] * chA_data[i] + chB_data[i] * chB_data[i];
    }
}

void CPU::power_kernel_v1_no_background(
    short* chA_data,
    short* chB_data,
    float* sq_data,
    int sp_points,
    int r_points,
    int number_of_threads
    ) {

    int no_points  = sp_points * r_points;
    unsigned int* flat_sq_data = new unsigned int[no_points];

    std::thread* t = new std::thread[number_of_threads];

    int idx = 0;
    int increment = floor((no_points) / number_of_threads);

    // 1. launch multiple parallel threads
    for (int i(0); i < number_of_threads - 1; i++) {
        t[i] = std::thread(
            power_kernel_v1_no_background_runner,
            chA_data, chB_data, flat_sq_data,
            idx, idx + increment);
        idx += increment;
    }
    t[number_of_threads - 1] = std::thread(
        power_kernel_v1_no_background_runner,
        chA_data, chB_data, flat_sq_data,
        idx, no_points);

    //2. join the threads
    for (int i(0); i < number_of_threads; i++)
        t[i].join();

    //3. reduce
    reduction_average(flat_sq_data, sq_data, sp_points, r_points);
    delete[] t;
    delete[] flat_sq_data;
}

void power_kernel_v2_const_background_runner(
    short* chA_data, short* chB_data, unsigned int* sq_data,
    short chA_back, short chB_back,
    int start_idx, int stop_idx) {

    for (int i(start_idx); i < stop_idx; i++) {
        chA_data[i] -= chA_back;
        chB_data[i] -= chB_back;
        sq_data[i] = chA_data[i] * chA_data[i] + chB_data[i] * chB_data[i];
    }
}

void CPU::power_kernel_v2_const_background(
    short* chA_data,
    short* chB_data,
    float* sq_data,
    short chA_back, short chB_back,
    int sp_points,
    int r_points,
    int number_of_threads
    ) {

    int no_points  = sp_points * r_points;
    unsigned int* flat_sq_data = new unsigned int[no_points];

    std::thread* t = new std::thread[number_of_threads];

    int idx = 0;
    int increment = floor((no_points) / number_of_threads);

    // 1. launch multiple parallel threads
    for (int i(0); i < number_of_threads - 1; i++) {
        t[i] = std::thread(
            power_kernel_v2_const_background_runner,
            chA_data, chB_data, flat_sq_data,
            chA_back, chB_back,
            idx, idx + increment);
        idx += increment;
    }
    t[number_of_threads - 1] = std::thread(
        power_kernel_v2_const_background_runner,
        chA_data, chB_data, flat_sq_data,
        chA_back, chB_back,
        idx, no_points);

    //2. join the threads
    for (int i(0); i < number_of_threads; i++)
        t[i].join();

    //3. reduce
    reduction_average(flat_sq_data, sq_data, sp_points, r_points);
    delete[] t;
}

void power_kernel_v3_background_runner(
    short* chA_data, short* chB_data, unsigned int* sq_data,
    short *chA_back, short *chB_back,
    int start_idx, int stop_idx, int* cycle_array) {

    for (int i(start_idx); i < stop_idx; i++) {
        // As the background data only countains sp_points, a cycle wrap is required
        chA_data[i] -= chA_back[cycle_array[i]];
        chB_data[i] -= chB_back[cycle_array[i]];
        sq_data[i] = chA_data[i] * chA_data[i] + chB_data[i] * chB_data[i];
    }
}

void CPU::power_kernel_v3_background(
    short *chA_data, short *chB_data, float *sq_data,
    short *chA_back, short *chB_back,
    int sp_points,
    int r_points,
    int number_of_threads){

    int no_points  = sp_points * r_points;
    unsigned int* flat_sq_data = new unsigned int[no_points];

    // As the background data only countains sp_points, an auxillary cycle array
    // will hold the valid indicies for accessing the bacgkound data when
    int* cycle_array = new int[no_points];
    for (int sp(0); sp < sp_points; sp++) {
        for (int r(0); r < r_points; r++) {
            cycle_array[sp + r * sp_points] = sp;
        }
    }

    std::thread* t = new std::thread[number_of_threads];

    int idx = 0;
    int increment = floor((no_points) / number_of_threads);

    // 1. launch multiple parallel threads
    for (int i(0); i < number_of_threads - 1; i++) {
        t[i] = std::thread(
            power_kernel_v3_background_runner,
            chA_data, chB_data, flat_sq_data,
            chA_back, chB_back,
            idx, idx + increment, cycle_array);
        idx += increment;
    }
    t[number_of_threads - 1] = std::thread(
        power_kernel_v3_background_runner,
        chA_data, chB_data, flat_sq_data,
        chA_back, chB_back,
        idx, no_points, cycle_array);

    //2. join the threads
    for (int i(0); i < number_of_threads; i++)
        t[i].join();

    //3. reduce
    reduction_average(flat_sq_data, sq_data, sp_points, r_points);
    delete[] flat_sq_data;
    delete[] t;
}
