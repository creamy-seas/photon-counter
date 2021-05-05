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
    unsigned int* sq_data,
    int no_points,
    int number_of_threads
    ) {
    std::thread* t = new std::thread[number_of_threads];
    int idx = 0;
    int increment = floor((no_points) / number_of_threads);

    // 1. launch multiple parallel threads
    for (int i(0); i < number_of_threads - 1; i++) {
        t[i] = std::thread(
            power_kernel_v1_no_background_runner,
            chA_data, chB_data, sq_data,
            idx, idx + increment);
        idx += increment;
    }
    t[number_of_threads - 1] = std::thread(
        power_kernel_v1_no_background_runner,
        chA_data, chB_data, sq_data,
        idx, no_points);

    //2. join the threads
    for (int i(0); i < number_of_threads; i++)
        t[i].join();
    delete[] t;
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
    unsigned int* sq_data,
    short chA_back, short chB_back,
    int no_points,
    int number_of_threads
    ) {

    std::thread* t = new std::thread[number_of_threads];
    int idx = 0;
    int increment = floor((no_points) / number_of_threads);

    // 1. launch multiple parallel threads
    for (int i(0); i < number_of_threads - 1; i++) {
        t[i] = std::thread(
            power_kernel_v2_const_background_runner,
            chA_data, chB_data, sq_data,
            chA_back, chB_back,
            idx, idx + increment);
        idx += increment;
    }
    t[number_of_threads - 1] = std::thread(
        power_kernel_v2_const_background_runner,
        chA_data, chB_data, sq_data,
        chA_back, chB_back,
        idx, no_points);

    //2. join the threads
    for (int i(0); i < number_of_threads; i++)
                t[i].join();
        delete[] t;
}

void power_kernel_v3_background_runner(
    short* chA_data, short* chB_data, unsigned int* sq_data,
    short *chA_back, short *chB_back,
    int start_idx, int stop_idx) {

    for (int i(start_idx); i < stop_idx; i++) {
        chA_data[i] -= chA_back[i];
        chB_data[i] -= chB_back[i];
        sq_data[i] = chA_data[i] * chA_data[i] + chB_data[i] * chB_data[i];
    }
}

void CPU::power_kernel_v3_background(
    short *chA_data, short *chB_data, unsigned int *sq_data,
    short *chA_back, short *chB_back,
    int no_points, int number_of_threads){

    std::thread* t = new std::thread[number_of_threads];
    int idx = 0;
    int increment = floor((no_points) / number_of_threads);

    // 1. launch multiple parallel threads
    for (int i(0); i < number_of_threads - 1; i++) {
        t[i] = std::thread(
            power_kernel_v3_background_runner,
            chA_data, chB_data, sq_data,
            chA_back, chB_back,
            idx, idx + increment);
        idx += increment;
    }
    t[number_of_threads - 1] = std::thread(
        power_kernel_v3_background_runner,
        chA_data, chB_data, sq_data,
        chA_back, chB_back,
        idx, no_points);

    //2. join the threads
    for (int i(0); i < number_of_threads; i++)
                t[i].join();
        delete[] t;
}
