#include <thread>
#include <iostream>
#include <string>
#include "colours.h"
/* #include "ADQILYAAPI_x64.h" */
/* #include "DLL_Imperium.h" */
/* #include <fstream> */
/* #include "fftw3.h" */
/* #include <ctime> */

void power_kernel(
        short* chA_data, short* chB_data, unsigned int* sq_data,
        unsigned int samples_per_record,
        int start_idx, int stop_idx) {

        for (int i(start_idx); i < stop_idx; i++) {
                sq_data[i] = chA_data[i] * chA_data[i] + chB_data[i] * chB_data[i];
        }
}

void power_kernel(
        short* chA_data, short* chB_data, unsigned int* sq_data,
        short chA_back, short chB_back,
        unsigned int samples_per_record,
        int start_idx, int stop_idx) {

        for (int i(start_idx); i < stop_idx; i++) {
                chA_data[i] -= chA_back;
                chB_data[i] -= chB_back;
                sq_data[i] = chA_data[i] * chA_data[i] + chB_data[i] * chB_data[i];
        }
}

void power_kernel(
        short* chA_data, short* chB_data, unsigned int* sq_data,
        short* chA_back, short* chB_back,
        unsigned int samples_per_record,
        int start_idx, int stop_idx) {

        for (int i(start_idx); i < stop_idx; i++) {
                chA_data[i] -= chA_back[i];
                chB_data[i] -= chB_back[i];
                sq_data[i] = chA_data[i] * chA_data[i] + chB_data[i] * chB_data[i];
        }
}
