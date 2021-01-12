#include <thread> // thread
#include <iostream>
#include <string>
#include <math.h> // floor

#include "power.hpp"
/* #include "ADQILYAAPI_x64.h" */
/* #include "DLL_Imperium.h" */
/* #include <fstream> */
/* #include "fftw3.h" */
/* #include <ctime> */

/*
 * Equivalent power evaluation based of the signal from the two channels
 */

void power_kernel_v1_runner(
        short* chA_data, short* chB_data, unsigned int* sq_data,
        int samples_per_record,
        int start_idx, int stop_idx) {

        for (int i(start_idx); i < stop_idx; i++) {
                sq_data[i] = chA_data[i] * chA_data[i] + chB_data[i] * chB_data[i];
        }
}

/*
 * Equivalent power evaluation based of the signal from the two channels
 */
// void power_kernel_base(
//         short* chA_data, short* chB_data, unsigned int* sq_data,
//         short chA_back, short chB_back,
//         int samples_per_record,
//         int start_idx, int stop_idx) {

//         for (int i(start_idx); i < stop_idx; i++) {
//                 chA_data[i] -= chA_back;
//                 chB_data[i] -= chB_back;
//                 sq_data[i] = chA_data[i] * chA_data[i] + chB_data[i] * chB_data[i];
//         }
// }

/*
  Data from ChA and ChB is corrected for background

  chA_data, chB_data:                     raw data from the digitiser
  sq_data:                                evaluate power
  chA_back, chB_back:                     background set of measurements
  samples_per_record, number of records
  number_of_threads:                      number of threads to launch
*/
void CPU::power_kernel_v1(
        short* chA_data,
        short* chB_data,
        unsigned int* sq_data,
        int samples_per_record,
        int number_of_records,
        int number_of_threads
        ) {

        std::thread* t = new std::thread[number_of_threads];
        int idx = 0;
        int increment = floor((samples_per_record *  number_of_records) / number_of_threads);

        // 1. launch multiple parallel threads
        for (int i(0); i < number_of_threads - 1; i++) {
                t[i] = std::thread(
                        power_kernel_v1_runner,
                        chA_data, chB_data, sq_data,
                        samples_per_record,
                        idx, idx + increment);
                idx += increment;
        }
        t[number_of_threads - 1] = std::thread(
                power_kernel_v1_runner,
                chA_data, chB_data, sq_data,
                samples_per_record,
                idx, samples_per_record *  number_of_records);

        //2. join the threads
        for (int i(0); i < number_of_threads; i++)
                t[i].join();
        delete[] t;
}


// /*
//   Data from ChA and ChB is corrected for background

//   chA_data, chB_data:                     raw data from the digitiser
//   sq_data:                                evaluate power
//   chA_back, chB_back:                     background set of measurements
//   samples_per_record, number of records
//   number_of_threads:                      number of threads to launch
// */
// void CPU::power(
//         short* chA_data, short* chB_data, unsigned int* sq_data,
//         short chA_back, short chB_back,
//         int samples_per_record, int number_of_records,
//         int number_of_threads) {

//         std::thread* t = new std::thread[number_of_threads];
//         int idx = 0;
//         int increment = floor((samples_per_record *  number_of_records) / number_of_threads);

//         // 1. launch multiple parallel threads
//         for (int i(0); i < number_of_threads - 1; i++) {
//                 t[i] = std::thread(power_kernel_base,
//                                    chA_data, chB_data, sq_data,
//                                    chA_back, chB_back,
//                                    samples_per_record,
//                                    idx, idx + increment);
//                 idx += increment;
//         }
//         t[number_of_threads - 1] = std::thread(power_kernel_base,
//                                                chA_data, chB_data, sq_data,
//                                                chA_back, chB_back,
//                                                samples_per_record,
//                                                idx, samples_per_record *  number_of_records);

//         //2. join the threads
//         for (int i(0); i < number_of_threads; i++)
//                 t[i].join();
//         delete[] t;
// }
