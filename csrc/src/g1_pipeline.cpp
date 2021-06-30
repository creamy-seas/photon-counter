#include "logging.hpp"
#include "sp_digitiser.hpp"
#include "g1_kernel.hpp"

// int run_power_measurements(void* adq_cu_ptr,
//                            short* chA_background, short* chB_background,
//                            unsigned long no_runs, char* base_filename){
//     const int no_threads = 3;

//     PYTHON_START;

//     G1::check_g1_kernel_parameters(false);

//     // 1. Allocation of memory
//     // There will be 3 copies of chA_data and chB_data.
//     // One thread can be reading into one pair (chA, chB),
//     // One thread will be preprocessing one pair (chA, chB)
//     // Other thread will be evaluating the correlatio (chA, chB)
//     short** chA_data = new short*[no_threads]();
//     short** chB_data = new short*[no_threads]();
//     double*** data_out = new long**[POWER::no_outputs];
//     for (int i(0); i < POWER::no_outputs; i++)
//         data_out[i] = new long[SP_POINTS]();

//     // 2. Prepare for multirecord mode
//     ADQ214_MultiRecordSetup(adq_cu_ptr, 1, R_POINTS, SP_POINTS);

//     // 3. Copy background data onto GPU
//     POWER::GPU::copy_background_arrays_to_gpu(chA_background, chB_background);

//     // 4. Launch 2 parrallel threads, alternating between fetching from digitiser and processing on GPU.
//     std::thread thread_list[NO_THREADS];
//     int dth(0), pth(1); // flip-floppers between 0 and 1. DigitizerTHread and ProcessingTHread

//     // Initial read into digitiser
//     fetch_digitiser_data(adq_cu_ptr, chA_data[dth], chB_data[dth], SP_POINTS, R_POINTS);
//     for (unsigned long r(1); r < no_runs; r++) {
//         // XOR to switch 0 <-> 1
//         dth ^= 1; pth ^= 1;

//         thread_list[0] = std::thread(fetch_digitiser_data,
//                                      adq_cu_ptr,
//                                      chA_data[dth], chB_data[dth],
//                                      SP_POINTS, R_POINTS);

//         thread_list[1] = std::thread(process_digitiser_data,
//                                      chA_data[pth], chB_data[pth],
//                                      data_out,
//                                      gpu_in, gpu_out, cpu_out,
//                                      NO_GPU_STREAMS,
//                                      r, base_filename);
//         thread_list[0].join();
//         thread_list[1].join();
//     }
//     dth ^= 1; pth ^= 1;
//     // Final processing of digitiser data
//     process_digitiser_data(
//         chA_data[pth], chB_data[pth],
//         data_out,
//         gpu_in, gpu_out, cpu_out,
//         NO_GPU_STREAMS,
//         no_runs, base_filename);

//     // Deallocation of memory
//     POWER::GPU::free_memory(chA_data[0], chB_data[0], gpu_in, gpu_out, cpu_out, NO_GPU_STREAMS);
//     POWER::GPU::free_memory(chA_data[1], chB_data[1], 0, 0, 0, NO_GPU_STREAMS);
//     delete[] chA_data;
//     delete[] chB_data;
//     for (int i(0); i < POWER::no_outputs; i++)
//         delete[] data_out[i];
//     delete[] data_out;

//     // Resetting device
//     ADQ214_MultiRecordClose(adq_cu_ptr, 1);

//     PYTHON_END;
//     return 0;
// }
