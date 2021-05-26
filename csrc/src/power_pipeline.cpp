#include <stdexcept> //for std::runtime_error
#include <string> // for std::to_string
#include <thread> // for std::thread
#include <limits.h> // For LONG_MAX

#include "colours.hpp"
#include "utils.hpp"
#include "ia_ADQAPI.hpp"
#include "power_pipeline.hpp"
#include "power_kernel.hpp"

const int NO_THREADS = 2; ///< One thread will be used to request data from digitiser. Other thread will process the data and store it in file
const int NO_GPU_STREAMS = 2; ///< Benchmarking showed that 2 streams should be used on the GPU for best performance.

/**
 * \copy GPU::power_kernel
 *
 * Evaluates the following from chA_data and chB_data data
 * - \f[ \left\langle{chA_data}\right\rangle \f]
 * - \f[ \left\langle{chB_data}\right\rangle \f]
 * - \f[ \left\langle{chA_data^2}\right\rangle \f]
 * - \f[ \left\langle{chB_data^2}\right\rangle \f]
 * - \f[ \left\langle{chA_data^2 + chB_data^2}\right\rangle \f]
 *
 * @param chA_data, chB_data data from the digitiser. Must be of length `SP_POINTS * R_POINTS`.
 * @param data_out cumulative array which will be incremented with the power kernel outputs
 * @param repetition cumulative data is normalised by the number runs completed
 * @param output_filename where data is dumped so that Python can plot it.
 */
void process_digitiser_data(short *chA_data, short *chB_data,
                            long **data_out,
                            short ***gpu_in, long ***gpu_out, long ***cpu_out, int no_streams,
                            unsigned long repetition, std::string output_filename){
    int normalisation = GPU::power_kernel(
        chA_data, chB_data,
        data_out,
        gpu_in, gpu_out, cpu_out, no_streams);

    normalisation *= repetition;

    dump_arrays_to_file(data_out,
                        NO_OF_POWER_KERNEL_OUTPUTS,
                        SP_POINTS,
                        output_filename,
                        "CHA\tCHB\tCHASQ\tCHBSQ\tSQ",
                        (double)normalisation);

};

void run_power_measurements(void* adq_cu_ptr, unsigned long no_repetitions, std::string output_filename){

    // Casting to largest data type of comparisson
    if ((unsigned long long)no_repetitions * MAX_DIGITISER_CODE * R_POINTS
        >
        (unsigned long long)LONG_MAX)
        throw std::runtime_error("No repetitions ("
                                 + std::to_string(no_repetitions)
                                 + ") x 14bit Code ("
                                 + std::to_string(MAX_DIGITISER_CODE)
                                 + ") x R_POINTS(number of records per point="
                                 + std::to_string(R_POINTS)
                                 + ") can overflow the cumulative arrays of type LONG ("
                                 + std::to_string(LONG_MAX)
                                 + ")");

    // 1. Allocation of memory
    // There will be 2 copies of chA_data and chB_data.
    // One thread can be reading into one pair (chA, chB),
    // Other thread can be processing the other pair (chA, chB)
    short** chA_data = new short*[NO_THREADS];
    short** chB_data = new short*[NO_THREADS];
    GPU::allocate_memory(&chA_data[1], &chB_data[1], 0, 0, 0, NO_GPU_STREAMS);

    // Single copy of these, since only the processing thread will use them
    short ***gpu_in; long ***gpu_out; long ***cpu_out;
    GPU::allocate_memory(&chA_data[0], &chB_data[0], &gpu_in, &gpu_out, &cpu_out, NO_GPU_STREAMS);
    long** data_out = new long*[NO_OF_POWER_KERNEL_OUTPUTS];
    for (int i(0); i < NO_OF_POWER_KERNEL_OUTPUTS; i++)
        data_out[i] = new long[SP_POINTS];

    // 2. Launch 2 parrallel threads, alternating between fetching from digitiser and processing on GPU.
    std::thread thread_list[NO_THREADS];
    int dth(0), pth(1); // flip-floppers between 0 and 1. DigitizerTHread and ProcessingTHread

    // Initial read into digitiser
    fetch_digitiser_data(adq_cu_ptr, chA_data[dth], chB_data[dth], SP_POINTS, R_POINTS);
    for (unsigned long r(1); r < no_repetitions; r++) {
        // XOR to switch 0 <-> 1
        dth ^= 1; pth ^= 1;

        thread_list[0] = std::thread(fetch_digitiser_data,
                                     adq_cu_ptr,
                                     chA_data[dth], chB_data[dth],
                                     SP_POINTS, R_POINTS);

        thread_list[1] = std::thread(process_digitiser_data,
                                     chA_data[pth], chB_data[pth],
                                     data_out,
                                     gpu_in, gpu_out, cpu_out,
                                     NO_GPU_STREAMS,
                                     r, output_filename);
        thread_list[dth].join();
        thread_list[pth].join();
    }
    dth ^= 1; pth ^= 1;
    // Closing accumulation
    process_digitiser_data(
        chA_data[pth], chB_data[pth],
        data_out,
        gpu_in, gpu_out, cpu_out,
        NO_GPU_STREAMS,
        no_repetitions, output_filename);

    // Deallocation of memory
    GPU::free_memory(chA_data[0], chB_data[0], gpu_in, gpu_out, cpu_out, NO_GPU_STREAMS);
    GPU::free_memory(chA_data[1], chB_data[1], 0, 0, 0, NO_GPU_STREAMS);
    delete[] chA_data;
    delete[] chB_data;
    for (int i(0); i < NO_OF_POWER_KERNEL_OUTPUTS; i++)
        delete[] data_out[i];
    delete[] data_out;
};
