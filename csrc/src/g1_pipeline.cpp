#include <thread> // for std::thread
#include "ADQAPI.h" // For MultiRecordSetup and MultiRecordClose
#include <string>

#include "logging.hpp"
#include "sp_digitiser.hpp"
#include "g1_kernel.hpp"
#include "g1_pipeline.hpp"
#include "utils.hpp"

/**
 * \copy POWER::GPU::power_kernel
 *
 * Evaluates the following from chA_data and chB_data data
 * - \f[ \left\langle{chA_data}\right\rangle \f]
 * - \f[ \left\langle{chB_data}\right\rangle \f]
 * - \f[ \left\langle{chA_data^2}\right\rangle \f]
 * - \f[ \left\langle{chB_data^2}\right\rangle \f]
 * - \f[ \left\langle{chA_data^2 + chB_data^2}\right\rangle \f]
 *
 * Data is dumped to a file using log rotation format.
 *
 * @param chA_data, chB_data data from the digitiser. Must be of length `SP_POINTS * R_POINTS`.
 * @param cumulative cumulative array which will be incremented with the power kernel outputs
 * @param run cumulative data is normalised by the number runs completed. This also specifies the unique log rotate command
 * @param base_filename Used to ctonar data is dumped so that Python can plot it.
 */
void process_digitiser_data(
    short *chA_data, short *chB_data, float **cumulative,
    G1::GPU::g1_memory memory,
    cufftHandle *plans_forward, cufftHandle *plans_backward,
    unsigned long run, std::string base_filename){

    G1::GPU::g1_kernel(
        chA_data, chB_data,
        memory,
        plans_forward, plans_backward);

    dump_arrays_to_file(
        memory.cpu_out, cumulative,
        G1::no_outputs,
        100,
        base_filename + std::to_string(run % LOG_ROTATE) + ".csv",
        "# Run " + std::to_string(run) +  "\n# CHAG1\tCHBG1\tSQG1",
        (double)run
        );
};

int run_g1_measurements(void* adq_cu_ptr,
                        unsigned long no_runs, char* base_filename){
    const int no_threads = 2;

    PYTHON_START;

    G1::check_g1_kernel_parameters(false);

    cufftHandle *plans_forward(0); cufftHandle *plans_backward(0);
    G1::GPU::g1_prepare_fftw_plan(plans_forward, plans_backward);

    // Allocation of memory
    G1::GPU::g1_memory memory = G1::GPU::allocate_memory();

    // There will be 2 copies of chA_data and chB_data.
    // One thread can be reading into one pair (chA, chB),
    // Other thread will be evaluating the correlatio (chA, chB)
    short** chA_data = new short*[no_threads]();
    short** chB_data = new short*[no_threads]();
    float** cumulative = new float*[G1::no_outputs];
    for (int i(0); i < G1::no_outputs; i++)
        cumulative[i] = new float[SP_POINTS]();

    // 2. Prepare for multirecord mode
    ADQ214_MultiRecordSetup(adq_cu_ptr, 1, R_POINTS, SP_POINTS);

    // 4. Launch 2 parrallel threads, alternating between fetching from digitiser and processing on GPU.
    std::thread thread_list[no_threads];
    int dth(0), pth(1); // flip-floppers between 0 and 1. DigitizerTHread and ProcessingTHread

    // Initial read into digitiser
    fetch_digitiser_data(adq_cu_ptr, chA_data[dth], chB_data[dth], SP_POINTS, R_POINTS);
    for (unsigned long r(1); r < no_runs; r++) {
        // XOR to switch 0 <-> 1
        dth ^= 1; pth ^= 1;

        thread_list[0] = std::thread(fetch_digitiser_data,
                                     adq_cu_ptr,
                                     chA_data[dth], chB_data[dth],
                                     SP_POINTS, R_POINTS);

        thread_list[1] = std::thread(process_digitiser_data,
                                     chA_data[pth], chB_data[pth], cumulative,
                                     memory,
                                     plans_forward, plans_backward,
                                     r, base_filename);
        thread_list[0].join();
        thread_list[1].join();
    }
    dth ^= 1; pth ^= 1;
    // Final processing of digitiser data
    process_digitiser_data(chA_data[pth], chB_data[pth], cumulative,
                           memory,
                           plans_forward, plans_backward,
                           no_runs, base_filename);

    // Reset
    G1::GPU::free_memory(memory);
    ADQ214_MultiRecordClose(adq_cu_ptr, 1);

    PYTHON_END;
    return 0;
}
