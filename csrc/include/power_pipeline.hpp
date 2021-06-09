#ifndef POWER_PIPELINE_HPP
#define POWER_PIPELINE_HPP

#include <string>

#define LOG_ROTATE 100 ///< Rotation of log files for safe reading/writting between Cpp and python

#ifdef __cplusplus
extern "C" {
#endif
    /**
     * Run pipeline for power measurements using 2 threads parallel that alternate between:
     * - Read in from digitiser
     * - Evaluate CHA, CHB, CHASQ, CHBSQ, SQ and dumping to file
     *
     * Ensure that:
     * - The GPU kernel has been built with defined `SP_POINTS`, `R_POINTS` and `R_POINTS_PER_CHUNK`.
     * - Background data has been copied over to the GPU
     * - `MultiRecordSetup` has been run to prepare the digitizer for measurements.
     * - `MultiRecordClose` is run after the function to reset digitiser.
     *
     * @param adq_cu_ptr Allocated pointer for communication with digitizer.
     * @param chA_background, chB_background of length SP_POINTS each. These will be subtracted from the channels before processing begins.
     * @param no_runs Number of times to repeat the measurements.
     * @param base_filename File to which to dump the results, using log-rotation format `base_filename_X.txt`
     * @returns 0 for success.
     */
    int run_power_measurements(void* adq_cu_ptr,
                               short *chA_background, short *chB_background,
                               unsigned long no_runs, char* base_filename);
#ifdef __cplusplus
}
#endif

// Unit testing and bechmarking
#ifdef TESTENV
void process_digitiser_data(short *chA_data, short *chB_data,
                            long **data_out,
                            short ***gpu_in, long ***gpu_out, long ***cpu_out, int no_streams,
                            unsigned long repetition, std::string base_filename);
#endif

#endif
