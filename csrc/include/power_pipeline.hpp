#ifndef POWER_PIPELINE_HPP
#define POWER_PIPELINE_HPP

#include <string>

/**
 * Run pipeline for power measurements using 2 threads parallel that alternate between:
 * - Read in from digitiser
 * - Evaluate CHA, CHB, CHASQ, CHBSQ, SQ and dumping to file
 *
 * Ensure that:
 * - The GPU kernel has been built with defined `SP_POINTS`, `R_POINTS` and `R_POINTS_PER_CHUNK`.
 * - `MultiRecordSetup` has been run to prepare the digitizer for measurements.
 * - `MultiRecordClose` is run after the function to reset digitiser.
 *
 * @param adq_cu_ptr Allocated pointer for communication with digitizer.
 * @param no_repetitions Number of times to repeat the measurements.
 * @param output_filename File to which to dump the results.
 */
void run_power_measurements(void* adq_cu_ptr, unsigned long no_repetitions, std::string output_filename);

#endif
