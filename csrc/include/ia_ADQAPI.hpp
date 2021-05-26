/**
 * Library for basic communication with the digitiser
 * Recommended to use python for any setup as it is easier to follow
 */
#ifndef _IA_ADQAPI_
#define _IA_ADQAPI_

// Constants
// static unsigned char FETCH_CHANNEL_A = 0x1;
// static unsigned char FETCH_CHANNEL_B = 0x2;
static unsigned char FETCH_CHANNEL_BOTH = 0x3;

// UL suffix in order to compare with unsigned long int in the code
#define MAX_DIGITISER_CODE (1UL << 14) ///< Digitiser is 14 bit
#define MAX_NUMBER_OF_RECORDS 254200UL

#define TRIGGER_SOFTWARE 1
#define TRIGGER_EXTERNAL 2
#define TRIGGER_LEVEL 3

#define INTERNAL_CLOCK_SOURCE_INTERNAL_10MHZ_REFFERENCE 0
#define INTERNAL_CLOCK_SOURCE_EXTERNAL_10MHZ_REFFERENCE 1

// static int LOW_FREQUENCY_MODE = 0;
// static int HIGH_FREQUENCY_MODE = 1;

#define BLINK 1
#define NO_BLINK 0

#ifdef __cplusplus
extern "C" {
#endif
    /**
     * Rewrapped function not available in python.
     *
     * @param adq_cu_ptr Allocated pointer for communication with digitizer.
     *
     * @returns maximum number of samples that the digitizer can acuumulate for supplied number of repetitions
     */
    int GetMaxNofSamplesFromNofRecords(void* adq_cu_ptr, int no_of_records);

    /**
     * Rewrapped function not available in python.
     *
     * @param adq_cu_ptr Allocated pointer for communication with digitizer
     *
     * @returns maximum number of records that the digitizer can acuumulate for supplied number of samples
     */
    int GetMaxNofRecordsFromNofSamples(void* adq_cu_ptr, int no_of_samples);

    /**
     * Reading and transfering data from digitiser. Ensure that `MultiRecordSetup` has been called.
     *
     * @param adq_cu_ptr Allocated pointer for communication with digitizer
     * @param buff_a, buff_b Allocataed memeory where data will be read into
     * @param samples_per_record, number_of_records
     */
    void fetch_digitiser_data(
        void* adq_cu_ptr,
        short* buff_a, short* buff_b,
        unsigned int samples_per_record,
        unsigned int number_of_records
        );

    /**
     * Only use for development in c++
     */
    void* master_setup(int blink, int clock_source, unsigned int trigger_type);

#ifdef __cplusplus
}
#endif

#endif
