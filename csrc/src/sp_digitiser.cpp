#include "ADQAPI.h"
#include <stdio.h>

// Project Libraries
#include "logging.hpp"
#include "sp_digitiser.hpp"

#define INTERNAL_CLOCK_SOURCE_EXTERNAL_10MHZ_REFFERECNCE

int GetMaxNofSamplesFromNofRecords(void* adq_cu_ptr, int no_of_records){
    unsigned int max_number_of_samples = 0;
    ADQ_GetMaxNofRecordsFromNofSamples(adq_cu_ptr, 1, no_of_records, &max_number_of_samples);
    return max_number_of_samples;
}

int GetMaxNofRecordsFromNofSamples(void* adq_cu_ptr, int no_of_samples){
    unsigned int max_number_of_records = 0;
    ADQ_GetMaxNofRecordsFromNofSamples(adq_cu_ptr, 1, no_of_samples, &max_number_of_records);
    return max_number_of_records;
}

void* master_setup(int blink, int clock_source, unsigned int trigger_mode) {

    //1. Create an instance of the ADQControlUnit, required to find and setup ADQ devices
    void* adq_cu_ptr = CreateADQControlUnit();
    if (!ADQControlUnit_EnableErrorTrace(adq_cu_ptr, LOG_LEVEL_INFO, ".")) FAIL("DIGITISER: Failed to enable error logging.");

    //2. Find all ADQ units connect them. Store the address in adq_cu_ptr variable
    int no_of_devices = ADQControlUnit_FindDevices(adq_cu_ptr);
    if(no_of_devices == 0) {
        FAIL("No devices found! Make sure all programs refferencing devices are closed and that the box is switched on. When rebooting, turn the pc on after the digitiser.");
    }

    // Hard coded /////////////////////////////////////////////////////////
    // Set the data format to 14 bit unpacked, to map 1to1 the collected data memory inefficiently, but quickly
    if (!ADQ_SetDataFormat(adq_cu_ptr, 1, ADQ214_DATA_FORMAT_PACKED_14BIT)) FAIL("DIGITISER: Failed to set data format.");

    // Synthesise 400MHz signal from the 10MHz one
    // (phase locked loop generates a clock @ f*800/divider_value, so in this case its 400MHz. That is the sampling frequency)
    if (!ADQ_SetPllFreqDivider(adq_cu_ptr, 1, 2)) FAIL("DIGITISER: Failed to setup freq divider.");

    if (blink == 1) {
        YELLOWBG("Blinking");
        if (!ADQ_Blink(adq_cu_ptr, 1)) FAIL("DIGITISER: Failed to blink.");
        OKGREEN("Blinking complete!");
    }

    switch (trigger_mode) {
    case TRIGGER_SOFTWARE:
        OKBLUE("Software trigger");
        if (!ADQ_SetInternalTriggerPeriod(adq_cu_ptr, 1, 1)) FAIL("DIGITISER: Failed to set software trigger period.");
        break;
    case TRIGGER_EXTERNAL:
        WARNING("External trigger!");
        break;
    case TRIGGER_LEVEL:
        FAIL("Level trigger!");
    default :
        FAIL("Please select valid trigger!");
    }

    if (!ADQ_SetTriggerMode(adq_cu_ptr, 1, trigger_mode)) FAIL("DIGITISER: Failed to set the trigger mode.");

    clock_source == INTERNAL_CLOCK_SOURCE_EXTERNAL_10MHZ_REFFERENCE ? WARNING("External clock!") : OKBLUE("Internal clock");
    if (!ADQ_SetClockSource(adq_cu_ptr, 1, clock_source)) FAIL("DIGITISER: Failed set the clock source.");

    return adq_cu_ptr;
}

void fetch_digitiser_data(
        void* adq_cu_ptr,
        short* buff_a, short* buff_b,
        unsigned int samples_per_record,
        unsigned int number_of_records
        ) {

        ADQ_DisarmTrigger(adq_cu_ptr, 1);
        ADQ_ArmTrigger(adq_cu_ptr, 1);

        YELLOWBG("Begginning aquisition");

        while (!ADQ_GetAcquiredAll(adq_cu_ptr, 1)){
#ifdef TESTENV
            WARNING("Software trigger in test environment");
            ADQ_SWTrig(adq_cu_ptr, 1);
#endif
        }

        YELLOWBG("Begginning transfer");
        void* target_buffers[2] = { buff_a, buff_b };
        ADQ_GetData(adq_cu_ptr,
                    1,
                    target_buffers,
                    number_of_records * samples_per_record, // target bugger size
                    2, // bytes per sample
                    0, // start record
                    number_of_records,
                    FETCH_CHANNEL_BOTH,
                    0, // start sample
                    samples_per_record, // number of samples
                    0x00 // transfer mode - default
                );

        ADQ_DisarmTrigger(adq_cu_ptr, 1);

        OKGREEN("Completed read from digitiser");
}
