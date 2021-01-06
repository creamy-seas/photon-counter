#include "ADQAPI.h"
#include <stdio.h>
#include <stdexcept>

// Project Libraries
#include "colours.hpp"
#include "iaADQAPI.hpp"

#define ADQ1 (adq_cu_ptr, 1)
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

// Connect to the ADQ unit and setup 400MHz data acquisition
void* master_setup(int blink, int clock_source, unsigned int trigger_mode) {

        //1. Create an instance of the ADQControlUnit, required to find and setup ADQ devices
        void* adq_cu_ptr = CreateADQControlUnit();
        ADQControlUnit_EnableErrorTrace(adq_cu_ptr, LOG_LEVEL_INFO, ".");

        //2. Find all ADQ units connect them. Store the address in adq_cu_ptr variable
        int no_of_devices = ADQControlUnit_FindDevices(adq_cu_ptr);
        if(no_of_devices == 0) {
                FAIL("No devices found! Make sure all programs refferencing devices are closed");
                throw std::runtime_error("Failed to find devices");
        }

        // Hard coded /////////////////////////////////////////////////////////
        // Set the data format to 14 bit unpacked, to map 1to1 the collected data memory inefficiently, but quickly
        ADQ_SetDataFormat(adq_cu_ptr, 1, 1);

        // Synthesise 400MHz signal from the 10MHz one
        // (phase locked loop generates a clock @ f*800/divider_value, so in this case its 400MHz. That is the sampling frequency)
        ADQ_SetPllFreqDivider(adq_cu_ptr, 1, 2);

        // Variable
        if (blink == 1) {
                YELLOWBG("Blinking");
                ADQ_Blink(adq_cu_ptr, 1);
                OKGREEN("Blinking complete!");
        }

        switch (trigger_mode){
        case TRIGGER_SOFTWARE:
                OKBLUE("Software trigger");
                ADQ_SetInternalTriggerPeriod(adq_cu_ptr, 1, 100);
                break;
        case TRIGGER_EXTERNAL:
                WARNING("External trigger!");
                break;
        case TRIGGER_LEVEL:
                WARNING("Level trigger!");
                throw "Level trigger is not setup!";
        default :
                throw "Please select valid trigger!";
        }
        ADQ_SetTriggerMode(adq_cu_ptr, 1, trigger_mode);

        clock_source ==  INTERNAL_CLOCK_SOURCE_EXTERNAL_10MHZ_REFFERENCE ? WARNING("External clock!") : OKBLUE("Internal clock");
        ADQ_SetClockSource(adq_cu_ptr, 1, clock_source);

        return adq_cu_ptr;
}

void fetch_channel_data(
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
