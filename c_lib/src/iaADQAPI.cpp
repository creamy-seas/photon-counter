#include "ADQAPI.h"
#include <stdio.h>
#include <stdexcept>

// Project Libraries
#include "colours.hpp"
#include "iaADQAPI.hpp"

#define ADQ1 (adq_cu_ptr, 1)

int GetMaxNofSamplesFromNofRecords(void* adq_cu_ptr, int no_of_records){
        unsigned int max_number_of_samples = 0;
        ADQ214_GetMaxNofRecordsFromNofSamples(adq_cu_ptr, 1, no_of_records, &max_number_of_samples);
        return max_number_of_samples;
}

int GetMaxNofRecordsFromNofSamples(void* adq_cu_ptr, int no_of_samples){
        unsigned int max_number_of_records = 0;
        ADQ214_GetMaxNofRecordsFromNofSamples(adq_cu_ptr, 1, no_of_samples, &max_number_of_records);
        return max_number_of_records;
}

// Connect to the ADQ unit and setup 400MHz data acquisition
void* master_setup(int blink) {

        // 1) Create an instance of the ADQControlUnit, required to find and setup ADQ devices
        void* adq_cu_ptr = CreateADQControlUnit();

        //2) Find all ADQ units connect them. Store the address in adq_cu_ptr variable
        int no_of_devices = ADQControlUnit_FindDevices(adq_cu_ptr);
        if(no_of_devices == 0) {
                FAIL("No devices found! Make sure all programs refferencing devices are closed");
                throw std::runtime_error("Failed to find devices");
        }

        if (blink == 1) {
                YELLOW("Blinking");
                ADQ214_Blink(adq_cu_ptr, 1);
                OKGREEN("Blinking complete!");
        }

        return adq_cu_ptr;

        // //3) Choose 10MHz external clock source (supplied via cable)
        // ADQ214_SetClockSource(adq_cu_ptr, 1, 1);			//choose extenral clock source (10MHz)
        // ADQ214_SetClockFrequencyMode(adq_cu_ptr, 1, 0);	//choose the correct range of the external refference

        // //4) Synthesise 400MHz signal from the 10MHz one (phase locked loop generates a clock @ f*800/divider_value, so in this case its 400MHz. That is the sampling frequency)
        // ADQ214_SetPllFreqDivider(adq_cu_ptr, 1, 2);

        // //5) Set the trigger to external (2)
        // ADQ214_SetTriggerMode(adq_cu_ptr, 1, 2);

        // //8) Set the data format to 14bit unpacked always
        // ADQ214_SetDataFormat(adq_cu_ptr, 1, 1);

        // //default mask is everything on
        // _MASK = 0b111;



}

void fetch_channel_data(
        void* adq_cu_ptr,
        short* buff_a, short* buff_b,
        unsigned int samples_per_record,
        unsigned int number_of_records
        ) {

        ADQ214_ArmTrigger(adq_cu_ptr, 1);

        OKBLUE("Begginning aquisition");
        int finished_reading(0);
        while (!finished_reading)
                finished_reading = ADQ214_GetAcquiredAll(adq_cu_ptr, 1);

        void* target_buffers[2] = { buff_a, buff_b };
        ADQ214_GetData(adq_cu_ptr,
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
        ADQ214_DisarmTrigger(adq_cu_ptr, 1);

        OKBLUE("Completed read from digitiser");
}
