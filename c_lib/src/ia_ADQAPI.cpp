#include "ADQAPI.h"
#include <sdtio.h>

// Project Libraries
#include "colours.hpp"
#include "digiti"

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

void fetch_channel_data(
        void* adq_cu_ptr,
        short* buff_a, short* buff_b,
        unsigned int samples_per_record,
        unsigned int number_of_records
        ) {

        ADQ214_ArmTrigger(adq_cu_ptr, 1);

        printf("%s\n", OKBLUE("Begginning aquisition"));
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

        printf("%s\n", OKBLUE("Completed read from digitiser"));
}
