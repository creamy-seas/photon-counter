#include "ADQAPI.h"

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
