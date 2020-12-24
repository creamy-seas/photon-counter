#ifndef _IA_ADQAPI_
#define _IA_ADQAPI_

// Constants //////////////////////////////////////////////////////////////////
static unsigned char FETCH_CHANNEL_A = 0x1;
static unsigned char FETCH_CHANNEL_B = 0x2;
static unsigned char FETCH_CHANNEL_BOTH = 0x3;

static unsigned int TRIGGER_EXTERNAL = 1;
static unsigned int TRIGGER_INTERNAL = 0;

// Rewrapped functions ////////////////////////////////////////////////////////
int GetMaxNofSamplesFromNofRecords(void* adq_cu_ptr, int no_of_records);
int GetMaxNofRecordsFromNofSamples(void* adq_cu_ptr, int no_of_samples);

// Custom functions ///////////////////////////////////////////////////////////
void fetch_channel_data(
        void* adq_cu_ptr,
        short* buff_a, short* buff_b,
        unsigned int samples_per_record,
        unsigned int number_of_records
        );

void* master_setup(int);
#endif
