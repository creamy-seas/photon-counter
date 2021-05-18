///////////////////////////////////////////////////////////////////////////////
//             Library for basic communication with the digitiser            //
//     Recommended to use python for any setup as it is easier to follow     //
///////////////////////////////////////////////////////////////////////////////
#ifndef _IA_ADQAPI_
#define _IA_ADQAPI_

// Constants //////////////////////////////////////////////////////////////////
// static unsigned char FETCH_CHANNEL_A = 0x1;
// static unsigned char FETCH_CHANNEL_B = 0x2;
static unsigned char FETCH_CHANNEL_BOTH = 0x3;

// Digitiser will output 2^14 code
// UL suffix in order to compare with unsigned long int in the code
#define MAX_CODE (1UL << 14)
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
        // Rewrapped functions ////////////////////////////////////////////////////////
        int GetMaxNofSamplesFromNofRecords(void* adq_cu_ptr, int no_of_records);
        int GetMaxNofRecordsFromNofSamples(void* adq_cu_ptr, int no_of_samples);
        void fetch_channel_data(
                void* adq_cu_ptr,
                short* buff_a, short* buff_b,
                unsigned int samples_per_record,
                unsigned int number_of_records
                );

        // Only for testing in C
        void* master_setup(int blink, int clock_source, unsigned int trigger_type);

#ifdef __cplusplus
}
#endif

#endif
