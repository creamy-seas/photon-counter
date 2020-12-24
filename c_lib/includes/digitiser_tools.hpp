#ifndef _DIGITISER_TOOLS
#define _DIGITISER_TOOLS

void fetch_channel_data(
        void* adq_cu_ptr,
        short* buff_a, short* buff_b,
        unsigned int samples_per_record,
        unsigned int number_of_records
        );
#endif
