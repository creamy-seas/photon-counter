#include <stdio.h>
#include "ADQAPI.h"

int main(void)
{
        void* adq_cu_ptr = CreateADQControlUnit();
        int no_of_devices = ADQControlUnit_FindDevices(adq_cu_ptr);
        if(no_of_devices == 0) printf("!!** No devices founds. Make sure \"AD Capture lab is closed\" -> Restart the NI PXIE-1065 crate -> Restart computer");

        unsigned int no_of_records = 1;
        unsigned int max_number_of_samples;

        ADQ214_GetMaxNofSamplesFromNofRecords(adq_cu_ptr, 1, no_of_records, &max_number_of_samples);

        printf("%i", max_number_of_samples);
        /* ADQ214_Blink(adq_cu_ptr, 1); */

                return 0;
}
