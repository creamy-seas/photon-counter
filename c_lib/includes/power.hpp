#ifndef POWER_HPP
#define POWER_HPP

namespace CPU {

        void power_kernel_v2(
                short* chA_data,
                short* chB_data,
                unsigned int* sq_data,
                short chA_back,
                short chB_back,
                int samples_per_record,
                int number_of_records,
                int number_of_threads
                );
        void power_kernel_v1(
                short* chA_data,
                short* chB_data,
                unsigned int* sq_data,
                int samples_per_record,
                int number_of_records,
                int number_of_threads
                );

}

#endif
