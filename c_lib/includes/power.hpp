#ifndef POWER_HPP
#define POWER_HPP

namespace CPU {

        /*
          short* chA_data, chB_data:              raw data from the digitiser
          unsigned int* sq_data:                  evaluate power
          int no_points = samples_per_record * number of records
          number_of_threads:                      number of threads to launch
        */
        void power_kernel(
                short* chA_data,
                short* chB_data,
                unsigned int* sq_data,
                int no_points,
                int number_of_threads
                );

        /*
          short* chA_data, chB_data:              raw data from the digitiser
          unsigned int* sq_data:                  evaluate power
          short chA_back, chB_back:               background set of measurements
          int no_points = samples_per_record * number of records
          number_of_threads:                      number of threads to launch
        */
        void power_kernel(
                short *chA_data,
                short *chB_data,
                unsigned int *sq_data,
                short chA_back,
                short chB_back,
                int no_points,
                int number_of_threads
                );

        /*
          short* chA_data, chB_data:              raw data from the digitiser
          unsigned int* sq_data:                  evaluate power
          short* chA_back, chB_back:              background set of measurements
          int no_points = samples_per_record * number of records
          number_of_threads:                      number of threads to launch
        */
        void power_kernel(
                short *chA_data,
                short *chB_data,
                unsigned int *sq_data,
                short *chA_back,
                short *chB_back,
                int no_points,
                int number_of_threads
                );
}

namespace GPU {

        float power_kernel(
                short a,
                short b);

        /*
          short* chA_data, chB_data:              raw data from the digitiser
          unsigned int* sq_data:                  evaluate power
          int samples_per_record, number of records
          number_of_threads:                      number of threads to launch
        */
        // void power_kernel(
        //         short* chA_data,
        //         short* chB_data,
        //         unsigned int* sq_data,
        //         int samples_per_record,
        //         int number_of_records
        //         );

        /*
          short* chA_data, chB_data:              raw data from the digitiser
          unsigned int* sq_data:                  evaluate power
          short chA_back, chB_back:               background set of measurements
          int no_points = samples_per_record * number of records
          number_of_threads:                      number of threads to launch
        */
        // void power_kernel(
        //         short *chA_data,
        //         short *chB_data,
        //         unsigned int *sq_data,
        //         short chA_back,
        //         short chB_back,
        //         int no_points,
        //         int number_of_threads
        //         );

        /*
          short* chA_data, chB_data:              raw data from the digitiser
          unsigned int* sq_data:                  evaluate power
          short* chA_back, chB_back:              background set of measurements
          int no_points = samples_per_record * number of records
          number_of_threads:                      number of threads to launch
        */
        // void power_kernel(
        //         short *chA_data,
        //         short *chB_data,
        //         unsigned int *sq_data,
        //         short *chA_back,
        //         short *chB_back,
        //         int no_points,
        //         int number_of_threads
        //         );
}

#endif
