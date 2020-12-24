#include <thread>
#include <iostream>
#include <string>
#include "colours.h"
/* #include "ADQILYAAPI_x64.h" */
/* #include "DLL_Imperium.h" */
/* #include <fstream> */
/* #include "fftw3.h" */
/* #include <ctime> */

static unsigned char FETCH_CHANNEL_A = 0x1;
static unsigned char FETCH_CHANNEL_B = 0x2;
static unsigned char FETCH_CHANNEL_BOTH = 0x3;

///////////////////////////////////////////////////////////////////////////////
// incoherent
///////////////////////////////////////////////////////////////////////////////
void ilya_incoherent_process_THREAD(
        short* chA_data, short* chB_data, unsigned int* sq_data,
        short chA_back, short chB_back,
        unsigned int samples_per_record,
        int rangeMin, int rangeMax) {
        /*
          Data processed for incoherent measurements:
          1) Take away background
          2) Compute squares
        */
        for (int i(rangeMin); i < rangeMax; i++) {
                chA_data[i] -= chA_back;
                chB_data[i] -= chB_back;
                sq_data[i] = chA_data[i] * chA_data[i] + chB_data[i] * chB_data[i];
        }
}

void ilya_incoherent_process(
        short* chA_data, short* chB_data, unsigned int* sq_data,
        short chA_back, short chB_back,
        unsigned int samples_per_record, unsigned int number_of_records
        , int number_of_threads) {
        /*
          Data processed for incoherent measurements:
          1) Take away background
          2) Compute squares
        */

        //call threads to perform computation on different parts of the array
        //const int number_of_threads = 200;		std::thread t[number_of_threads];
        //const int number_of_threads = 10; //optimised
        std::thread* t = new std::thread[number_of_threads];
        int lastValue = 0;
        int increment = floor((samples_per_record *  number_of_records) / number_of_threads);

        for (int i(0); i < number_of_threads - 1; i++) {
                t[i] = std::thread(ilya_incoherent_process_THREAD,
                                   chA_data, chB_data, sq_data,
                                   chA_back, chB_back,
                                   samples_per_record,
                                   lastValue, lastValue + increment);
                lastValue += increment;
        }
        t[number_of_threads - 1] = std::thread(ilya_incoherent_process_THREAD,
                                               chA_data, chB_data, sq_data,
                                               chA_back, chB_back,
                                               samples_per_record,
                                               lastValue, samples_per_record *  number_of_records);
        //join the threads
        for (int i(0); i < number_of_threads; i++)
                t[i].join();

        delete[] t;
}

void ilya_incoherent_accumulate_THREAD(
        short* chA_data, short* chB_data, unsigned int* square_data,
        long long* chA_cumulative, long long* chB_cumulative, unsigned long long* sq_cumulative,
        int sampleMin, int sampleMax) {
        /*
          Adds **_data to the cumulative arrays
        */
        long long a(0), b(0), sq(0);
        for (int i(sampleMin); i < sampleMax; i++) {
                chA_cumulative[i] += chA_data[i];
                chB_cumulative[i] += chB_data[i];
                sq_cumulative[i] += square_data[i];
        }
}

void ilya_incoherent_accumulate(
        short* chA_data, short* chB_data, unsigned int* square_data,
        long long* chA_cumulative, long long* chB_cumulative, unsigned long long* sq_cumulative,
        unsigned int samples_per_record, unsigned int number_of_records, int number_of_threads) {
        /*
          Adds **_data to the cumulative arrays
        */
        //const int number_of_threads = 8; is optimised		std::thread t[number_of_threads];
        std::thread* t = new std::thread[number_of_threads];
        int lastValue = 0;						int increment = floor((samples_per_record*number_of_records) / number_of_threads);

        /////////////////////////////////////////////////////////////////////////////////////////////////
        //1) Using multiple threads add this data to the accumulated arrays
        /////////////////////////////////////////////////////////////////////////////////////////////////
        for (int i(0); i < number_of_threads - 1; i++) {
                t[i] = std::thread(ilya_incoherent_accumulate_THREAD, chA_data, chB_data, square_data,
                                   chA_cumulative, chB_cumulative, sq_cumulative, lastValue, lastValue + increment);
                lastValue += increment;
        }
        /////////////////////////////////////////////////////////////////////////////////////////////////
        //2) add on the final array
        /////////////////////////////////////////////////////////////////////////////////////////////////
        t[number_of_threads - 1] = std::thread(ilya_incoherent_accumulate_THREAD, chA_data, chB_data, square_data,
                                               chA_cumulative, chB_cumulative, sq_cumulative, lastValue, samples_per_record*number_of_records);

        /////////////////////////////////////////////////////////////////////////////////////////////////
        //3) collect the multiple threads together
        /////////////////////////////////////////////////////////////////////////////////////////////////
        for (int i(0); i < number_of_threads; i++)
                t[i].join();
        delete[] t;
}

void ilya_incoherent_average_THREAD(long long* chA_cumulative, long long* chB_cumulative, unsigned long long* sq_cumulative,
                                    double* average_A, double* average_B, double* average_sq,
                                    unsigned int number_of_records, unsigned int samples_per_record, int sampleMin, int sampleMax,
                                    unsigned int number_of_repetitions) {
        /*
          Data from the channels (***_data) has a linear set of readings for multiple records: r1(s1s2s3...)....r2(s1s2s3...)....
          This method finds the average value at each sample point e.g. average(s1) = s1(r1+r2+r3+....)

          Works with long long pieces of data and normalises by the
        */

        long long chA_cumulative_temp(0);	long long chB_cumulative_temp(0);	long long sq_cumulative_temp(0);
        for (int i(sampleMin); i < sampleMax; i++) {
                for (int j(0); j < number_of_records; j++) {
                        //we have three arrays for which we need to sum up down each column
                        chA_cumulative_temp += chA_cumulative[j * samples_per_record + i];
                        chB_cumulative_temp += chB_cumulative[j * samples_per_record + i];
                        sq_cumulative_temp += sq_cumulative[j * samples_per_record + i];
                }
                //compute average for every sample point
                average_A[i] = double(chA_cumulative_temp) / (number_of_records * number_of_repetitions);
                average_B[i] = double(chB_cumulative_temp) / (number_of_records * number_of_repetitions);
                average_sq[i] = double(sq_cumulative_temp) / (number_of_records * number_of_repetitions);
                //zero arrays and repeat
                chA_cumulative_temp = 0;
                chB_cumulative_temp = 0;
                sq_cumulative_temp = 0;
        }
}

void ilya_incoherent_average(long long* chA_cumulative, long long* chB_cumulative, unsigned long long* sq_cumulative,
                             double* average_A, double* average_B, double* average_sq,
                             unsigned int number_of_records, unsigned int samples_per_record, unsigned int number_of_repetitions) {
        /*
          Data from the channels(***_data) has a linear set of readings for multiple records : r1(s1s2s3...)....r2(s1s2s3...)....
          This method finds the average value at each sample point e.g.average(s1) = s1(r1 + r2 + r3 + ....)

          Works with long long pieces of data and normalises by the
        */
        const int number_of_threads = 8;		std::thread t[number_of_threads];
        int lastValue = 0;						int increment = floor((samples_per_record) / number_of_threads);

        //launch mutltiple threads to do the averaging
        for (int i(0); i < number_of_threads - 1; i++) {
                t[i] = std::thread(ilya_incoherent_average_THREAD, chA_cumulative, chB_cumulative, sq_cumulative,
                                   average_A, average_B, average_sq, number_of_records, samples_per_record, lastValue, lastValue + increment, number_of_repetitions);
                lastValue += increment;
        }
        t[number_of_threads - 1] = std::thread(ilya_incoherent_average_THREAD, chA_cumulative, chB_cumulative, sq_cumulative,
                                               average_A, average_B, average_sq, number_of_records, samples_per_record, lastValue, samples_per_record, number_of_repetitions);
        //collect the multiple threads together
        for (int i(0); i < number_of_threads; i++)
                t[i].join();
}

void ilya_incoherent(void* adq_cu_ptr,
                     double* chA_out, double* chB_out, double* sq_out,
                     short chA_background, short chB_background,
                     unsigned int samples_per_record, unsigned int number_of_records,
                     unsigned int number_of_repetitions)
{	/*
      C = cumulative
      P = Parallel
      B = Background
      - Reads out channel A and B data and adds it to the **_cum arrays with the background taken away
      - Evalaute the square for each point and at it to the sq_cum array
      Repeat this for the specified number of repititions
    */

    //arrays to read the data to during each run. We alternative between reading into x and y
        short* channelA_dataX = new short[samples_per_record*number_of_records];
        short* channelB_dataX = new short[samples_per_record*number_of_records];
        unsigned int* square_dataX = new unsigned int[samples_per_record*number_of_records]();
        short* channelA_dataY = new short[samples_per_record*number_of_records];
        short* channelB_dataY = new short[samples_per_record*number_of_records];
        unsigned int* square_dataY = new unsigned int[samples_per_record*number_of_records];
        short* channelA_dataZ = new short[samples_per_record*number_of_records]();
        short* channelB_dataZ = new short[samples_per_record*number_of_records]();
        unsigned int* square_dataZ = new unsigned int[samples_per_record*number_of_records]();


        //arrays to store the accumulated data
        long long* chA_cumulative = new long long[samples_per_record*number_of_records]();
        long long* chB_cumulative = new long long[samples_per_record*number_of_records]();
        unsigned long long* sq_cumulative = new unsigned long long[samples_per_record*number_of_records]();

        //parameters for launching thread processes
        /*const int number_of_threads = 200;
          int number_of_threads_for_arg = number_of_threads;
          std::thread t[number_of_threads];
          int lastValue = 0;							int increment = floor((samples_per_record*number_of_records) / number_of_threads);
        */
        std::thread x, y, z;

        /////////////////////////////////////////////////////////////////////////////////////////////////
        //1) Setup multirecord mode and run first set of data
        /////////////////////////////////////////////////////////////////////////////////////////////////
        ADQ214_MultiRecordSetup(adq_cu_ptr, 1, number_of_records, samples_per_record);
        ilya_CPB_collect_data(adq_cu_ptr, channelA_dataX, channelB_dataX, samples_per_record, number_of_records);

        /////////////////////////////////////////////////////////////////////////////////////////////////
        //2) launch parallel threads to read and store the data
        /////////////////////////////////////////////////////////////////////////////////////////////////
        int flipFlop(0);
        for (int run(0); run <= number_of_repetitions; run++) {
                if (flipFlop == 0) {
                        //read Y, process X, store Z
                        y = std::thread(ilya_CPB_collect_data, adq_cu_ptr, channelA_dataY, channelB_dataY, samples_per_record, number_of_records);
                        x = std::thread(ilya_incoherent_process, channelA_dataX, channelB_dataX, square_dataX,
                                        chA_background, chB_background,
                                        samples_per_record, number_of_records,
                                        3);
                        z = std::thread(ilya_incoherent_accumulate, channelA_dataZ, channelB_dataZ, square_dataZ,
                                        chA_cumulative, chB_cumulative, sq_cumulative,
                                        samples_per_record, number_of_records, 3);
                        x.join();
                        y.join();
                        z.join();
                        flipFlop = 1;
                }
                else if (flipFlop == 1) {
                        //read Z, process Y, store X
                        z = std::thread(ilya_CPB_collect_data, adq_cu_ptr, channelA_dataZ, channelB_dataZ, samples_per_record, number_of_records);
                        y = std::thread(ilya_incoherent_process, channelA_dataY, channelB_dataY, square_dataY,
                                        chA_background, chB_background,
                                        samples_per_record, number_of_records, 3);
                        x = std::thread(ilya_incoherent_accumulate, channelA_dataX, channelB_dataX, square_dataX,
                                        chA_cumulative, chB_cumulative, sq_cumulative,
                                        samples_per_record, number_of_records, 3);
                        x.join();
                        y.join();
                        z.join();
                        flipFlop = 2;
                }
                else {
                        //read X, process Z, store Y
                        x = std::thread(ilya_CPB_collect_data, adq_cu_ptr, channelA_dataX, channelB_dataX, samples_per_record, number_of_records);
                        z = std::thread(ilya_incoherent_process, channelA_dataZ, channelB_dataZ, square_dataZ,
                                        chA_background, chB_background,
                                        samples_per_record, number_of_records, 3);
                        y = std::thread(ilya_incoherent_accumulate, channelA_dataY, channelB_dataY, square_dataY,
                                        chA_cumulative, chB_cumulative, sq_cumulative,
                                        samples_per_record, number_of_records, 3);
                        x.join();
                        y.join();
                        z.join();
                        flipFlop = 0;
                }
        }

        /////////////////////////////////////////////////////////////////////////////////////////////////
        //3) Now that all of the data has been accumulated, we can compute averages over the columns
        /////////////////////////////////////////////////////////////////////////////////////////////////
        ilya_incoherent_average(chA_cumulative, chB_cumulative, sq_cumulative,
                                chA_out, chB_out, sq_out, number_of_records, samples_per_record, number_of_repetitions);

        delete[] channelA_dataX;
        delete[] channelB_dataX;
        delete[] square_dataX;
        delete[] channelA_dataY;
        delete[] channelB_dataY;
        delete[] square_dataY;
        delete[] channelA_dataZ;
        delete[] channelB_dataZ;
        delete[] square_dataZ;
        delete[] chA_cumulative;
        delete[] chB_cumulative;
        delete[] sq_cumulative;
}

void ilya_incoherent_python(void* adq_cu_ptr, double* chA_out, double* chB_out, double* sq_out,
                            short chA_background, short chB_background,
                            unsigned int samples_per_record, unsigned int number_of_records,
                            long long* chA_cumulative, long long *chB_cumulative, unsigned long long* sq_cumulative,
                            int repetition_number) {
        /*
          Has the same functionality as ilya_CPB, except that there are no repitions or parrallel threads

          !!!!!!!!!! Multirecord must be set up !!!!!!!!!!
        */
        short* chA_data = new short[samples_per_record*number_of_records];
        short* chB_data = new short[samples_per_record*number_of_records];
        unsigned int* sq_data = new unsigned int[samples_per_record*number_of_records]();


        //1) collect the data
        ilya_CPB_collect_data(adq_cu_ptr, chA_data, chB_data, samples_per_record, number_of_records);

        //2) process the data
        ilya_incoherent_process(chA_data, chB_data, sq_data, chA_background, chB_background, samples_per_record, number_of_records, 8);

        //3) store data
        ilya_incoherent_accumulate(chA_data, chB_data, sq_data, chA_cumulative, chB_cumulative, sq_cumulative, samples_per_record, number_of_records, 8);

        //4) compute squares
        ilya_incoherent_average(chA_cumulative, chB_cumulative, sq_cumulative, chA_out, chB_out, sq_out, number_of_records, samples_per_record, repetition_number);

        delete[] chA_data;
        delete[] chB_data;
        delete[] sq_data;
}


///////////////////////////////////////////////////////////////////////////////
// correlation
///////////////////////////////////////////////////////////////////////////////
void ilya_correlation_process_THREAD(
        short* chA_data, short* chB_data, unsigned int* sq_data,
        short chA_back, short chB_back,
        long long* chA_av_cum, long long* chB_av_cum, long long* sq_av_cum,
        unsigned int samples_per_record,
        int thread,
        int rangeMin, int rangeMax) {
        /*
          1) Normalise input values by background;
          2) Compute the squares
          3?) Store the sum for computing the average later
        */
        long long a(0), b(0), sq(0);

        for (int i(rangeMin); i < rangeMax; i++) {
                chA_data[i] -= chA_back;
                chB_data[i] -= chB_back;
                sq_data[i] = chA_data[i] * chA_data[i] + chB_data[i] * chB_data[i];
                a += chA_data[i];
                b += chB_data[i];
                sq += sq_data[i];
        }
        chA_av_cum[thread] = a;
        chB_av_cum[thread] = b;
        sq_av_cum[thread] = sq;
}

void ilya_correlation_process(
        short* chA_data, short* chB_data, unsigned int* sq_data,
        short chA_back, short chB_back,
        double& chA_av, double& chB_av, double& sq_av,
        unsigned int samples_per_record
        , int number_of_threads
        ) {
        /*
          1) Normalise input values by background;
          2) Compute the squares
          3?) Compute average
        */
        long long* chA_av_cum = new long long[number_of_threads];
        long long* chB_av_cum = new long long[number_of_threads];
        long long* sq_av_cum = new long long[number_of_threads];

        //1) call threads to perform computation on different parts of the array
        std::thread* t = new std::thread[number_of_threads];	//10 is optimised
        int lastValue = 0;						int increment = floor(samples_per_record / number_of_threads);

        for (int i(0); i < number_of_threads - 1; i++) {
                t[i] = std::thread(ilya_correlation_process_THREAD,
                                   chA_data, chB_data, sq_data,
                                   chA_back, chB_back,
                                   chA_av_cum, chB_av_cum, sq_av_cum,
                                   samples_per_record,
                                   i,
                                   lastValue, lastValue + increment);
                lastValue += increment;
        }
        t[number_of_threads - 1] = std::thread(ilya_correlation_process_THREAD,
                                               chA_data, chB_data, sq_data,
                                               chA_back, chB_back,
                                               chA_av_cum, chB_av_cum, sq_av_cum,
                                               samples_per_record,
                                               number_of_threads - 1,
                                               lastValue, samples_per_record);
        //2) join the threads and assemble averaged values
        for (int i(0); i < number_of_threads; i++) {
                t[i].join();
                chA_av += (double)chA_av_cum[i];
                chB_av += (double)chB_av_cum[i];
                sq_av += (double)sq_av_cum[i];
        }

        chA_av /= (samples_per_record);
        chB_av /= (samples_per_record);
        sq_av /=  (samples_per_record);
        delete[] t;
        delete[] chA_av_cum;
        delete[] chB_av_cum;
        delete[] sq_av_cum;
}

void ilya_correlation_g1_THREAD(
        short* chA_data, short* chB_data, unsigned int* sq_data,
        double* chA_g1, double* chB_g1, double* sq_g1,
        double chA_av, double chB_av, double sq_av,
        unsigned int samples_per_record,
        int min, int max) {
        /*
          Given a set of consecutive points, compute the correlation G(T) = <X(t)X(t+T)>
          Mupltiple records are placed in a single line, and tau values are taken across them.
        */

        /*int number_of_points = number_of_records * samples_per_record - tau_value;*/
        long long chA_cum(0), chB_cum(0), sq_cum(0);

        for (int tau(min); tau < max; tau++) {
                for (int i(0); i < samples_per_record - tau; i++) {
                        //iterate over all of the data points for computing the set tau value for each array
                        //chA_cum += ((long long)chA_data[i] - chA_av) * ((long long)chA_data[i + tau] - chA_av);
                        chA_cum += (long long)chA_data[i] * (long long)chA_data[i + tau];
                        //chB_cum += (chB_data[i] - chB_av) * (chB_data[i + tau] - chB_av);
                        //sq_cum += ((double)sq_data[i] - sq_av) * ((double)sq_data[i + tau] - sq_av);
                        sq_cum += (long long(sq_data[i]) - (long long)sq_av) * (long long(sq_data[i + tau]) - (long long)sq_av);
                }
                chA_g1[tau] = double(chA_cum) / (samples_per_record - tau) - chA_av * chA_av;
                //chB_g1[tau] = double(chB_cum) / (number_of_records * samples_per_record - tau);
                sq_g1[tau] = double(sq_cum) / (samples_per_record - tau);
                chA_cum = 0;
                chB_cum = 0;
                sq_cum = 0;
        }
}

void ilya_correlation_g1(
        short* chA_data, short* chB_data, unsigned int* sq_data,
        double* chA_g1, double* chB_g1, double* sq_g1,
        double chA_av, double chB_av, double sq_av,
        int tauMax,
        unsigned int samples_per_record,
        int number_of_threads) {
        /*
          Given a set of consecutive points, compute the correlation G(T) = <X(t)X(t+T)>
          Mupltiple records are placed in a single line, and tau values are taken across them.
        */

        std::thread* t = new std::thread[number_of_threads];
        int lastValue = 10;			int increment = floor((tauMax) / number_of_threads);

        //launch mutltiple threads
        for (int i(0); i < number_of_threads - 1; i++) {
                t[i] = std::thread(ilya_correlation_g1_THREAD,
                                   chA_data, chB_data, sq_data,
                                   chA_g1, chB_g1, sq_g1,
                                   chA_av, chB_av, sq_av,
                                   samples_per_record,
                                   lastValue, lastValue + increment);
                lastValue += increment;
        }
        t[number_of_threads - 1] = std::thread(ilya_correlation_g1_THREAD,
                                               chA_data, chB_data, sq_data,
                                               chA_g1, chB_g1, sq_g1,
                                               chA_av, chB_av, sq_av,
                                               samples_per_record,
                                               lastValue, tauMax);

        //collect the multiple threads together
        for (int i(0); i < number_of_threads; i++)
                t[i].join();

        delete[] t;
}

void ilya_CPB_PYTHON_correlation(void* adq_cu_ptr,
                                 short chA_background, short chB_background,
                                 unsigned int samples_per_record,
                                 double* chA_g1_cumlative, double* chB_g1_cumulative, double* sq_g1_cumulative,
                                 int tauMax,
                                 int repetition_number) {

        /*
          Single record is used to read all of the data and perform correlation.
        */
        short* chA_data = new short[samples_per_record]();
        short* chB_data = new short[samples_per_record]();
        unsigned int* sq_data = new unsigned int[samples_per_record]();
        double* chA_g1 = new double[samples_per_record]();
        double* chB_g1 = new double[samples_per_record]();
        double* sq_g1 = new double[samples_per_record]();
        double chA_average(-1), chB_average(-1), sq_average(-1);

        //1) collect the data
        ilya_CPB_collect_data(adq_cu_ptr, chA_data, chB_data, samples_per_record, 1);

        //2) process the data
        ilya_correlation_process(chA_data, chB_data, sq_data, chA_background, chB_background, chA_average, chB_average, sq_average, samples_per_record, 10);

        //3) evaluate the correlation
        ilya_correlation_g1(chA_data, chB_data, sq_data, chA_g1, chB_g1, sq_g1, chA_average, chB_average, sq_average, tauMax, samples_per_record, 10);
        for (int i(0); i < tauMax; i++) {
                chA_g1_cumlative[i] = (chA_g1_cumlative[i] * (repetition_number - 1) + chA_g1[i]) / (repetition_number);
                chB_g1_cumulative[i] = (chB_g1_cumulative[i] * (repetition_number - 1) + chB_g1[i]) / (repetition_number);
                sq_g1_cumulative[i] = (sq_g1_cumulative[i] * (repetition_number - 1) + sq_g1[i]) / (repetition_number);
        }

        //chA_g1_cumlative[0] = chA_average;
        //chA_g1_cumlative[1] = chB_average;
        //chA_g1_cumlative[2] = sq_average;

        delete[] chA_data;
        delete[] chB_data;
        delete[] sq_data;
        delete[] chA_g1;
        delete[] chB_g1;
        delete[] sq_g1;
}

void ilya_correlation(void* adq_cu_ptr,
                      short chA_background, short chB_background,
                      unsigned int samples_per_record,
                      double* chA_g1_cumlative, double* chB_g1_cumulative, double* sq_g1_cumulative,
                      int tauMax,
                      int number_of_repetitions) {
        /*
          Method is fully executed in C++.
          Reading in the full 64M samples, the correlation is
        */
}

///////////////////////////////////////////////////////////////////////////////
// correlation with the FFT
///////////////////////////////////////////////////////////////////////////////
#define REAL 0
#define IMAG 1

void ilya_g1FFT_process_THREAD(
        short* chA_data, short* chB_data,
        short chA_back, short chB_back,
        double* chA_processed, double* chB_processed, double* sq_processed,
        unsigned int samples_per_record,
        int thread,
        int rangeMin, int rangeMax) {
        /*
          1) Normalise input values by background;
          2) Compute the squares
        */
        short chA_temp, chB_temp;

        for (int i(rangeMin); i < rangeMax; i++) {
                chA_temp = chA_data[i] - chA_back;
                chB_temp = chB_data[i] - chB_back;
                chA_processed[i] = double(chA_temp);
                chB_processed[i] = double(chB_temp);
                sq_processed[i] = chA_temp*chA_temp + chB_temp*chB_temp;
        }
}

void ilya_g1FFT_process(
        short* chA_data, short* chB_data,
        short chA_back, short chB_back,
        double* chA_processed, double* chB_processed, double* sq_processed,
        unsigned int samples_per_record
        , int number_of_threads
        ) {
        /*
          1) Normalise input values by background;
          2) Compute the squares

          Note that *_processed arrays must be allocated
        */

        //1) call threads to perform computation on different parts of the array
        std::thread* t = new std::thread[number_of_threads];	//8 is optimised
        int lastValue = 0;						int increment = floor(samples_per_record / number_of_threads);

        for (int i(0); i < number_of_threads - 1; i++) {
                t[i] = std::thread(ilya_g1FFT_process_THREAD,
                                   chA_data, chB_data,
                                   chA_back, chB_back,
                                   chA_processed, chB_processed, sq_processed,
                                   samples_per_record,
                                   i,
                                   lastValue, lastValue + increment);
                lastValue += increment;
        }
        t[number_of_threads - 1] = std::thread(ilya_g1FFT_process_THREAD,
                                               chA_data, chB_data,
                                               chA_back, chB_back,
                                               chA_processed, chB_processed, sq_processed,
                                               samples_per_record,
                                               number_of_threads - 1,
                                               lastValue, samples_per_record);
        //2) join the threads
        for (int i(0); i < number_of_threads; i++)
                t[i].join();

        delete[] t;
}

void ilya_g1FFT_square(
        fftw_complex* _fourier_transform,
        unsigned int samples_per_record
        ) {
        /*
          For the second part of the procedure, the fourier-transformed values are squared
        */
        for (int i(0); i < samples_per_record/2 - 1; i++) {
                _fourier_transform[i][REAL] = _fourier_transform[i][REAL] * _fourier_transform[i][REAL] + _fourier_transform[i][IMAG] * _fourier_transform[i][IMAG];
                _fourier_transform[i][IMAG] = 0;
        }
}

void ilya_g1FFT_correlation(
        double* chA_data_formatted,
        double* chB_data_formatted,
        float* chA_correlation,
        float* chB_correlation,
        fftw_complex* supporting_complex_array,
        fftw_plan plan_FORWARD,
        fftw_plan plan_BACKWARD,
        int size_of_transform
        ){	/*
              Method performs correlation evaluation using 8 threads (optimal for an 8 core computer) and the Wiener theorem
            */
        // run the forward transform -> square -> backward transform
        fftw_execute_dft_r2c(plan_FORWARD, chA_data_formatted, supporting_complex_array);
        ilya_g1FFT_square(supporting_complex_array, size_of_transform);
        fftw_execute_dft_c2r(plan_BACKWARD, supporting_complex_array, chA_data_formatted);

        fftw_execute_dft_r2c(plan_FORWARD, chB_data_formatted, supporting_complex_array);
        ilya_g1FFT_square(supporting_complex_array, size_of_transform);
        fftw_execute_dft_c2r(plan_BACKWARD, supporting_complex_array, chB_data_formatted);

        //we need to divide twice by N, since we square the FT
        for (int i(0); i < size_of_transform; i++) {
                chA_correlation[i] = float(chA_data_formatted[i])/(size_of_transform*size_of_transform);
                chB_correlation[i] = float(chB_data_formatted[i])/(size_of_transform*size_of_transform);
        }
}

void ilya_g1FFT_prepareData(
        short* chA_data,
        short* chB_data,
        double* chA_data_formatted,
        double* chB_data_formatted,
        int size_of_transform
        ) {
        /*
          This arranges the data in an array that the FFTW will be able to process
        */
        for (int i(0); i < size_of_transform; i++) {
                chA_data_formatted[i] = chA_data[i];
                chB_data_formatted[i] = chB_data[i];
        }
}

void testMeman(void* adq_cu_ptr) {
        short* chA_data = new short[400];
        short* chB_data = new short[400];
        ilya_CPB_collect_data(adq_cu_ptr, chA_data, chB_data, 400, 1);
        delete[] chA_data;
        delete[] chB_data;
}

void ilya_g1FFT(void* adq_cu_ptr,
                unsigned int size_of_transform,
                int time_limit,
                float* chA_correlation,
                float* chB_correlation
        ) {

        /*
          Single record is used to read all of the data and perform correlation.
        */

        short* chA_data = new short[size_of_transform]();
        short* chB_data = new short[size_of_transform]();
        double chA_average(-1), chB_average(-1), sq_average(-1);
        double* chA_data_formatted = (double*)fftw_malloc(sizeof(double) * size_of_transform);
        double* chB_data_formatted = (double*)fftw_malloc(sizeof(double) * size_of_transform);
        fftw_complex* dummyArray = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * (int(size_of_transform / 2) + 1));

        fftw_init_threads();
        fftw_plan_with_nthreads(8);
        fftw_set_timelimit(int(time_limit));

        ////3) Load or generatre the plans from wisdom files (for Forwards and backwards FT)
        if (time_limit == 0)	fftw_import_wisdom_from_filename("fftw_optimised_real_to_complex_RW.wis");
        fftw_plan plan_FORWARD = fftw_plan_dft_r2c_1d(size_of_transform, chA_data_formatted, dummyArray, FFTW_EXHAUSTIVE);
        fftw_forget_wisdom();
        if (time_limit == 0)	fftw_import_wisdom_from_filename("fftw_optimised_complex_to_real_RW.wis");
        fftw_plan plan_BACKWARD = fftw_plan_dft_c2r_1d(size_of_transform, dummyArray, chA_data_formatted, FFTW_EXHAUSTIVE);
        fftw_forget_wisdom();
        ilya_CPB_collect_data(adq_cu_ptr, chA_data, chB_data, 400, 1);

        //1) collect the data
        ilya_CPB_collect_data(adq_cu_ptr, chA_data, chB_data, size_of_transform, 1);

        //2) prepare the data into format for the FFT
        ilya_g1FFT_prepareData(chA_data, chB_data, chA_data_formatted, chB_data_formatted, size_of_transform);

        //3) correlation part
        ilya_g1FFT_correlation(chA_data_formatted, chB_data_formatted, chA_correlation, chB_correlation, dummyArray, plan_FORWARD, plan_BACKWARD, size_of_transform);

        delete[] chA_data;
        delete[] chB_data;
        fftw_destroy_plan(plan_FORWARD);
        fftw_destroy_plan(plan_BACKWARD);
        fftw_free(dummyArray); fftw_free(chA_data_formatted); fftw_free(chB_data_formatted);
        fftw_cleanup_threads();
        fftw_cleanup();
}

int ilya_g1FFT_optimise(int time_limit, int number_of_samples){
        /*
          Method creates two wisdom files that optimise the forwards and backwards transforms used in g1FFT functions.

          time_limit:			time in second to do the evaluation for
          number_of_samples:	size of the FFT
        */

        //1) setup
        fftw_init_threads();						//instruct to use threads
        fftw_plan_with_nthreads(8);					//define number of threads to use. For an 8 core system, 8 is best
        fftw_set_timelimit(time_limit);			//set time for evalution
        int return_success = -2;

        //2) allocating arrays
        double* x = (double*)fftw_malloc(sizeof(double) * number_of_samples);
        fftw_complex* y = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * (int(number_of_samples / 2) + 1));

        //3) optimising and writing into ReWrite files
        fftw_plan planF = fftw_plan_dft_r2c_1d(number_of_samples, x, y, FFTW_EXHAUSTIVE);
        return_success += fftw_export_wisdom_to_filename("fftw_optimised_real_to_complex_RW.wis");
        fftw_forget_wisdom();
        fftw_plan planB = fftw_plan_dft_c2r_1d(number_of_samples, y, x, FFTW_EXHAUSTIVE);
        return_success += fftw_export_wisdom_to_filename("fftw_optimised_complex_to_real_RW.wis");
        fftw_forget_wisdom();

        // cleanup
        fftw_destroy_plan(planF); fftw_destroy_plan(planB);
        fftw_cleanup();
        fftw_free(x); fftw_free(y);
        fftw_cleanup_threads();

        return return_success;
}

///////////////////////////////////////////////////////////////////////////////
// tests
///////////////////////////////////////////////////////////////////////////////
int ilya_g1FFT_demonstration(
        float* arrayIn,
        float* arrayOut,
        int time_limit,
        int samples_per_record,
        bool load_wisdom
        ) {
        /*
          Performs the evaluation of the forward FFT using the Wiener-Khinchin Theorem. This is a demonstration,
          used to show hwo the internals function

          arrayIn:			array of the raw unprocessed data
          arraOut:			array of the correlated values
          time_limit:			0 for immediate processing
          number_of_samples:	data points to process
        */

        int success = -2;
        //1) Allocate threads for the FFTW procedures
        fftw_init_threads();
        fftw_plan_with_nthreads(8);
        fftw_set_timelimit(time_limit);

        //2) Allocate the array for storing the transform. Dummmy is for plan configuration
        double* x = (double*)fftw_malloc(sizeof(double) * samples_per_record);
        fftw_complex* y = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * (int(samples_per_record / 2) + 1));
        //double* z = (double*)fftw_malloc(sizeof(double) * samples_per_record);

        ////3) Load the plans from wisdom files (for Forwards and backwards FT)
        fftw_forget_wisdom();
        if (load_wisdom)	success += fftw_import_wisdom_from_filename("fftw_optimised_real_to_complex_RW.wis");
        fftw_plan planF = fftw_plan_dft_r2c_1d(samples_per_record, x, y, FFTW_EXHAUSTIVE);

        fftw_forget_wisdom();
        //fftw_set_timelimit(int(time_limit));
        if (load_wisdom) success += fftw_import_wisdom_from_filename("fftw_optimised_complex_to_real_RW.wis");
        fftw_plan planB = fftw_plan_dft_c2r_1d(samples_per_record, y, x, FFTW_EXHAUSTIVE);
        fftw_forget_wisdom();

        //4-5) load the data
        for (int i(0); i < samples_per_record; i++)
                x[i] = arrayIn[i];

        //4) run the forward transform -> square -> backward transform
        fftw_execute_dft_r2c(planF, x, y);
        ilya_g1FFT_square(y, samples_per_record);
        fftw_execute_dft_c2r(planB, y, x);
        for (int i(0); i < samples_per_record; i++)
                x[i] /= (samples_per_record*samples_per_record);	//we need to divide twice by N, since we square the FT

        for (int i(0); i < samples_per_record; i++)
                arrayOut[i] = x[i];

        //// cleanup
        fftw_destroy_plan(planF);
        fftw_destroy_plan(planB);
        fftw_free(y); fftw_free(x);// fftw_free(x);
        fftw_cleanup_threads();
        fftw_cleanup();

        return success;
}

int ilya_FFT_forward_backward_demonstration(
        float* arrayIn,
        float* arrayOut,
        int time_limit,
        int samples_per_record,
        bool load_wisdom
        ) {
        /*
          Performs the evaluation of the forward FFT using the Wiener-Khinchin Theorem. This is a demonstration,
          used to show hwo the internals function

          arrayIn:			array of the raw unprocessed data
          arraOut:			array of the correlated values
          time_limit:			0 for immediate processing
          number_of_samples:	data points to process
        */

        int success = -2;
        //1) Allocate threads for the FFTW procedures
        fftw_init_threads();
        fftw_plan_with_nthreads(8);
        fftw_set_timelimit(time_limit);

        //2) Allocate the array for storing the transform. Dummmy is for plan configuration
        double* x = (double*)fftw_malloc(sizeof(double) * samples_per_record);
        fftw_complex* y = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * (int(samples_per_record / 2) + 1));

        ////3) Load the plans from wisdom files (for Forwards and backwards FT)
        fftw_forget_wisdom();
        if (load_wisdom)	success += fftw_import_wisdom_from_filename("fftw_optimised_real_to_complex_56448000.wis");
        fftw_plan planF = fftw_plan_dft_r2c_1d(samples_per_record, x, y, FFTW_EXHAUSTIVE);

        fftw_forget_wisdom();
        //fftw_set_timelimit(int(time_limit));
        if (load_wisdom) success += fftw_import_wisdom_from_filename("fftw_optimised_complex_to_real_56448000.wis");
        fftw_plan planB = fftw_plan_dft_c2r_1d(samples_per_record, y, x, FFTW_EXHAUSTIVE);
        fftw_forget_wisdom();

        //4-5) load the data
        for (int i(0); i < samples_per_record; i++)
                x[i] = arrayIn[i];

        //4) run the forward transform -> square -> backward transform
        fftw_execute_dft_r2c(planF, x, y);
        fftw_execute_dft_c2r(planB, y, x);
        for (int i(0); i < samples_per_record; i++)
                x[i] /= (samples_per_record);

        for (int i(0); i < samples_per_record; i++)
                arrayOut[i] = x[i];

        //// cleanup
        fftw_destroy_plan(planF);
        fftw_destroy_plan(planB);
        fftw_free(y); fftw_free(x);
        fftw_cleanup_threads();
        fftw_cleanup();

        return success;
}

void TEST_ilya_CPB_collect_data(void* adq_cu_ptr,
                                unsigned int samples_per_record, unsigned int number_of_records, int repetitions) {
        /*
          Rerun the collect data method for speed testing
        */
        //1) setup multirecord
        short* buff_a = new short[number_of_records*samples_per_record];
        short* buff_b = new short[number_of_records*samples_per_record];

        ADQ214_MultiRecordSetup(adq_cu_ptr, 1, number_of_records, samples_per_record);
        for (int i(0); i < repetitions; i++)
                ilya_CPB_collect_data(adq_cu_ptr, buff_a, buff_b, samples_per_record, number_of_records);

        delete[] buff_a;
        delete[] buff_b;
}

void TEST_ilya_incoherent_process(unsigned int samples_per_record, unsigned int number_of_records, int repetitions,
                                  int number_of_threads
        ) {
        /*
          Rerun process data for speed testing
        */
        short* chA_data = new short[number_of_records*samples_per_record];
        short* chB_data = new short[number_of_records*samples_per_record];
        short chA_back = 0;
        short chB_back = 0;
        unsigned int* sq_data = new unsigned int[number_of_records*samples_per_record];

        for (int i(0); i < samples_per_record*number_of_records; i++) {
                chA_data[i] = i;
                chB_data[i] = i;
        }

        for (int i(0); i < repetitions; i++) {
                ilya_incoherent_process(chA_data, chB_data, sq_data, chA_back, chB_back, samples_per_record, number_of_records, number_of_threads);
                //ilya_CPB_process_data(chA_data, chB_data, sq_data, chA_back, chB_back, a, b, c, samples_per_record, number_of_records);
        }

        delete[] chA_data;
        delete[] chB_data;
        delete[] sq_data;
}

void test_passing_array(long long* chA, long long* chB, unsigned long long* sq) {
        chA[0] = 0;
        chB[0] = 0;
        sq[0] = 0;
}

float test_ilya_g1FFT_prepareData(int size_of_transform) {
        /*
          Test for the preparation of data
        */
        float success = 1;

        short* chA_data = new short[size_of_transform];
        short* chB_data = new short[size_of_transform];
        for (int i(0); i < size_of_transform; i++) {
                chA_data[i] = i;
                chB_data[i] = size_of_transform - i;
        }

        double* chA_data_formatted = (double*)fftw_malloc(sizeof(double) * size_of_transform);
        double* chB_data_formatted = (double*)fftw_malloc(sizeof(double) * size_of_transform);
        const clock_t begin_time = clock();

        ilya_g1FFT_prepareData(chA_data, chB_data, chA_data_formatted, chB_data_formatted, size_of_transform);
        success = double(clock() - begin_time) / CLOCKS_PER_SEC;

        for (int i(0); i < size_of_transform; i++) {
                if (chA_data[i] != chA_data_formatted[i])
                        success = -1;
                if (chB_data[i] != chB_data_formatted[i])
                        success = -1;
        }

        delete[] chA_data;
        delete[] chB_data;

        //delete[] chA_data_formatted;
        //delete[] chB_data_formatted;
        return success;
}

//int ilya_g1FFT(
//	float* forward,
//	float* backward,
//	int samples_per_record
//) {
//	/*
//	Performs the evaluation of the forward FFT using the Wiener-Khinchin Theorem
//	*/
//
//	int success = -1;
//	//1) Allocate threads for the FFTW procedures
//	fftw_init_threads();
//	fftw_plan_with_nthreads(8);
//	fftw_set_timelimit(20);
//
//	//2) Allocate the array for storing the transform. Dummmy is for plan configuration
//	double* x = (double*)fftw_malloc(sizeof(double) * samples_per_record);
//	double* x1 = (double*)fftw_malloc(sizeof(double) * samples_per_record);
//	fftw_complex* y = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * (int(samples_per_record / 2) + 1));
//	double* z = (double*)fftw_malloc(sizeof(double) * samples_per_record);
//
//	////3) Load the plans from wisdom files (for Forwards and backwards FT)
//	success = fftw_import_wisdom_from_filename("fftw_optimised_real_to_complex_RW.wis");
//	fftw_plan planF = fftw_plan_dft_r2c_1d(samples_per_record, x, y, FFTW_EXHAUSTIVE);
//
//	for (int i(0); i < samples_per_record; i++)
//		x1[i] = i + 1;
//	//
//	fftw_forget_wisdom();
//	success += fftw_import_wisdom_from_filename("fftw_optimised_complex_to_real_RW.wis");
//	fftw_plan planB = fftw_plan_dft_c2r_1d(samples_per_record, y, z, FFTW_EXHAUSTIVE);
//	fftw_forget_wisdom();
//
//	//4) run the forward transform -> square -> backward transform
//	fftw_execute_dft_r2c(planF, x1, y);
//	////fftw_execute(planF);
//	for (int i(0); i < samples_per_record; i++)
//		forward[i] = y[REAL][i];
//
//	//ilya_g1FFT_square(_fourier_transform, samples_per_record);
//	fftw_execute_dft_c2r(planB, y, z);
//	for (int i(0); i < samples_per_record; i++)
//		z[i] /= samples_per_record;
//
//	for (int i(0); i < samples_per_record; i++)
//		backward[i] = z[i];
//
//	//// cleanup
//	fftw_destroy_plan(planF);
//	fftw_destroy_plan(planB);
//	fftw_free(y); fftw_free(z); fftw_free(x1); fftw_free(x);
//	fftw_cleanup_threads();
//	fftw_cleanup();
//
//	return success;
//}
//void ilya_CPB_process_data_THREAD(
//	short* chA_data, short* chB_data, unsigned int* sq_data,
//	short chA_back, short chB_back,
//	long long* chA_av_cum, long long* chB_av_cum, long long* sq_av_cum,
//	int flat_average_point,
//	unsigned int samples_per_record,
//	int rangeMin, int rangeMax) {
//	/*
//	1) Normalise input values by background;
//	2) Compute the squares
//	3?) Store the sum for computing the average later
//	*/
//	long long a(0), b(0), sq(0);
//
//	for (int i(rangeMin); i < rangeMax; i++) {
//		chA_data[i] -= chA_back;
//		chB_data[i] -= chB_back;
//		sq_data[i] = chA_data[i] * chA_data[i] + chB_data[i] * chB_data[i];
//		//if (i%samples_per_record < flat_average_point) { //only taking the 'flat bits' (when there are no pulses for sure)
//		a += chA_data[i];
//		b += chB_data[i];
//		sq += sq_data[i];
//		//}
//	}
//	*chA_av_cum += a;
//	*chB_av_cum += b;
//	*sq_av_cum += sq;
//}
//
//void ilya_CPB_process_data(
//	short* chA_data, short* chB_data, unsigned int* sq_data,
//	short chA_back, short chB_back,
//	int& chA_av, int& chB_av, int& sq_av,
//	int flat_average_point,
//	unsigned int samples_per_record, unsigned int number_of_records
//	,int number_of_threads
//	) {
//	/*
//	1) Normalise input values by background;
//	2) Compute the squares
//	3?) Compute average
//	*/
//	long long chA_av_cum(0), chB_av_cum(0), sq_av_cum(0);
//
//	//call threads to perform computation on different parts of the array
//	//const int number_of_threads = 200;		std::thread t[number_of_threads];
//	//const int number_of_threads = 10; //optimised
//	std::thread* t = new std::thread[number_of_threads];
//	int lastValue = 0;						int increment = floor((samples_per_record *  number_of_records) / number_of_threads);
//
//	for (int i(0); i < number_of_threads - 1; i++) {
//		t[i] = std::thread(ilya_CPB_process_data_THREAD,
//			chA_data, chB_data, sq_data,
//			chA_back, chB_back,
//			&chA_av_cum, &chB_av_cum, &sq_av_cum,
//			flat_average_point,
//			samples_per_record,
//			lastValue, lastValue + increment);
//		lastValue += increment;
//	}
//	t[number_of_threads - 1] = std::thread(ilya_CPB_process_data_THREAD,
//		chA_data, chB_data, sq_data,
//		chA_back, chB_back,
//		&chA_av_cum, &chB_av_cum, &sq_av_cum,
//		flat_average_point,
//		samples_per_record,
//		lastValue, samples_per_record *  number_of_records);
//	//join the threads
//	for (int i(0); i < number_of_threads; i++)
//		t[i].join();
//
//	chA_av = (int)(chA_av_cum / (flat_average_point * number_of_records));
//	chB_av = (int)(chB_av_cum / (flat_average_point * number_of_records));
//	sq_av = (int)(sq_av_cum / (flat_average_point * number_of_records));
//	delete[] t;
//}
//
//void ilya_CPB_basic_accumulate_THREAD(
//	short* chA_data, short* chB_data, unsigned int* square_data,
//	long long* chA_cumulative, long long* chB_cumulative, unsigned long long* sq_cumulative,
//	int sampleMin, int sampleMax) {
//	/*
//	Adds **_data to the cumulative arrays
//	*/
//
//	for (int i(sampleMin); i < sampleMax; i++) {
//		chA_cumulative[i] += chA_data[i];//?
//		chB_cumulative[i] += chB_data[i];//?
//		sq_cumulative[i] += square_data[i];
//	}
//}
//
//void ilya_CPB_basic_accumulate(
//	short* chA_data, short* chB_data, unsigned int* square_data,
//	long long* chA_cumulative, long long* chB_cumulative, unsigned long long* sq_cumulative,
//	unsigned int samples_per_record, unsigned int number_of_records, int number_of_threads){
//	/*
//	Adds **_data to the cumulative arrays
//	*/
//	//const int number_of_threads = 10; is optimised		std::thread t[number_of_threads];
//	std::thread* t = new std::thread[number_of_threads];
//	int lastValue = 0;						int increment = floor((samples_per_record*number_of_records) / number_of_threads);
//
//	/////////////////////////////////////////////////////////////////////////////////////////////////
//	//1) Using multiple threads add this data to the accumulated arrays
//	/////////////////////////////////////////////////////////////////////////////////////////////////
//	for (int i(0); i < number_of_threads - 1; i++) {
//		t[i] = std::thread(ilya_CPB_basic_accumulate_THREAD, chA_data, chB_data, square_data,
//			chA_cumulative, chB_cumulative, sq_cumulative, lastValue, lastValue + increment);
//		lastValue += increment;
//	}
//	/////////////////////////////////////////////////////////////////////////////////////////////////
//	//2) add on the final array
//	/////////////////////////////////////////////////////////////////////////////////////////////////
//	t[number_of_threads - 1] = std::thread(ilya_CPB_basic_accumulate_THREAD, chA_data, chB_data, square_data,
//		chA_cumulative, chB_cumulative, sq_cumulative, lastValue, samples_per_record*number_of_records);
//
//	/////////////////////////////////////////////////////////////////////////////////////////////////
//	//3) collect the multiple threads together
//	/////////////////////////////////////////////////////////////////////////////////////////////////
//	for (int i(0); i < number_of_threads; i++)
//		t[i].join();
//	delete[] t;
//}
//void ilya_CPB_g1_THREAD(
//	short* chA_data, short* chB_data, unsigned int* sq_data,
//	double* chA_g1, double* chB_g1, double* sq_g1,
//	int chA_av, int chB_av, int sq_av,
//	unsigned int number_of_records, unsigned int samples_per_record,
//	int min, int max) {
//	/*
//	Given a set of consecutive points, compute the correlation G(T) = <X(t)X(t+T)>
//	Mupltiple records are placed in a single line, and tau values are taken across them.
//	*/
//
//	/*int number_of_points = number_of_records * samples_per_record - tau_value;*/
//	long long chA_cum(0), chB_cum(0), sq_cum(0);
//
//	for (int tau(min); tau < max; tau++) {
//		for (int i(0); i < number_of_records * samples_per_record - tau; i++) {
//			//iterate over all of the data points for computing the set tau value for each array
//			//chA_cum += ((long long)chA_data[i] - chA_av) * ((long long)chA_data[i + tau] - chA_av);
//			//chB_cum += (chB_data[i] - chB_av) * (chB_data[i + tau] - chB_av);
//			/*sq_cum += ((long long)sq_data[i] - sq_av) * ((long long)sq_data[i + tau] - sq_av);*/
//			/*sq_cum += ((long long)sq_data[i] - 5400) * ((long long)sq_data[i + tau] - 5400);*/
//			sq_cum += ((long long)sq_data[i]-sq_av) * ((long long)sq_data[i + tau]- sq_av);
//		}
//		//chA_g1[tau] = double(chA_cum) / (number_of_records * samples_per_record - tau);
//		//chB_g1[tau] = double(chB_cum) / (number_of_records * samples_per_record - tau);
//		sq_g1[tau] = double(sq_cum) / (number_of_records * samples_per_record - tau);
//		chA_cum = 0;
//		chB_cum = 0;
//		sq_cum = 0;
//	}
//}
//
//void ilya_CPB_g1(
//	short* chA_data, short* chB_data, unsigned int* sq_data,
//	double* chA_g1, double* chB_g1, double* sq_g1,
//	int tauMax, int chA_av, int chB_av, int sq_av,
//	unsigned int number_of_records, unsigned int samples_per_record,
//	int number_of_threads) {
//	/*
//	Given a set of consecutive points, compute the correlation G(T) = <X(t)X(t+T)>
//	Mupltiple records are placed in a single line, and tau values are taken across them.
//	*/
//
//	//long long chA_cum(0), chB_cum(0), sq_cum(0);
//
//	//for (int tau(0); tau < samples_per_record; tau++) {
//	//	for (int i(0); i < number_of_records * samples_per_record - tau; i++) {
//	//		//iterate over all of the data points for computing the set tau value for each array
//	//		chA_cum += (chA_data[i] - chA_av) * (chA_data[i + tau] - chA_av);
//	//		chB_cum += (chB_data[i] - chB_av) * (chB_data[i + tau] - chB_av);
//	//		sq_cum += ((long long)sq_data[i] - 34991) * ((long long)sq_data[i + tau] - sq_av);
//	//	}
//	//	chA_g1[tau] = double(chA_cum) / (number_of_records * samples_per_record - tau);
//	//	chB_g1[tau] = double(chB_cum) / (number_of_records * samples_per_record - tau);
//	//	sq_g1[tau] = double(sq_cum) / (number_of_records * samples_per_record - tau);
//	//	chA_cum = 0;
//	//	chB_cum = 0;
//	//	sq_cum = 0;
//	//}
//
//	// evaluate the correlators by launching threads for different values of tau
//	/*std::thread* t = new std::thread[tauMax];*/
//	std::thread* t = new std::thread[number_of_threads];
//	int lastValue = 50;			int increment = floor((300) / number_of_threads);/////////////////////////////!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!200 = number of samples
//
//	//launch mutltiple threads
//	for (int i(0); i < number_of_threads - 1; i++) {
//		t[i] = std::thread(ilya_CPB_g1_THREAD,
//			chA_data, chB_data, sq_data,
//			chA_g1, chB_g1, sq_g1,
//			chA_av, chB_av, sq_av,
//			number_of_records, samples_per_record,
//			lastValue, lastValue + increment);
//		lastValue += increment;
//	}
//	t[number_of_threads - 1] = std::thread(ilya_CPB_g1_THREAD,
//		chA_data, chB_data, sq_data,
//		chA_g1, chB_g1, sq_g1,
//		chA_av, chB_av, sq_av,
//		number_of_records, samples_per_record,
//		lastValue, 300);														/////////////////////////////!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!200 = number of samples
//
//	//collect the multiple threads together
//	for (int i(0); i < number_of_threads; i++)
//		t[i].join();
//
//	delete[] t;
//}
//void ilya_CPB_g1_THREAD(short* chA_data, short* chB_data, unsigned int* sq_data,
//	double* chA_g1, double* chB_g1, double* sq_g1,
//	unsigned int number_of_records, unsigned int samples_per_record, int tau_value) {
//	/*
//	Given a set of consecutive points, compute the correlation G(T) = <X(t)X(t+T)>
//	Mupltiple records are placed in a single line, and tau values are taken across them.
//	*/
//
//	int number_of_points = number_of_records * samples_per_record - tau_value;
//	long chA_cum(0), chB_cum(0), sq_cum(0);
//	//std::thread a, b, sq;
//
//	//sq = std::thread(ilya_CPB_g1_THREADTHREADSQ, sq_data, sq_g1, number_of_points, tau_value);
//	//a = std::thread(ilya_CPB_g1_THREADTHREAD, chA_data, chA_g1, number_of_points, tau_value);
//	//b = std::thread(ilya_CPB_g1_THREADTHREAD, chB_data, chB_g1, number_of_points, tau_value);
//	//
//	//a.join();
//	//b.join();
//	//sq.join();
//	for (int i(0); i < number_of_points; i++) {
//		//iterate over all of the data points for computing the set tau value for each array
//		chA_cum += chA_data[i] * chA_data[i + tau_value];
//		chB_cum += chB_data[i] * chB_data[i + tau_value];
//		sq_cum += sq_data[i] * sq_data[i + tau_value];
//	}
//
//	chA_g1[tau_value] = double(chA_cum) / number_of_points;
//	chB_g1[tau_value] = double(chB_cum) / number_of_points;
//	sq_g1[tau_value]  = double(sq_cum) / number_of_points;
//}
//
//void ilya_CPB_g1(short* chA_data, short* chB_data, unsigned int* sq_data,
//	double* chA_g1, double* chB_g1, double* sq_g1,
//	unsigned int number_of_records, unsigned int samples_per_record, int tauMax) {
//	/*
//	Given a set of consecutive points, compute the correlation G(T) = <X(t)X(t+T)>
//	Mupltiple records are placed in a single line, and tau values are taken across them.
//	*/
//
//	// evaluate the correlators by launching threads for different values of tau
//	std::thread* t = new std::thread[tauMax];
//	//int lastValue = 0;			int increment = floor(samples_per_record / number_of_threads);
//
//	//launch mutltiple threads
//	for (int tau_value(0); tau_value < tauMax; tau_value++) {
//		t[tau_value] = std::thread(ilya_CPB_g1_THREAD, chA_data, chB_data, sq_data,
//			chA_g1, chB_g1, sq_g1,
//			number_of_records, samples_per_record, tau_value);
//	}
//	//collect the multiple threads together
//	for (int i(0); i < tauMax; i++)
//		t[i].join();
//
//	delete[] t;
//}
//
//void ilya_CPB_g1_THREADTHREAD(short* chA_data, short* chB_data, unsigned int* sq_data,
//	double* chA_g1, double* chB_g1, double* sq_g1,
//	int tau_value, int min, int max) {
//	/*
//	Given a set of consecutive points, compute the correlation G(T) = <X(t)X(t+T)>
//	Mupltiple records are placed in a single line, and tau values are taken across them.
//	*/
//	for (int i(min); i < max; i++) {
//		//iterate over all of the data points for computing the set tau value for each array
//		chA_g1[tau_value] += chA_data[i] * chA_data[i + tau_value];
//		chB_g1[tau_value] += chB_data[i] * chB_data[i + tau_value];
//		sq_g1[tau_value] += sq_data[i] * sq_data[i + tau_value];
//	}
//}
//void ilya_CPB_g1_THREAD(short* chA_data, short* chB_data, unsigned int* sq_data,
//	double* chA_g1, double* chB_g1, double* sq_g1,
//	unsigned int number_of_records, unsigned int samples_per_record, int tau_value) {
//	/*
//	Given a set of consecutive points, compute the correlation G(T) = <X(t)X(t+T)>
//	Mupltiple records are placed in a single line, and tau values are taken across them.
//	*/
//
//	//const int number_of_threads = 100;		std::thread t[number_of_threads];
//	int number_of_threads = 100;
//	std::thread* t = new std::thread[number_of_threads];
//	int lastValue = 0;			int increment = floor(number_of_records * samples_per_record / number_of_threads);
//
//	//launch mutltiple threads
//	for (int i(0); i < number_of_threads - 1; i++) {
//		t[i] = std::thread(ilya_CPB_g1_THREADTHREAD, chA_data, chB_data, sq_data,
//			chA_g1, chB_g1, sq_g1,
//			tau_value, lastValue, lastValue + increment);
//		lastValue += increment;
//	}
//	t[number_of_threads - 1] = std::thread(ilya_CPB_g1_THREADTHREAD, chA_data, chB_data, sq_data,
//		chA_g1, chB_g1, sq_g1,
//		tau_value, lastValue, number_of_records * samples_per_record - tau_value);
//	//collect the multiple threads together
//	for (int i(0); i < number_of_threads; i++)
//		t[i].join();
//
//	//for (int i(0); i < number_of_records*samples_per_record - tau_value; i++) {
//	//	//iterate over all of the data points for computing the set tau value for each array
//	//	chA_g1[tau_value] += chA_data[i] * chA_data[i + tau_value];
//	//	chB_g1[tau_value] += chB_data[i] * chB_data[i + tau_value];
//	//	sq_g1[tau_value] += sq_data[i] * sq_data[i + tau_value];
//	//}
//
//	//normalise by the number of points
//	chA_g1[tau_value] /= (samples_per_record * number_of_records - tau_value);
//	chB_g1[tau_value] /= (samples_per_record * number_of_records - tau_value);
//	sq_g1[tau_value] /= (samples_per_record * number_of_records - tau_value);
//
//	delete[] t;
//}
//
//void ilya_CPB_g1(short* chA_data, short* chB_data, unsigned int* sq_data,
//	double* chA_g1, double* chB_g1, double* sq_g1,
//	unsigned int number_of_records, unsigned int samples_per_record, int tauMax) {
//	/*
//	Given a set of consecutive points, compute the correlation G(T) = <X(t)X(t+T)>
//	Mupltiple records are placed in a single line, and tau values are taken across them.
//	*/
//
//	// evaluate the correlators by launching threads for different values of tau
//	std::thread* t = new std::thread[tauMax];
//	//std::thread t[200];
//
//	//launch mutltiple threads
//	for (int tau_value(0); tau_value < tauMax; tau_value++) {
//		t[tau_value] = std::thread(ilya_CPB_g1_THREAD, chA_data, chB_data, sq_data,
//			chA_g1, chB_g1, sq_g1,
//			number_of_records, samples_per_record, tau_value);
//	}
//	//collect the multiple threads together
//	for (int i(0); i < tauMax; i++)
//		t[i].join();
//
//	delete[] t;
//}
//
//void ilya_CPB_g1_THREADTHREAD(short* chA_data, short* chB_data, unsigned int* sq_data,
//		double* chA_g1, double* chB_g1, double* sq_g1,
//		int tau_value, int min, int max) {
//		/*
//		Given a set of consecutive points, compute the correlation G(T) = <X(t)X(t+T)>
//		Mupltiple records are placed in a single line, and tau values are taken across them.
//		*/
//		for (int i(min); i < max; i++) {
//			//iterate over all of the data points for computing the set tau value for each array
//			chA_g1[tau_value] += chA_data[i] * chA_data[i + tau_value];
//			chB_g1[tau_value] += chB_data[i] * chB_data[i + tau_value];
//			sq_g1[tau_value] += sq_data[i] * sq_data[i + tau_value];
//		}
//}
//
//void ilya_CPB_g1_THREAD(short* chA_data, short* chB_data, unsigned int* sq_data,
//	double* chA_g1, double* chB_g1, double* sq_g1, int tauMax,
//	int min, int max) {
//	/*
//	Given a set of consecutive points, compute the correlation G(T) = <X(t)X(t+T)>
//	Mupltiple records are placed in a single line, and tau values are taken across them.
//	*/
//
//	//evaluate correlators for these parts of the array
//	for (int i(min); i < max; i++) {
//			//iterate over all of the data points for computing the set tau value for each array
//			chA_g1[tau] += chA_data[i] * chA_data[i + tau];
//			chB_g1[tau] += chB_data[i] * chB_data[i + tau];
//			sq_g1[tau] += sq_data[i] * sq_data[i + tau];
//	}
//}
//
//void ilya_CPB_g1(short* chA_data, short* chB_data, unsigned int* sq_data,
//	double* chA_g1, double* chB_g1, double* sq_g1,
//	unsigned int number_of_records, unsigned int samples_per_record, int tauMax, int number_of_threads) {
//	/*
//	Given a set of consecutive points, compute the correlation G(T) = <X(t)X(t+T)>
//	Mupltiple records are placed in a single line, and tau values are taken across them.
//	*/
//
//	/*ilya_CPB_g1_THREAD(chA_data, chB_data, sq_data,
//				chA_g1, chB_g1, sq_g1, tauMax,
//				0, number_of_records*samples_per_record - tauMax);*/
//
//	std::thread* t = new std::thread[number_of_threads];
//	int lastValue = 0;
//	int increment = floor(number_of_records * samples_per_record / number_of_threads);
//
//	//1) launch multiple threads for different parts of the dataset
//	for (int i(0); i < number_of_threads - 1; i++) {
//		t[i] = std::thread(ilya_CPB_g1_THREAD, chA_data, chB_data, sq_data,
//			chA_g1, chB_g1, sq_g1, tauMax,
//			lastValue, lastValue + increment);
//		lastValue += increment;
//	}
//	t[number_of_threads - 1] = std::thread(ilya_CPB_g1_THREAD, chA_data, chB_data, sq_data,
//		chA_g1, chB_g1, sq_g1, tauMax,
//		lastValue, number_of_records * samples_per_record - tauMax);
//
//	//2) collect the multiple threads together
//	for (int i(0); i < number_of_threads; i++)
//		t[i].join();
//
//	delete[] t;
//}
//
//void ilya_CPB_g1RECORD_THREAD(short* chA_data, short* chB_data, double* chA_correlation, double* chB_correlation,
//	unsigned int number_of_record, unsigned int samples_per_record,
//	int min, int max, int tau_cutOff) {
//	/*
//	Given a set of points X(t), computes the correlation G(T) = <X(t+T)X(t)>
//	In this method RECORD is fixec and tau are iterated over in each thread
//	 Note that the ouput needs to be averaged!
//	*/
//	int base_record_index(0);
//
//	for (int record(min); record < max; record++) {
//		//enter each record
//		base_record_index = record * samples_per_record;
//		for (int counter_tau(tau_cutOff); counter_tau < samples_per_record; counter_tau++) {
//			//iterate over the tau values
//			for (int counter_t(0); counter_t < samples_per_record - counter_tau; counter_t++) {
//				//iterate over the t values in each record
//				chA_correlation[counter_tau] += chA_data[base_record_index + counter_t] * chA_data[base_record_index + counter_t + counter_tau];
//				chB_correlation[counter_tau] += chB_data[base_record_index + counter_t] * chB_data[base_record_index + counter_t + counter_tau];
//			}
//		}
//	}
//}
//
//void ilya_CPB_g1RECORD(short* chA_data, short* chB_data,
//	unsigned int number_of_records, unsigned int samples_per_record,
//	double* chA_g1, double* chB_g1, int tau_cutOff) {
//	/*
//	Given a set of points X(t), computes the correlation G(T) = <X(t+T)X(t)>
//	In this method RECORD is fixec and tau are iterated over in each thread
//	 Note that the ouput needs to be averaged!
//	*/
//
//	// evaluate the correlators by launching threads for different values of tau
//	const int number_of_threads = 1000;		std::thread t[number_of_threads];
//	int lastValue = 0;			int increment = floor(number_of_records / number_of_threads);
//
//	//launch mutltiple threads
//	for (int i(0); i < number_of_threads - 1; i++) {
//		t[i] = std::thread(ilya_CPB_g1RECORD_THREAD, chA_data, chB_data,
//			chA_g1, chB_g1,
//			number_of_records, samples_per_record,
//			lastValue, lastValue + increment, tau_cutOff);
//		lastValue += increment;
//	}
//	t[number_of_threads - 1] = std::thread(ilya_CPB_g1RECORD_THREAD, chA_data, chB_data,
//		chA_g1, chB_g1,
//		number_of_records, samples_per_record,
//		lastValue, number_of_records, tau_cutOff);
//	//collect the multiple threads together
//	for (int i(0); i < number_of_threads; i++)
//		t[i].join();
//}
//
//
//void ilya_CPB_PYTHON_1(void* adq_cu_ptr,
//	double* chA_out, double* chB_out, double* sq_out,
//	short chA_background, short chB_background,
//	unsigned int samples_per_record, unsigned int number_of_records,
//	long long* chA_cumulative, long long* chB_cumulative, unsigned long long* sq_cumulative,
//	double* chA_g1_cumlative, double* chB_g1_cumulative, double* sq_g1_cumulative,
//	int flat_average_points,
//	int repetition_number) {
//
//	/*
//	Has the same functionality as ilya_CPB, except that there are no repitions or parrallel threads
//	!!!!!!!!!! Multirecord must be set up !!!!!!!!!!
//	*/
//	short* chA_data = new short[samples_per_record*number_of_records]();
//	short* chB_data = new short[samples_per_record*number_of_records]();
//	unsigned int* sq_data = new unsigned int[samples_per_record*number_of_records]();
//	double* chA_g1 = new double[samples_per_record]();
//	double* chB_g1 = new double[samples_per_record]();
//	double* sq_g1 = new double[samples_per_record]();
//	int chA_average(-1), chB_average(-1), sq_average(-1);
//
//	//1) collect the data
//	ilya_CPB_collect_data(adq_cu_ptr, chA_data, chB_data, samples_per_record, number_of_records);
//
//	//2) process the data
//	ilya_CPB_process_data(chA_data, chB_data, sq_data, chA_background, chB_background, chA_average, chB_average, sq_average, flat_average_points, samples_per_record, number_of_records, 10);
//
//	//3) evaluate the correlation
//	/*ilya_CPB_g1TAU(chA_data, chB_data, number_of_records, samples_per_record, chA_g1, chB_g1);
//	for (int i(0); i < samples_per_record; i++) {
//	chA_g1_cumlative[i] = (chA_g1_cumlative[i] * (repetition_number - 1) + chA_g1[i]) / repetition_number;
//	chB_g1_cumulative[i] = (chB_g1_cumulative[i] * (repetition_number - 1) + chB_g1[i]) / repetition_number;
//	}*/
//
//	ilya_CPB_g1(chA_data, chB_data, sq_data, chA_g1, chB_g1, sq_g1, samples_per_record, chA_average, chB_average, sq_average, number_of_records, samples_per_record, 10);
//	for (int i(0); i < samples_per_record; i++) {
//		chA_g1_cumlative[i] = (chA_g1_cumlative[i] * (repetition_number - 1) + chA_g1[i]) / (repetition_number);
//		chB_g1_cumulative[i] = (chB_g1_cumulative[i] * (repetition_number - 1) + chB_g1[i]) / (repetition_number);
//		sq_g1_cumulative[i] = (sq_g1_cumulative[i] * (repetition_number - 1) + sq_g1[i]) / (repetition_number);
//	}
//
//	//4) store data
//	ilya_CPB_basic_accumulate(chA_data, chB_data, sq_data, chA_cumulative, chB_cumulative, sq_cumulative, samples_per_record, number_of_records, 10);
//
//	//5) compute squares
//	ilya_CPB_cumulative_average(chA_cumulative, chB_cumulative, sq_cumulative, chA_out, chB_out, sq_out, number_of_records, samples_per_record, repetition_number);
//
//	/*chA_out[0] = chA_average;
//	chA_out[1] = chB_average;
//	chA_out[2] = sq_average;*/
//	/*sq_out[0] = chA_average;
//	sq_out[1] = chB_average;
//	sq_out[2] = sq_average;*/
//
//	delete[] chA_data;
//	delete[] chB_data;
//	delete[] sq_data;
//	delete[] chA_g1;
//	delete[] chB_g1;
//	delete[] sq_g1;
//}
/////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////
//
//
//
//
//void TEST_ilya_CPB_g1(unsigned int samples_per_record, unsigned int number_of_records, int repetitions
//	,int number_of_threads
//	) {
//	/*
//	Test how fast the correlation function is
//	*/
//	int total_number_samples = samples_per_record * number_of_records;
//	short* chA_data = new short[total_number_samples];
//	short* chB_data = new short[total_number_samples];
//	unsigned int* sq_data = new unsigned int[total_number_samples];
//	double* chA_correlation = new double[samples_per_record]();
//	double* chB_correlation = new double[samples_per_record]();
//	double* sq_g1 = new double[samples_per_record]();
//
//	for (int i(0); i < total_number_samples; i++) {
//		chA_data[i] = 2*i;
//		chB_data[i] = 2 * i;
//		sq_data[i] = 3 * i;
//	}
//	int chA_av, chB_av, sq_av;
//
//	for (int i(0); i < repetitions; i++)
//		ilya_CPB_g1(chA_data, chB_data, sq_data, chA_correlation, chB_correlation, sq_g1, samples_per_record, chA_av, chB_av, sq_av, number_of_records, samples_per_record, number_of_threads);
//
//	delete[] chA_data;
//	delete[] chA_correlation;
//	delete[] chB_data;
//	delete[] chB_correlation;
//	delete[] sq_data;
//	delete[] sq_g1;
//}
//
//void ilya_CPB_g1TAU_THREAD_THREAD(short* chA_data, short* chB_data,
//	unsigned int number_of_record, unsigned int samples_per_record, int chA_correlation_cumulative, int chB_correlation_cumulative, int counter_tau,
//	int min, int max) {
//	/*
//	Given a set of points X(t), computes the correlation G(T) = <X(t+T)X(t)>
//	In this method a tau is fixec and records are iterated over in each thread
//	The corresponding average tau is written to the array
//
//	1) This is the first layer, where with a given tau value, we iterate over each record
//	*/
//	int base_record_index(0);
//
//	for (int record(min); record < max; record++) {
//		//enter each record
//		base_record_index = record * samples_per_record;
//		for (int counter_t(0); counter_t < samples_per_record - counter_tau; counter_t++) {
//			//iterate over the t values in each record
//			chA_correlation_cumulative += chA_data[base_record_index + counter_t] * chA_data[base_record_index + counter_t + counter_tau];
//			chB_correlation_cumulative += chB_data[base_record_index + counter_t] * chB_data[base_record_index + counter_t + counter_tau];
//		}
//	}
//}
//
//short average_section(short* array_to_average, int begin, int end) {
//	/*
//	Finds average of an array
//	*/
//	int sum(0);
//
//	for (int i(begin); i < end; i++)
//		sum += array_to_average[i];
//
//	return (short)((float)sum / (end - begin));
//}
//
//void ilya_CPB_g1TAU_THREAD(short* chA_data, short* chB_data,
//	double* chA_g1, double* chB_g1,
//	unsigned int number_of_record, unsigned int samples_per_record,
//	int min, int max) {
//	/*
//	Given a set of points X(t), computes the correlation G(T) = <X(t+T)X(t)>
//	The corresponding average tau is written to the array
//	*/
//	int base_record_index(0);
//	long long chA_correlation_cumulative(0);
//	//long long chB_correlation_cumulative(0);
//
//	for (int counter_tau(min); counter_tau < max; counter_tau++) {
//		//iterate over the tau values
//		for (int record(0); record < number_of_record; record++) {
//			//enter each record
//			base_record_index = record * samples_per_record;
//			short record_average = average_section(chA_data, base_record_index, base_record_index + samples_per_record);
//			for (int counter_t(0); counter_t < samples_per_record - counter_tau; counter_t++) {
//				//iterate over the t values in each record
//				chA_correlation_cumulative += (chA_data[base_record_index + counter_t] - record_average) * (chA_data[base_record_index + counter_t + counter_tau] - record_average);
//				//chB_correlation_cumulative += (chB_data[base_record_index + counter_t]) * chB_data[base_record_index + counter_t + counter_tau];
//			}
//		}
//		chA_g1[counter_tau] = double(chA_correlation_cumulative) / (number_of_record * (samples_per_record - counter_tau));
//		//chB_g1[counter_tau] = double(chB_correlation_cumulative) / (number_of_record * (samples_per_record - counter_tau));
//		chA_correlation_cumulative = 0;
//		//chB_correlation_cumulative = 0;
//	}
//}
//
//void ilya_CPB_g1TAU(short* chA_data, short* chB_data,
//	unsigned int number_of_records, unsigned int samples_per_record,
//	double* chA_g1, double* chB_g1) {
//	/*
//	Given a set of points X(t), computes the correlation G(T) = <X(t+T)X(t)>
//	In this method a tau is fixed and records are iterated over in each thread
//	The corresponding average tau is written to the array
//
//	!* We only evaluate for tau greater than 100ns, since otherwise there will be the large initial overlap*!
//	*/
//
//	// evaluate the correlators by launching threads for different values of tau
//	const int number_of_threads = 200;		std::thread t[number_of_threads];
//	int tau_cutOff = 0;					int lastValue = tau_cutOff;			int increment = floor((samples_per_record - tau_cutOff) / number_of_threads);
//
//	//launch mutltiple threads
//	for (int i(0); i < number_of_threads - 1; i++) {
//		t[i] = std::thread(ilya_CPB_g1TAU_THREAD, chA_data, chB_data,
//			chA_g1, chB_g1,
//			number_of_records, samples_per_record,
//			lastValue, lastValue + increment);
//		lastValue += increment;
//	}
//	t[number_of_threads - 1] = std::thread(ilya_CPB_g1TAU_THREAD, chA_data, chB_data,
//		chA_g1, chB_g1,
//		number_of_records, samples_per_record,
//		lastValue, samples_per_record - tau_cutOff);
//	//collect the multiple threads together
//	for (int i(0); i < number_of_threads; i++)
//		t[i].join();
//}
//void TEST_ilya_CPB_basic_accumulate(int TOTAL_number_of_samples, int repetitions, int number_of_threads) {
//	/*
//	Run the writting process
//	*/
//	short* chA_data = new short[TOTAL_number_of_samples];
//	short* chB_data = new short[TOTAL_number_of_samples];
//	unsigned int* sq_data = new unsigned int[TOTAL_number_of_samples];
//	long long* chA_cumulative = new long long[TOTAL_number_of_samples]();
//	long long* chB_cumulative = new long long[TOTAL_number_of_samples]();
//	unsigned long long* sq_cumulative = new unsigned long long[TOTAL_number_of_samples]();
//
//	for (int i(0); i < TOTAL_number_of_samples; i++) {
//		chA_data[i] = i;
//		chB_data[i] = i;
//		sq_data[i] = 3*i;
//	}
//
//	for (int i(0); i < repetitions; i++)
//		ilya_CPB_basic_accumulate(chA_data, chB_data, sq_data,
//			chA_cumulative, chB_cumulative, sq_cumulative, TOTAL_number_of_samples, 1, number_of_threads);
//
//	delete[] chA_data;
//	delete[] chB_data;
//	delete[] sq_data;
//	delete[] chA_cumulative;
//	delete[] chB_cumulative;
//	delete[] sq_cumulative;
//}
//
//void TEST_ilya_CPB_g1TAU(unsigned int samples_per_record, unsigned int number_of_records, int repetitions) {
//	/*
//	Test how fast the correlation function is
//	*/
//	int total_number_samples = samples_per_record * number_of_records;
//	short* chA_data = new short[total_number_samples];
//	short* chB_data = new short[total_number_samples];
//	double* chA_correlation = new double[samples_per_record];
//	double* chB_correlation = new double[samples_per_record];
//
//	for (int i(0); i < total_number_samples; i++) {
//		chA_data[i] = i;
//		chB_data[i] = 2 * i;
//	}
//
//	for (int i(0); i < repetitions; i++)
//		ilya_CPB_g1TAU(chA_data, chB_data, number_of_records, samples_per_record, chA_correlation, chB_correlation);
//
//	delete[] chA_data;
//	delete[] chA_correlation;
//	delete[] chB_data;
//	delete[] chB_correlation;
//}
