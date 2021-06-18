#include <cppunit/extensions/HelperMacros.h>
#include <cppunit/Exception.h>

#include "power_kernel.hpp"
#include "utils.hpp"
#include "logging.hpp"

#ifdef TESTENV
// #define R_POINTS 8 defined in Makefile
// #define SP_POINTS 3 defined in Makefile
#endif

class PowerGpuTest : public CppUnit::TestFixture{

    // Macro for generating suite
    CPPUNIT_TEST_SUITE( PowerGpuTest );

    // Population with tests
    CPPUNIT_TEST( test_power_kernel_1_streams );
    CPPUNIT_TEST( test_power_kernel_2_streams );
    CPPUNIT_TEST( test_power_kernel_cumulative );

    CPPUNIT_TEST_SUITE_END();
private:
    short *chA_data;
    short *chB_data;
    double **data_out;
    short *chA_background;
    short *chB_background;

    // For kernel using streams, data needs to be locked in memory
    short* chA_data_locked;
    short* chB_data_locked;

    // Auxillary arrays for storing pointers to memory allocated on GPU and memory locked CPU
    // 1. To communicate with the GPU, we need pointers (short *pointer_name) that keep the memory locations on the GPU
    // 2. In order to modify these pointers, we need to pass them in as addresses (&pointer_name) into the functions that allocate and bind the GPU memory
    // 3. There are addresses for every computation being run: chA, chB, chAsq, chBsq
    // 4. There are addresses for every stream being run: stream0, stream1
    // This it's a 2D array (**) that keeps the (*)GPU POINTERS so it's a triple pointer
    short*** gpu_in; long ***gpu_out; long ***cpu_out;

    // We will only test 2 of the output arrays
    double *expected_A_out;
    double *expected_B_out;
    double *expected_sq_out;

public:
    void setUp(){
        /*
         * chA
         * 1  0  0
         * 0  0  0
         * 0  0  0
         * 0  0  0
         * 2  1  1
         * 4  3  2
         * 3  1  1
         * 3  1  8
         *
         * chB
         * 0  1  2
         * 3  4  5
         * 6  7  9
         * 10 11 12
         * 0  0  0
         * 1  1  1
         * 2  2  2
         * 3  3  3
         *
         * sq
         * 1  1  4
         * 9  16 25
         * 36 49 81
         * 100 121 144
         * 4   1   1
         * 17  10  5
         * 13  5   5
         * 18  10  73
         */

        chA_data = new short[24]{2, 2, 3,
                1, 2, 3,
                1, 2, 3,
                1, 2, 3,
                3, 3, 4,
                5, 5, 5,
                4, 3, 4,
                4, 3, 11};
        chB_data = new short[24]{0, 1, 2,
                3, 4, 5,
                6, 7, 9,
                10, 11, 12,
                0, 0, 0,
                1, 1, 1,
                2, 2, 2,
                3, 3, 3};

        chA_background = new short[3]{1, 2, 3};
        chB_background = new short[3]{0, 0, 0};

        data_out = new double*[POWER::no_outputs];
        for (int i(0); i < POWER::no_outputs; i++)
            data_out[i] = new double[3]();

        expected_A_out = new double[3]{1.625, 0.75, 1.5};
        expected_B_out = new double[3]{3.125, 3.625, 4.25};
        expected_sq_out = new double[3]{24.75, 26.625, 42.25};
    }
    void tearDown(){
        delete[] chA_data;
        delete[] chB_data;

        delete[] chA_background;
        delete[] chB_background;

        for (int i(0); i < POWER::no_outputs; i++)
            delete[] data_out[i];
        delete[] data_out;

        delete[] expected_A_out;
        delete[] expected_B_out;
        delete[] expected_sq_out;
    }

    void test_power_kernel_2_streams() {
        /* Data would be split like so
         * chA
         * 1  0  0   stream0
         * 0  0  0   stream0
         * 0  0  0   stream1
         * 0  0  0   stream1
         * 2  1  1   stream0
         * 4  3  2   stream0
         * 3  1  1   stream1
         * 3  1  8   stream1
         */

        int no_streams = 2;

        // Check kernel
        POWER::GPU::check_power_kernel_parameters();

        // Copy over background data
        POWER::GPU::copy_background_arrays_to_gpu(chA_background, chB_background);

        // Allocate memory and fill it up
        POWER::GPU::allocate_memory(&chA_data_locked, &chB_data_locked, &gpu_in, &gpu_out, &cpu_out, no_streams);
        for (int i(0); i < 24; i++) {
            chA_data_locked[i] = chA_data[i];
            chB_data_locked[i] = chB_data[i];
        }

        // Run power kernel
        POWER::GPU::power_kernel(chA_data_locked, chB_data_locked,
                                 data_out,
                                 gpu_in, gpu_out, cpu_out, no_streams);

        // Free memory
        POWER::GPU::free_memory(chA_data_locked, chB_data_locked,
                                gpu_in, gpu_out, cpu_out, no_streams);

        // Compare
        for (int i(0); i < 1; i++) {
            CPPUNIT_ASSERT_EQUAL_MESSAGE("Failed chA", expected_A_out[i], data_out[CHA][i]);
            CPPUNIT_ASSERT_EQUAL_MESSAGE("Failed chB", expected_B_out[i], data_out[CHB][i]);
            CPPUNIT_ASSERT_EQUAL_MESSAGE("Failed sq", expected_sq_out[i], data_out[SQ][i]);
        }
    }

    void test_power_kernel_1_streams() {
        /* Data would be split like so
         * chA
         * 1  0  0   stream0
         * 0  0  0   stream0
         * 0  0  0   stream0
         * 0  0  0   stream0
         * 2  1  1   stream0
         * 4  3  2   stream0
         * 3  1  1   stream0
         * 3  1  8   stream0
         */
        int no_streams = 1;

        // Check kernel
        POWER::GPU::check_power_kernel_parameters();

        // Copy over background data
        POWER::GPU::copy_background_arrays_to_gpu(chA_background, chB_background);

        // Allocate memory and fill it up
        POWER::GPU::allocate_memory(&chA_data_locked, &chB_data_locked, &gpu_in, &gpu_out, &cpu_out, no_streams);
        for (int i(0); i < 24; i++) {
            chA_data_locked[i] = chA_data[i];
            chB_data_locked[i] = chB_data[i];
        }

        // Run power kernel
        POWER::GPU::power_kernel(chA_data_locked, chB_data_locked,
                                 data_out,
                          gpu_in, gpu_out, cpu_out, no_streams);

        // Free memory
        POWER::GPU::free_memory(chA_data_locked, chB_data_locked,
                                gpu_in, gpu_out, cpu_out, no_streams);

        // Compare
        for (int i(0); i < 1; i++) {
            CPPUNIT_ASSERT_EQUAL_MESSAGE("Failed chA", expected_A_out[i], data_out[CHA][i]);
            CPPUNIT_ASSERT_EQUAL_MESSAGE("Failed chB", expected_B_out[i], data_out[CHB][i]);
            CPPUNIT_ASSERT_EQUAL_MESSAGE("Failed sq", expected_sq_out[i], data_out[SQ][i]);
        }
    }

    void test_power_kernel_cumulative() {
        // For this test accumulate data instead of normalising it
        long **data_out_cumulative = new long*[POWER::no_outputs];
        for (int i(0); i < POWER::no_outputs; i++)
            data_out_cumulative[i] = new long[3]();

        int no_streams = 4;
        int no_runs = 10;

        // Check kernel
        POWER::GPU::check_power_kernel_parameters();

        // Copy over background data
        POWER::GPU::copy_background_arrays_to_gpu(chA_background, chB_background);

        // Allocate memory and fill it up
        POWER::GPU::allocate_memory(&chA_data_locked, &chB_data_locked, &gpu_in, &gpu_out, &cpu_out, no_streams);
        for (int i(0); i < 24; i++) {
            chA_data_locked[i] = chA_data[i];
            chB_data_locked[i] = chB_data[i];
        }

        // Run power kernel multiple times

        for (int i(0); i < no_runs; i++)
            POWER::GPU::power_kernel(chA_data_locked, chB_data_locked,
                                     data_out_cumulative,
                                     gpu_in, gpu_out, cpu_out, no_streams);

        // Free memory
        POWER::GPU::free_memory(chA_data_locked, chB_data_locked,
                                gpu_in, gpu_out, cpu_out, no_streams);

        // Compare
        for (int i(0); i < 1; i++) {
            CPPUNIT_ASSERT_DOUBLES_EQUAL_MESSAGE("Failed chA",
                                                 expected_A_out[i] * R_POINTS * no_runs, data_out_cumulative[CHA][i], 0.00001);
            CPPUNIT_ASSERT_DOUBLES_EQUAL_MESSAGE("Failed chB",
                                                 expected_B_out[i] * R_POINTS * no_runs, data_out_cumulative[CHB][i], 0.00001);
            CPPUNIT_ASSERT_DOUBLES_EQUAL_MESSAGE("Failed sq",
                                                 expected_sq_out[i] * R_POINTS * no_runs, data_out_cumulative[SQ][i], 0.00001);
        }
    }

};
CPPUNIT_TEST_SUITE_REGISTRATION( PowerGpuTest );
