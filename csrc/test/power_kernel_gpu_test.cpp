#include <cppunit/extensions/HelperMacros.h>
#include <cppunit/Exception.h>

#include "power_kernel.hpp"
#include "utils.hpp"

#ifdef TESTENV
#define NO_POINTS 9
// R_POINTS=3 defined in Makefile
// SP_POINTS=4 defined in Makefile
#endif

class PowerGpuTest : public CppUnit::TestFixture{

    // Macro for generating suite
    CPPUNIT_TEST_SUITE( PowerGpuTest );

    // Population with tests
    CPPUNIT_TEST( test_power_kernel_V1 );
    CPPUNIT_TEST( test_power_kernel_V2 );

    CPPUNIT_TEST_SUITE_END();
private:
    short *chA_data;
    short *chB_data;
    double **data_out;
    short *chA_background;
    short *chB_background;

    // We will only test 2 of the output arrays
    double *expected_A_out;
    double *expected_B_out;
    double *expected_sq_out;

    // Auxillary arrays for storing pointers to memory allocated on GPU
    short *gpu_chA_data0; short *gpu_chB_data0; short **gpu_in0[2] = {&gpu_chA_data0, &gpu_chB_data0};
    short *gpu_chA_data1; short *gpu_chB_data1; short **gpu_in1[2] = {&gpu_chA_data1, &gpu_chB_data1};

    double *gpu_chA_out0; double *gpu_chB_out0; double *gpu_chAsq_out0; double *gpu_chBsq_out0;
    double **gpu_out0[4] = {&gpu_chA_out0, &gpu_chB_out0, &gpu_chAsq_out0, &gpu_chBsq_out0};
    double *gpu_chA_out1; double *gpu_chB_out1; double *gpu_chAsq_out1; double *gpu_chBsq_out1;
    double **gpu_out1[4] = {&gpu_chA_out1, &gpu_chB_out1, &gpu_chAsq_out1, &gpu_chBsq_out1};

    double *cpu_chA_out0 = 0; double* cpu_chB_out0 = 0; double* cpu_chAsq_out0 = 0; double* cpu_chBsq_out0 = 0;
    double **cpu_out[4] = {&cpu_chA_out0, &cpu_chB_out0, &cpu_chAsq_out0, &cpu_chBsq_out0};


public:
    void setUp(){
        /*
         * chA
         * 0  0  0
         * 0  0  0
         * 0  0  0
         * 0  0  0
         *
         * chB
         * 0  1  2
         * 3  4  5
         * 6  7  9
         * 10 11 12
         *
         * sq
         * 0  1  4
         * 9  16 25
         * 36 49 81
         * 100 121 144
         */

        chA_data = new short[12]{1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3};
        chB_data = new short[12]{0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12};

        chA_background = new short[3]{1, 2, 3};
        chB_background = new short[3]{0, 0, 0};

        data_out = new double*[NO_OF_POWER_KERNEL_OUTPUTS];
        for (int i(0); i < NO_OF_POWER_KERNEL_OUTPUTS; i++)
            data_out[i] = new double[3]();

        expected_A_out = new double[3]{0, 0, 0};
        expected_B_out = new double[3]{(double)(0 + 3 + 6+ 10)/4,
                (double)(1 + 4 + 7 + 11)/4,
                (double)(2 + 5 + 9 + 12)/4};
        expected_sq_out = new double[3]{(double)(0 + 9 + 36 + 100) / 4,
                (double)(1 + 16 + 49 + 121) / 4,
                (double)(4 + 25 + 81 + 144) / 4};
    }
    void tearDown(){
        delete[] chA_data;
        delete[] chB_data;

        delete[] chA_background;
        delete[] chB_background;

        for (int i(0); i < NO_OF_POWER_KERNEL_OUTPUTS; i++)
            delete[] data_out[i];
        delete[] data_out;

        delete[] expected_A_out;
        delete[] expected_B_out;
        delete[] expected_sq_out;
    }

    void test_power_kernel_V1() {

        // Standrd pipeline for: copying over background data -> allocating memory -> running kernel
        GPU::copy_background_arrays_to_gpu(chA_background, chB_background);
        GPU::V1::allocate_memory(gpu_in0, gpu_out0);
        GPU::V1::power_kernel(chA_data, chB_data,
                              data_out,
                              gpu_in0, gpu_out0);
        GPU::V1::free_memory(gpu_in0, gpu_out0);

        // Compare
        for (int i(0); i < 3; i++) {
            CPPUNIT_ASSERT_EQUAL(expected_A_out[i], data_out[CHA][i]);
            CPPUNIT_ASSERT_EQUAL(expected_B_out[i], data_out[CHB][i]);
            CPPUNIT_ASSERT_EQUAL(expected_sq_out[i], data_out[SQ][i]);
        }
    }

    void test_power_kernel_V2() {
        /* Data would be split like so
         * chA
         * 0  0  0   stream0
         * 0  0  0   stream0
         * 0  0  0   stream1
         * 0  0  0   stream1
         *
         * chB
         * 0  1  2   stream0
         * 3  4  5   stream0
         * 6  7  9   stream1
         * 10 11 12  stream1
         *
         * sq
         * 0  1  4
         * 9  16 25
         * 36 49 81
         * 100 121 144
         */
        // For kernel using streams, data needs to be locked in memory
        short* chA_data_locked;
        short* chB_data_locked;

        // Standrd pipeline for: copying over background data -> allocating memory -> running kernel
        GPU::copy_background_arrays_to_gpu(chA_background, chB_background);
        GPU::V2::allocate_memory(&chA_data_locked, &chB_data_locked,
                                 gpu_in0, gpu_in1, gpu_out0, gpu_out1, cpu_out);
        // Copy over test input data to the allocated memory
        for (int i(0); i < 12; i++) {
            chA_data_locked[i] = chA_data[i];
            chB_data_locked[i] = chB_data[i];
        }

        GPU::V2::power_kernel(chA_data, chB_data,
                              data_out,
                              gpu_in0, gpu_in1,
                              gpu_out0, gpu_out1,
                              cpu_out);
        GPU::V2::free_memory(&chA_data_locked, &chB_data_locked,
                             gpu_in0, gpu_in1, gpu_out0, gpu_out1, cpu_out);

        // Compare
        for (int i(0); i < 3; i++) {
            CPPUNIT_ASSERT_EQUAL(expected_A_out[i], data_out[CHA][i]);
            CPPUNIT_ASSERT_EQUAL(expected_B_out[i], data_out[CHB][i]);
            CPPUNIT_ASSERT_EQUAL(expected_sq_out[i], data_out[SQ][i]);
        }
    }



};
CPPUNIT_TEST_SUITE_REGISTRATION( PowerGpuTest );
