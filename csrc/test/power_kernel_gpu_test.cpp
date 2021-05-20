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
    CPPUNIT_TEST( test_power_kernel_no_background );
    CPPUNIT_TEST( test_power_kernel_const_background );
    CPPUNIT_TEST( test_power_kernel_background );

    CPPUNIT_TEST_SUITE_END();
private:
    short *chA_data;
    short *chB_data;
    double **data_out;
    // We will only test 2 of the output arrays
    double *expected_A_out;
    double *expected_B_out;
    double *expected_sq_out;

    // Allocation of memory on the GPU
    short *gpu_chA_data; short *gpu_chB_data;
    short **gpu_in[2] = {&gpu_chA_data, &gpu_chB_data};

    double *gpu_chA_out; double *gpu_chB_out; double *gpu_chAsq_out; double *gpu_chBsq_out;
    double **gpu_out[4] = {&gpu_chA_out,
                           &gpu_chB_out,
                           &gpu_chAsq_out,
                           &gpu_chBsq_out};

public:
    void setUp(){
        data_out = new double*[NO_OF_POWER_KERNEL_OUTPUTS];
    }
    void tearDown(){
        delete[] data_out;
    }

    void test_power_kernel_no_background(){
        /*chA
         * 1  2  3 -> main axis (3)
         * 4  5  6
         * 7  8  9
         * 10 11 12
         * |
         * repetition axis (4)

         * chB
         * 0  1  0
         * 1  0  1
         * 0  1  0
         * 2  2  2

         * sq
         * 1  5   9
         * 17 25  37
         * 49 65  81
         * 104 125 148
         */

        // Allocation on CPU
        chA_data = new short[12]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
        chB_data = new short[12]{0, 1, 0, 1, 0, 1, 0, 1, 0, 2, 2, 2};
        short *chA_background = new short[3]{0, 0, 0};
        short *chB_background = new short[3]{0, 0, 0};

        // Initialise default ouput data
        for (int i(0); i < NO_OF_POWER_KERNEL_OUTPUTS; i++)
            data_out[i] = new double[3]();
        expected_sq_out = new double[3]{
                                        (double)(1 + 17 + 49 + 104) / 4,
                                        (double)(5 + 25 + 65 + 125) / 4,
                                        (double)(9 + 37 + 81 + 148) / 4};

        GPU::copy_background_arrays_to_gpu(chA_background, chB_background);
        GPU::V1::allocate_memory(gpu_in, gpu_out);
        GPU::V1::power_kernel(chA_data, chB_data,
                              data_out,
                              gpu_in, gpu_out);
        GPU::V1::free_memory(gpu_in, gpu_out);

        // Compare
        for (int i(0); i < 3; i++)
            CPPUNIT_ASSERT_EQUAL(expected_sq_out[i], data_out[SQ][i]);

        delete[] chA_data;
        delete[] chB_data;
        delete[] chA_background;
        delete[] chB_background;
        for (int i(0); i < NO_OF_POWER_KERNEL_OUTPUTS; i++) {
            delete[] data_out[i];
        }
        delete[] expected_A_out;
        delete[] expected_B_out;
        delete[] expected_sq_out;
    }

    void test_power_kernel_const_background(){
        /* chA
         * 1  2  3 -> main axis (3)
         * 4  5  6
         * 7  8  9
         * 10 11 12
         * |
         * repetition axis (4)

         * chB
         * 0  1  0
         * 1  0  1
         * 0  1  0
         * 2  2  2

         * sq
         * 0  2  4
         * 10 16 26
         * 36 50 64
         * 85 104 125
         */

        // Allocation on CPU
        chA_data = new short[12]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
        chB_data = new short[12]{0, 1, 0, 1, 0, 1, 0, 1, 0, 2, 2, 2};
        short *chA_background = new short[3]{1, 1, 1};
        short *chB_background = new short[3]{0, 0, 0};

        // Initialise default ouput data
        for (int i(0); i < NO_OF_POWER_KERNEL_OUTPUTS; i++) {
            data_out[i] = new double[3]();
        }
        expected_sq_out = new double[3]{(double)(0 + 10 + 36 + 85) / 4,
                (double)(2 + 16 + 50 + 104) / 4,
                (double)(4 + 26 + 64 + 125) / 4};

        GPU::copy_background_arrays_to_gpu(chA_background, chB_background);
        GPU::V1::allocate_memory(gpu_in, gpu_out);
        GPU::V1::power_kernel(chA_data, chB_data,
                              data_out,
                              gpu_in, gpu_out);
        GPU::V1::free_memory(gpu_in, gpu_out);

        // Compare
        for (int i(0); i < 3; i++)
            CPPUNIT_ASSERT_EQUAL(expected_sq_out[i], data_out[SQ][i]);

        delete[] chA_data;
        delete[] chB_data;
        delete[] chA_background;
        delete[] chB_background;
        for (int i(0); i < NO_OF_POWER_KERNEL_OUTPUTS; i++) {
            delete[] data_out[i];
        }
        delete[] expected_A_out;
        delete[] expected_B_out;
        delete[] expected_sq_out;
    }

    void test_power_kernel_background(){
        /*
         * sq
         * 0  1  4
         * 9  16 25
         * 36 49 81
         * 100 121 144
         */

        // Allocation on CPU
        chA_data = new short[12]{1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3};
        chB_data = new short[12]{0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12};
        short *chA_background = new short[3]{1, 2, 3};
        short *chB_background = new short[3]{0, 0, 0};

        // Initialise default ouput data
        for (int i(0); i < NO_OF_POWER_KERNEL_OUTPUTS; i++) {
            data_out[i] = new double[3]();
        }
        expected_A_out = new double[3]{0, 0, 0};
        expected_B_out = new double[3]{(double)(0 + 3 + 6+ 10)/4,
                (double)(1 + 4 + 7 + 11)/4,
                (double)(2 + 5 + 9 + 12)/4};
        expected_sq_out = new double[3]{(double)(0 + 9 + 36 + 100) / 4,
                (double)(1 + 16 + 49 + 121) / 4,
                (double)(4 + 25 + 81 + 144) / 4};

        GPU::copy_background_arrays_to_gpu(chA_background, chB_background);
        GPU::V1::allocate_memory(gpu_in, gpu_out);
        GPU::V1::power_kernel(chA_data, chB_data,
                              data_out,
                              gpu_in, gpu_out);
        GPU::V1::free_memory(gpu_in, gpu_out);

        // Compare
        for (int i(0); i < 3; i++) {
            CPPUNIT_ASSERT_EQUAL(expected_A_out[i], data_out[CHA][i]);
            CPPUNIT_ASSERT_EQUAL(expected_B_out[i], data_out[CHB][i]);
            CPPUNIT_ASSERT_EQUAL(expected_sq_out[i], data_out[SQ][i]);
        }

        delete[] chA_data;
        delete[] chB_data;
        delete[] chA_background;
        delete[] chB_background;
        for (int i(0); i < NO_OF_POWER_KERNEL_OUTPUTS; i++)
            delete[] data_out[i];
        delete[] expected_A_out;
        delete[] expected_B_out;
        delete[] expected_sq_out;
    }
};
CPPUNIT_TEST_SUITE_REGISTRATION( PowerGpuTest );
