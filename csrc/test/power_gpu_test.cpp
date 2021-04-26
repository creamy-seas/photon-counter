#include <cppunit/extensions/HelperMacros.h>
#include <cppunit/Exception.h>

#include "power_kernel.hpp"
#include "utils.hpp"

#ifdef TESTENV
#define NO_POINTS 9
// R_POINTS=3 defined in Makefile
// NP_POINTS=4 defined in Makefile
#endif

class PowerGpuTest : public CppUnit::TestFixture {

        // Macro for generating suite
        CPPUNIT_TEST_SUITE( PowerGpuTest );

        // Population with tests
        CPPUNIT_TEST( test_power_kernel_v1_no_background );
        // CPPUNIT_TEST( test_power_kernel_v2_const_background );
        // CPPUNIT_TEST( test_power_kernel_v3_background );

        CPPUNIT_TEST_SUITE_END();
private:
        short *chA_data;
        short *chB_data;
        float *sq_data;
        float *expected_sq_data;

        // Allocation on GPU
        short *dev_chA_data;
        short *dev_chB_data;
        float *dev_sq_data;
public:
        void tearDown(){
        }
        void test_power_kernel_v1_no_background(){
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
                sq_data = new float[3]{-1, -2, -3};
                expected_sq_data = new float[3]{(float)(1 + 17 + 49 + 104) / 4,
                                (float)(5 + 25 + 65 + 125) / 4,
                                (float)(9 + 37 + 81 + 148) / 4};

                GPU::allocate_memory_on_gpu(&dev_chA_data, &dev_chB_data, &dev_sq_data);
                GPU::power_kernel(
                        chA_data,
                        chB_data,
                        sq_data,
                        &dev_chA_data,
                        &dev_chB_data,
                        &dev_sq_data
                        );
                GPU::free_memory_on_gpu(&dev_chA_data, &dev_chB_data, &dev_sq_data);

                // Compare
                for (int i(0); i < 3; i++) {
                        CPPUNIT_ASSERT_EQUAL(expected_sq_data[i], sq_data[i]);
                }

                // Or compare using file
                // float** arr_to_dump = new float*[2];
                // arr_to_dump[1] = expected_sq_data;
                // arr_to_dump[0] = sq_data;
                // dump_arrays_to_file(
                //         arr_to_dump,
                //         2, 4,
                //         "./test/test_bin/dump-gpu-example.txt",
                //         "#Real\t#Expected");

                delete[] chA_data;
                delete[] chB_data;
                delete[] sq_data;
                delete[] expected_sq_data;
        }

        // void test_power_kernel_v2_const_background(){

        //         chA_data = new short[12]{1, 2, 3, 4, 5, 6, 7, 9, 10, 10, 10, 10};
        //         chB_data = new short[12]{0, 1, 2, 3, 4, 5, 6, 7, 9, 9, 9, 9};
        //         sq_data = new float[4]{-1, -2, -3, -4};
        //         short chA_const_background = 1;
        //         short chB_const_background = 0;
        //         expected_sq_data = new unsigned int[9]{0, 2, 8, 18, 32, 50, 72, 113, 162};

        //         CPU::power_kernel(
        //                 chA_data, chB_data, sq_data,
        //                 chA_const_background, chB_const_background,
        //                 NO_POINTS, no_threads
        //                 );
        //         for (int i(0); i < 9; i++) {
        //                 CPPUNIT_ASSERT_EQUAL(expected_sq_data[i], sq_data[i]);
        //         }
        // }

        // void test_power_kernel_v3_background(){
        //         no_threads = 1;

        //         chA_data = new short[9]{1, 2, 3, 4, 5, 6, 7, 9, 10};
        //         chB_data = new short[9]{0, 1, 2, 3, 4, 5, 6, 7, 9};
        //         short *chA_const_background = new short[9]{1, 2, 3, 4, 5, 6, 7, 9, 10};
        //         short *chB_const_background = new short[9]{0,0,0,0,0,0,0,0,0};
        //         short* expected_chA_data = new short[9]{0,0,0,0,0,0,0,0,0};
        //         short* expected_chB_data = new short[9]{0, 1, 2, 3, 4, 5, 6, 7, 9};
        //         expected_sq_data = new unsigned int[9]{0, 1, 4, 9, 16, 25, 36, 49, 81};

        //         CPU::power_kernel(
        //                 chA_data, chB_data, sq_data,
        //                 chA_const_background, chB_const_background,
        //                 NO_POINTS, no_threads
        //                 );
        //         for (int i(0); i < 9; i++) {
        //                 CPPUNIT_ASSERT_EQUAL(expected_sq_data[i], sq_data[i]);
        //                 CPPUNIT_ASSERT_EQUAL(expected_chA_data[i], chA_data[i]);
        //                 CPPUNIT_ASSERT_EQUAL(expected_chB_data[i], chB_data[i]);
        //         }
        // }
};
CPPUNIT_TEST_SUITE_REGISTRATION( PowerGpuTest );
