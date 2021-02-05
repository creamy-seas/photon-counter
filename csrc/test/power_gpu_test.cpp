#include <cppunit/extensions/HelperMacros.h>
#include <cppunit/Exception.h>

#include "power_kernel.hpp"
#include "utils.hpp"

#ifdef TESTENV
#define NO_POINTS 9
#endif

class PowerGpuTest : public CppUnit::TestFixture {

        // Macro for generating suite
        CPPUNIT_TEST_SUITE( PowerGpuTest );

        // Population with tests
        // CPPUNIT_TEST_EXCEPTION( 🐙, CppUnit::Exception );
        CPPUNIT_TEST( test_power_kernel_v1_no_background_runner );

        CPPUNIT_TEST_SUITE_END();
private:
        short *chA_data;
        short *chB_data;
        float *sq_data = new float[4];
        float *expected_sq_data = new float[4];
public:
        void tearDown(){
                delete chA_data;
                delete chB_data;
                delete sq_data;
                delete expected_sq_data;
        }
        void test_power_kernel_v1_no_background_runner(){
                /*chA
                 * 1  2   3   4    -> main axis (4)
                 * 5  6   7   8
                 * 9  10  11  12
                 * |
                 * repetition axis (3)

                 * chB
                 * 0  1   0   1
                 * 1  0   1   0
                 * 2  2   2   2

                 * sq
                 * 1  5   9   17
                 * 26 36  50  64
                 * 85 104 125 148
                 */

                // Allocation on CPU
                chA_data = new short[12]{1,2, 3, 4, 6, 7, 8, 9, 10, 11, 12};
                chB_data = new short[12]{0, 1, 0, 1, 1, 0, 1, 0, 2, 2, 2, 2};
                sq_data = new float[4]{-1, -2, -3, -4};

                // Allocation on GPU
                short *dev_chA_data;
                short *dev_chB_data;
                float *dev_sq_data;

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
                expected_sq_data = new float[4]{(float)(1 + 26 + 85) / 3,
                                (float)(5 + 36 + 104) / 3,
                                (float)(9 + 50 + 125) / 3,
                                (float)(17 + 64 + 148) / 3};

                float** arr_to_dump = new float*[2];
                arr_to_dump[1] = expected_sq_data;
                arr_to_dump[0] = sq_data;

                dump_arrays_to_file(
                        arr_to_dump,
                        2, 4,
                        "./test/test_bin/dump-gpu-example.txt",
                        "#Real\t#Expected");
        }
};
CPPUNIT_TEST_SUITE_REGISTRATION( PowerGpuTest );
