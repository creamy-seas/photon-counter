#include <string>
#include <cppunit/extensions/HelperMacros.h>
#include <cppunit/Exception.h>

#include "power_kernel.hpp"

#ifdef TESTENV
#define NO_POINTS 9
#endif

class PowerGpuTest : public CppUnit::TestFixture {

        // Macro for generating suite
        CPPUNIT_TEST_SUITE( PowerGpuTest );

        // Population with tests
        // CPPUNIT_TEST_EXCEPTION( 🐙, CppUnit::Exception );
        CPPUNIT_TEST( test_power_kernel );
        CPPUNIT_TEST( test_fetch_kernel_parameters );

        CPPUNIT_TEST_SUITE_END();
private:
        int no_threads;

        short* chA_data;
        short* chB_data;
        unsigned int* sq_data = new unsigned int[9];
        unsigned int* expected_sq_data = new unsigned int[9];
public:
        void tearDown(){
                delete chA_data;
                delete chB_data;
                delete sq_data;
                delete expected_sq_data;
        }

        void test_power_kernel(){
                // no_threads = 4;
                // chA_data = new short[9]{1, 2, 3, 4, 5, 6, 7, 9, 10};
                // chB_data = new short[9]{0, 1, 2, 3, 4, 5, 6, 7, 9};
                // expected_sq_data = new unsigned int[9]{1, 5, 13, 25, 41, 61, 85, 130, 181};

                float expected_result = 5;
                CPPUNIT_ASSERT_EQUAL(expected_result, GPU::power_kernel(1, 2));
        }

        void test_fetch_kernel_parameters(){
                GPU::PowerKernelParameters kp = GPU::fetch_kernel_parameters();
                CPPUNIT_ASSERT_EQUAL(1000, kp.r_points);
                CPPUNIT_ASSERT_EQUAL(100, kp.np_points);
                CPPUNIT_ASSERT_EQUAL(std::string("int"), kp.processing_array_type);
        }

        void test_power_kernel_v1_no_background_runner(){
                short *chA_data;
                short *chB_data;
                float *sq_data;

                short *dev_chA_data;
                short *dev_chB_data;
                float *dev_sq_data;

                GPU::allo

                        GPU::power_kernel(
                                chA_data,
                                chB_data,
                                sq_data,
                                dev_chA_data,
                                dev_chB_data,
                                dev_sq_data
                                )

                        }
};
CPPUNIT_TEST_SUITE_REGISTRATION( PowerGpuTest );
