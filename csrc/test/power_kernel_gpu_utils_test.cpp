#include <string>
#include <cppunit/extensions/HelperMacros.h>
#include <cppunit/Exception.h>
// #include <iostream>
// using namespace std;

#include "power_kernel.hpp"
#include "utils.hpp"

#ifdef TESTENV
#define NO_POINTS 9
#endif

class PowerKernelGpuUtilsTest : public CppUnit::TestFixture {

    // Macro for generating suite
    CPPUNIT_TEST_SUITE( PowerKernelGpuUtilsTest );

    // Population with tests
    CPPUNIT_TEST( test_fetch_kernel_parameters );
    CPPUNIT_TEST( test_allocate_memory_allocation );

    CPPUNIT_TEST_SUITE_END();

public:
    void test_fetch_kernel_parameters(){
        GPU::PowerKernelParameters kp = GPU::fetch_kernel_parameters();
        // Parameters for test set in Makefile
        CPPUNIT_ASSERT_EQUAL(4, kp.r_points);
        CPPUNIT_ASSERT_EQUAL(3, kp.np_points);
        CPPUNIT_ASSERT_EQUAL(std::string("int"), kp.processing_array_type);
    }

    void test_allocate_memory_allocation(){
        short* dev_chA_data = 0;
        short* dev_chB_data = 0;
        float* dev_sq_data = 0;

        void* dev_chA_data_addr = &dev_chA_data;
        void* dev_chB_data_addr = &dev_chB_data;
        void* dev_sq_data_addr = &dev_sq_data;

        GPU::allocate_memory_on_gpu(&dev_chA_data, &dev_chB_data, &dev_sq_data);

        // Check memory has been allocated
        CPPUNIT_ASSERT(dev_chA_data != 0);
        CPPUNIT_ASSERT(dev_chB_data != 0);
        CPPUNIT_ASSERT(dev_sq_data != 0);

        GPU::free_memory_on_gpu(&dev_chA_data, &dev_chB_data, &dev_sq_data);

        // Check address has stayed the same
        CPPUNIT_ASSERT_EQUAL(dev_chA_data_addr, (void *)&dev_chA_data);
        CPPUNIT_ASSERT_EQUAL(dev_chB_data_addr, (void *)&dev_chB_data);
        CPPUNIT_ASSERT_EQUAL(dev_sq_data_addr, (void *)&dev_sq_data);
        }
};
CPPUNIT_TEST_SUITE_REGISTRATION( PowerKernelGpuUtilsTest );
