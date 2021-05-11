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
        // Default values that must change once address is assigned. This will hold the address of the arrays allocated on the GPU
        // These will change!
        short* dev_chA_data = 0;
        short* dev_chB_data = 0;
        double* dev_chA_out = 0;
        double* dev_chB_out = 0;
        double* dev_chAsq_out = 0;
        double* dev_chBsq_out = 0;
        double* dev_sq_out = 0;

        // These values will hold the addresses of the above pointers
        // These will not change!
        void* dev_chA_data_addr = &dev_chA_data;
        void* dev_chB_data_addr = &dev_chB_data;
        void* dev_chA_out_addr = &dev_chA_out;
        void* dev_chB_out_addr = &dev_chB_out;
        void* dev_chAsq_out_addr = &dev_chAsq_out;
        void* dev_chBsq_out_addr = &dev_chBsq_out;
        void* dev_sq_out_addr = &dev_sq_out;

        GPU::allocate_memory_on_gpu(&dev_chA_data, &dev_chB_data, &dev_chA_out, &dev_chB_out,
                                    &dev_chAsq_out, &dev_chBsq_out, &dev_sq_out);

        // Check memory has been allocated
        CPPUNIT_ASSERT(dev_chA_data != 0);
        CPPUNIT_ASSERT(dev_chB_data != 0);
        CPPUNIT_ASSERT(dev_chA_out != 0);
        CPPUNIT_ASSERT(dev_chB_out != 0);
        CPPUNIT_ASSERT(dev_chAsq_out != 0);
        CPPUNIT_ASSERT(dev_chBsq_out != 0);
        CPPUNIT_ASSERT(dev_sq_out != 0);

        GPU::free_memory_on_gpu(&dev_chA_data, &dev_chB_data, &dev_chA_out, &dev_chB_out,
                                &dev_chAsq_out, &dev_chBsq_out, &dev_sq_out);

        // Check address has stayed the same
        CPPUNIT_ASSERT_EQUAL(dev_chA_data_addr, (void *)&dev_chA_data);
        CPPUNIT_ASSERT_EQUAL(dev_chB_data_addr, (void *)&dev_chB_data);
        CPPUNIT_ASSERT_EQUAL(dev_chA_out_addr, (void *)&dev_chA_out);
        CPPUNIT_ASSERT_EQUAL(dev_chB_out_addr, (void *)&dev_chB_out);
        CPPUNIT_ASSERT_EQUAL(dev_chAsq_out_addr, (void *)&dev_chAsq_out);
        CPPUNIT_ASSERT_EQUAL(dev_chBsq_out_addr, (void *)&dev_chBsq_out);
        CPPUNIT_ASSERT_EQUAL(dev_sq_out_addr, (void *)&dev_sq_out);
    }
};
CPPUNIT_TEST_SUITE_REGISTRATION( PowerKernelGpuUtilsTest );
