#include <limits.h>
#include <string>
#include <cppunit/extensions/HelperMacros.h>
#include <cppunit/Exception.h>

#include <iostream>
using namespace std;

#include "power_kernel.hpp"
#include "utils.hpp"

#ifdef TESTENV
#define NO_POINTS 9
#endif

class PowerKernelGpuUtilsTest : public CppUnit::TestFixture {

    // Macro for generating suite
    CPPUNIT_TEST_SUITE( PowerKernelGpuUtilsTest );

    // Population with tests
    CPPUNIT_TEST( test_check_power_kernel_parameters );
    CPPUNIT_TEST( test_allocate_memory );
    CPPUNIT_TEST( test_allocate_memory_bad_no_streams );

    CPPUNIT_TEST_SUITE_END();

public:
    // These will hold the address of arrays allocated on GPU or locked CPU memory
    short *chA_data; short *chB_data;
    short ***gpu_in;
    long ***gpu_out;
    long ***cpu_out;

    void test_check_power_kernel_parameters(){
        GPU::check_power_kernel_parameters();
        GPU::check_power_kernel_parameters(true);
        CPPUNIT_ASSERT_EQUAL(GPU::fetch_power_kernel_blocks(), 3);
        CPPUNIT_ASSERT_EQUAL_MESSAGE(
            "R_POINTS_PER_CHUNK=2 specified in Makefile!",
            GPU::fetch_power_kernel_threads(), 2);
    }

    void test_allocate_memory_bad_no_streams(){
        const int no_streams = 3;
        CPPUNIT_ASSERT_THROW_MESSAGE("Allocation should have failed as no of streams does not fit into number of chunks",
                                     GPU::allocate_memory(&chA_data, &chB_data, &gpu_in, &gpu_out, &cpu_out, no_streams),
                                     std::runtime_error);
    }

    void test_allocate_memory(){
        const int no_streams = 2;

        GPU::allocate_memory(&chA_data, &chB_data, &gpu_in, &gpu_out, &cpu_out, no_streams);

        // Check memory has been allocated on GPU i.e. the pointers now hold the addresses
        CPPUNIT_ASSERT_MESSAGE("Failed gpu_in memory allocation - not set", gpu_in[0][CHA] != 0);
        CPPUNIT_ASSERT_MESSAGE("Failed gpu_in memory allocation - not set", gpu_in[0][CHB] != 0);
        CPPUNIT_ASSERT_MESSAGE("Failed gpu_in memory allocation - not set", gpu_in[1][CHA] != 0);
        CPPUNIT_ASSERT_MESSAGE("Failed gpu_in memory allocation - not set", gpu_in[1][CHB] != 0);
        CPPUNIT_ASSERT_MESSAGE("Failed gpu_in memory allocation - duplicate address", gpu_in[0][CHA] != gpu_in[0][CHB]);
        CPPUNIT_ASSERT_MESSAGE("Failed gpu_in memory allocation - duplicate address", gpu_in[1][CHA] != gpu_in[1][CHB]);

        CPPUNIT_ASSERT(gpu_out[0][CHA] != 0);
        CPPUNIT_ASSERT(gpu_out[0][CHB] != 0);
        CPPUNIT_ASSERT(gpu_out[0][CHASQ] != 0);
        CPPUNIT_ASSERT(gpu_out[0][CHBSQ] != 0);
        CPPUNIT_ASSERT(gpu_out[1][CHA] != 0);
        CPPUNIT_ASSERT(gpu_out[1][CHB] != 0);
        CPPUNIT_ASSERT(gpu_out[1][CHASQ] != 0);
        CPPUNIT_ASSERT(gpu_out[1][CHBSQ] != 0);

        CPPUNIT_ASSERT(cpu_out != 0);

        GPU::free_memory(chA_data, chB_data, gpu_in, gpu_out, cpu_out, no_streams);
    }

};
CPPUNIT_TEST_SUITE_REGISTRATION( PowerKernelGpuUtilsTest );
