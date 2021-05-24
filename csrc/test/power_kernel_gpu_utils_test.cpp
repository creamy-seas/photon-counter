#include <limits.h>
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
    CPPUNIT_TEST( test_allocate_memory );

    CPPUNIT_TEST_SUITE_END();

public:
    void test_fetch_kernel_parameters(){
        GPU::PowerKernelParameters kp = GPU::fetch_kernel_parameters();
        // Parameters for test set in Makefile
        CPPUNIT_ASSERT_EQUAL(8, kp.r_points);
        CPPUNIT_ASSERT_EQUAL(3, kp.np_points);
    }


    void test_allocate_memory(){
        const int no_streams = 2;

        // Default values that must change once address is assigned. This will hold the address of the arrays allocated on the GPU
        // These will change!
        short *chA_data; short *chB_data;

        short *gpu_chA_data0(0), *gpu_chB_data0(0), **gpu_in0[2] = {&gpu_chA_data0, &gpu_chB_data0};
        short *gpu_chA_data1(0), *gpu_chB_data1(0), **gpu_in1[2] = {&gpu_chA_data1, &gpu_chB_data1};
        short ***gpu_in[2] = {gpu_in0, gpu_in1};

        long *gpu_chA_out0(0), *gpu_chB_out0(0), *gpu_chAsq_out0(0), *gpu_chBsq_out0(0);
        long **gpu_out0[4] = {&gpu_chA_out0, &gpu_chB_out0, &gpu_chAsq_out0, &gpu_chBsq_out0};
        long *gpu_chA_out1(0), *gpu_chB_out1(0), *gpu_chAsq_out1(0), *gpu_chBsq_out1(0);
        long **gpu_out1[4] = {&gpu_chA_out1, &gpu_chB_out1, &gpu_chAsq_out1,&gpu_chBsq_out1};
        long ***gpu_out[2] = {gpu_out0, gpu_out1};

        long *cpu_chA_out_s0, *cpu_chB_out_s0, *cpu_chAsq_out_s0, *cpu_chBsq_out_s0;
        long **cpu_out0[4] = {&cpu_chA_out_s0, &cpu_chB_out_s0, &cpu_chAsq_out_s0, &cpu_chBsq_out_s0};
        long *cpu_chA_out_s1, *cpu_chB_out_s1, *cpu_chAsq_out_s1, *cpu_chBsq_out_s1;
        long **cpu_out1[4] = {&cpu_chA_out_s1, &cpu_chB_out_s1, &cpu_chAsq_out_s1, &cpu_chBsq_out_s1};
        long ***cpu_out[2] = {cpu_out0, cpu_out1};

        GPU::allocate_memory(&chA_data, &chB_data, gpu_in, gpu_out, cpu_out, no_streams);

        // Check memory has been allocated on GPU i.e. the pointers now hold the addresses
        CPPUNIT_ASSERT(gpu_in[0][CHA] != 0);
        CPPUNIT_ASSERT(gpu_in[0][CHB] != 0);
        CPPUNIT_ASSERT(gpu_in[1][CHA] != 0);
        CPPUNIT_ASSERT(gpu_in[1][CHB] != 0);

        CPPUNIT_ASSERT(gpu_out[0][CHA] != 0);
        CPPUNIT_ASSERT(gpu_out[0][CHB] != 0);
        CPPUNIT_ASSERT(gpu_out[0][CHASQ] != 0);
        CPPUNIT_ASSERT(gpu_out[0][CHBSQ] != 0);
        CPPUNIT_ASSERT(gpu_out[1][CHA] != 0);
        CPPUNIT_ASSERT(gpu_out[1][CHB] != 0);
        CPPUNIT_ASSERT(gpu_out[1][CHASQ] != 0);
        CPPUNIT_ASSERT(gpu_out[1][CHBSQ] != 0);

        CPPUNIT_ASSERT(cpu_out != 0);

        GPU::free_memory(&chA_data, &chB_data, gpu_in, gpu_out, cpu_out, no_streams);
    }

};
CPPUNIT_TEST_SUITE_REGISTRATION( PowerKernelGpuUtilsTest );
