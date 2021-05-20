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
    CPPUNIT_TEST( test_allocate_memory_allocation_v1 );

    CPPUNIT_TEST_SUITE_END();

public:
    void test_fetch_kernel_parameters(){
        GPU::PowerKernelParameters kp = GPU::fetch_kernel_parameters();
        // Parameters for test set in Makefile
        CPPUNIT_ASSERT_EQUAL(4, kp.r_points);
        CPPUNIT_ASSERT_EQUAL(3, kp.np_points);
    }

    void test_allocate_memory_allocation_v1(){
        // This will hold the address of the arrays allocated on the GPU
        // These will change!
        short* gpu_chA_data = 0; short* gpu_chB_data = 0;
        short **gpu_in[2] = {&gpu_chA_data, &gpu_chB_data};

        double* gpu_chA_out = 0; double* gpu_chB_out = 0; double* gpu_chAsq_out = 0; double* gpu_chBsq_out = 0;
        double **gpu_out[4] = {&gpu_chA_out,
                               &gpu_chB_out,
                               &gpu_chAsq_out,
                               &gpu_chBsq_out};

        // These values will hold the addresses of the above pointers
        // These will not change!
        void* gpu_chA_data_addr = &gpu_chA_data;
        void* gpu_chB_data_addr = &gpu_chB_data;
        void* gpu_chA_out_addr = &gpu_chA_out;
        void* gpu_chB_out_addr = &gpu_chB_out;
        void* gpu_chAsq_out_addr = &gpu_chAsq_out;
        void* gpu_chBsq_out_addr = &gpu_chBsq_out;

        GPU::V1::allocate_memory(gpu_in, gpu_out);

        // Check memory has been allocated
        CPPUNIT_ASSERT(gpu_chA_data != 0);
        CPPUNIT_ASSERT(gpu_chB_data != 0);
        CPPUNIT_ASSERT(gpu_chA_out != 0);
        CPPUNIT_ASSERT(gpu_chB_out != 0);
        CPPUNIT_ASSERT(gpu_chAsq_out != 0);
        CPPUNIT_ASSERT(gpu_chBsq_out != 0);

        GPU::V1::free_memory(gpu_in, gpu_out);

        // Check address has stayed the same
        CPPUNIT_ASSERT_EQUAL(gpu_chA_data_addr, (void *)&gpu_chA_data);
        CPPUNIT_ASSERT_EQUAL(gpu_chB_data_addr, (void *)&gpu_chB_data);
        CPPUNIT_ASSERT_EQUAL(gpu_chA_out_addr, (void *)&gpu_chA_out);
        CPPUNIT_ASSERT_EQUAL(gpu_chB_out_addr, (void *)&gpu_chB_out);
        CPPUNIT_ASSERT_EQUAL(gpu_chAsq_out_addr, (void *)&gpu_chAsq_out);
        CPPUNIT_ASSERT_EQUAL(gpu_chBsq_out_addr, (void *)&gpu_chBsq_out);
    }


    void test_allocate_memory_allocation_v2(){
        // Default values that must change once address is assigned. This will hold the address of the arrays allocated on the GPU
        // These will change!
        short** gpu_in0[2] = {0, 0};
        short** gpu_in1[2] = {0, 0};
        double **gpu_out0[4] = {0, 0, 0, 0};
        double **gpu_out1[4] = {0, 0, 0, 0};
        double **cpu_out0[4] = {0, 0, 0, 0};
        double **cpu_out1[4] = {0, 0, 0, 0};

        GPU::V2::allocate_memory(gpu_in0, gpu_in1, gpu_out0, gpu_out1, cpu_out0, cpu_out1);

        // Check memory has been allocated on GPU i.e. the pointers now hold the addresses
        CPPUNIT_ASSERT(gpu_in0[CHA] != 0);
        CPPUNIT_ASSERT(gpu_in0[CHB] != 0);
        CPPUNIT_ASSERT(gpu_in1[CHA] != 0);
        CPPUNIT_ASSERT(gpu_in1[CHB] != 0);

        CPPUNIT_ASSERT(gpu_out0[CHA] != 0);
        CPPUNIT_ASSERT(gpu_out0[CHB] != 0);
        CPPUNIT_ASSERT(gpu_out0[CHASQ] != 0);
        CPPUNIT_ASSERT(gpu_out0[CHBSQ] != 0);
        CPPUNIT_ASSERT(gpu_out1[CHA] != 0);
        CPPUNIT_ASSERT(gpu_out1[CHB] != 0);
        CPPUNIT_ASSERT(gpu_out1[CHASQ] != 0);
        CPPUNIT_ASSERT(gpu_out1[CHBSQ] != 0);

        CPPUNIT_ASSERT(cpu_out0[CHA] != 0);
        CPPUNIT_ASSERT(cpu_out0[CHB] != 0);
        CPPUNIT_ASSERT(cpu_out0[CHASQ] != 0);
        CPPUNIT_ASSERT(cpu_out0[CHBSQ] != 0);
        CPPUNIT_ASSERT(cpu_out1[CHA] != 0);
        CPPUNIT_ASSERT(cpu_out1[CHB] != 0);
        CPPUNIT_ASSERT(cpu_out1[CHASQ] != 0);
        CPPUNIT_ASSERT(cpu_out1[CHBSQ] != 0);

        // GPU::V2::free_memory(gpu_in0, gpu_in1, gpu_out0, gpu_out1, cpu_out0, cpu_out1);
    }

};
CPPUNIT_TEST_SUITE_REGISTRATION( PowerKernelGpuUtilsTest );
