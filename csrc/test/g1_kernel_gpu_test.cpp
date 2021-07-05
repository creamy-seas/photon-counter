#include <cppunit/extensions/HelperMacros.h>
#include <cppunit/Exception.h>
#include <string>
#include <fftw3.h> // for all fttw related items

#include <cufft.h>
#include "utils.hpp"
#include "g1_kernel.hpp"

class G1KernelGpuTest : public CppUnit::TestFixture {

    // Macro for generating suite
    CPPUNIT_TEST_SUITE( G1KernelGpuTest );

    // Population with tests
    // CPPUNIT_TEST_EXCEPTION( 🐙, CppUnit::Exception );
    CPPUNIT_TEST( test_g1_kernel );

    CPPUNIT_TEST_SUITE_END();
private:
    const int tau_points = 200;

    short* chA_data = new short[G1_DIGITISER_POINTS];
    short* chB_data = new short[G1_DIGITISER_POINTS];

    double *chA_g1 = new double[G1_DIGITISER_POINTS];
    double *chB_g1 = new double[G1_DIGITISER_POINTS];
    double *sq_g1 = new double[G1_DIGITISER_POINTS];

public:
    void setUp() {
        // Auxillary arrays used to load in data from specific columns
        short *_aux_arr_1[2] = {chA_data, chB_data};
        load_arrays_from_file(_aux_arr_1, "./test/test_files/g1_test_data.txt", 2, G1_DIGITISER_POINTS);

        double *_aux_arr_2[3] = {chA_g1, chB_g1, sq_g1};
        load_arrays_from_file(_aux_arr_2, "./test/test_files/g1_expected_biased_normalisation.txt",
                              3, tau_points);
    }
    void tearDown() {
        delete[] chA_data; delete[] chB_data;
        delete[] chA_g1; delete[]  chB_g1; delete[] sq_g1;
    }

    void test_g1_kernel() {
        // Create plans
        cufftHandle *plans_forward(0); cufftHandle *plans_backward(0);
        G1::GPU::g1_prepare_fftw_plan(plans_forward, plans_backward);

        // Allocation of memory
        short **gpu_raw_data; cufftReal **gpu_inout;
        float **gpu_pp_aux; cufftComplex **gpu_fftw_aux; float *gpu_mean, *gpu_variance;
        float **cpu_inout;
        G1::GPU::allocate_memory(gpu_raw_data, gpu_inout, cpu_inout, gpu_pp_aux, gpu_fftw_aux, gpu_mean, gpu_variance);

        G1::GPU::g1_kernel(chA_data, chB_data,
                           gpu_inout, gpu_fftw_aux, cpu_inout,
                           plans_forward, plans_backward);

        for (int tau(0); tau < tau_points; tau++) {
            CPPUNIT_ASSERT_DOUBLES_EQUAL_MESSAGE("CHA Error on tau=" + std::to_string(tau),
                                                 chA_g1[tau],
                                                 cpu_inout[CHAG1][tau],
                                                 0.05);
            CPPUNIT_ASSERT_DOUBLES_EQUAL_MESSAGE("CHB Error on tau=" + std::to_string(tau),
                                                 chB_g1[tau],
                                                 cpu_inout[CHBG1][tau],
                                                 0.05);
            CPPUNIT_ASSERT_DOUBLES_EQUAL_MESSAGE("SQ Error on tau=" + std::to_string(tau),
                                                 sq_g1[tau],
                                                 cpu_inout[SQG1][tau],
                                                 0.05);
        }

        G1::GPU::free_memory(gpu_raw_data, gpu_inout, cpu_inout, gpu_pp_aux, gpu_fftw_aux, gpu_mean, gpu_variance);
    }
};
CPPUNIT_TEST_SUITE_REGISTRATION( G1KernelGpuTest );
