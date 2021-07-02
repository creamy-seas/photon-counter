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
    const int tau_points = 50;
    short* chA_data = new short[G1_DIGITISER_POINTS];
    short* chB_data = new short[G1_DIGITISER_POINTS];

    short *g1_test_data = new short[G1_DIGITISER_POINTS];
    float *g1_expected_biased_normalisation = new float[G1_DIGITISER_POINTS];
    float *g1_expected_unbiased_normalisation = new float[G1_DIGITISER_POINTS];

public:
    void setUp() {
        // Auxillary arrays used to load in data from specific columns
        short *_aux_arr_1[1] = {g1_test_data};
        load_arrays_from_file(_aux_arr_1, "./test/test_files/g1_test_data.txt", 1, G1_DIGITISER_POINTS);
        for (int i(0); i < G1_DIGITISER_POINTS; i++) {
            chA_data[i] = g1_test_data[i];
            chB_data[i] = g1_test_data[i];
        }

        float *_aux_arr_2[1] = {g1_expected_unbiased_normalisation};
        load_arrays_from_file(_aux_arr_2, "./test/test_files/g1_expected_unbiased_normalisation.txt",
                              1, G1_DIGITISER_POINTS);
        _aux_arr_2[0] = {g1_expected_biased_normalisation};
        load_arrays_from_file(_aux_arr_2, "./test/test_files/g1_expected_biased_normalisation.txt",
                              1, G1_DIGITISER_POINTS);
    }
    void tearDown() {
        delete[] chA_data; delete[] chB_data;
        delete[] g1_expected_unbiased_normalisation;
    }

    void test_g1_kernel() {
        // Create plans
        cufftHandle *plans_forward(0); cufftHandle *plans_backward(0);
        G1::GPU::g1_prepare_fftw_plan(plans_forward, plans_backward);

        // Allocation of memory
        cufftReal **gpu_inout; cufftComplex **gpu_aux; float **cpu_inout;
        G1::GPU::allocate_memory(gpu_inout, gpu_aux, cpu_inout);

        G1::GPU::g1_kernel(chA_data, chB_data,
                           gpu_inout, gpu_aux, cpu_inout,
                           plans_forward, plans_backward);

        for (int tau(0); tau < tau_points; tau++) {
            CPPUNIT_ASSERT_DOUBLES_EQUAL_MESSAGE("Error on tau=" + std::to_string(tau),
                                                 g1_expected_biased_normalisation[tau],
                                                 cpu_inout[CHAG1][tau],
                                                 0.05);
            CPPUNIT_ASSERT_DOUBLES_EQUAL_MESSAGE("Error on tau=" + std::to_string(tau),
                                                 g1_expected_biased_normalisation[tau],
                                                 cpu_inout[CHBG1][tau],
                                                 0.05);
        }

        G1::GPU::free_memory(gpu_inout, gpu_aux, cpu_inout);
    }
};
// CPPUNIT_TEST_SUITE_REGISTRATION( G1KernelGpuTest );
