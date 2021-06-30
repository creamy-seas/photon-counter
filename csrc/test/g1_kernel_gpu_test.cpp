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
    // CPPUNIT_TEST( test_g1_prepare_fftw_plan );
    // CPPUNIT_TEST( test_allocate_memory );
    CPPUNIT_TEST( test_g1_kernel );

    CPPUNIT_TEST_SUITE_END();
private:
    const static int N = 501;

    short* chA_data = new short[N];
    short* chB_data = new short[N];
    double** data_out;

    short *g1_test_data = new short[N];
    double *g1_expected_biased_normalisation = new double[N];
    double *g1_expected_unbiased_normalisation = new double[N];

public:
    void setUp() {
        // Auxillary arrays used toload in data
        short *_aux_arr_1[1] = {g1_test_data};
        load_arrays_from_file(_aux_arr_1,
                              "./test/test_files/g1_test_data.txt",
                              1, N);
        for (int i(0); i < N; i++) {
            chA_data[i] = g1_test_data[i];
            chB_data[i] = g1_test_data[i];
        }

        double *_aux_arr_2[1] = {g1_expected_unbiased_normalisation};
        load_arrays_from_file(_aux_arr_2,
                              "./test/test_files/g1_expected_unbiased_normalisation.txt",
                              1, N);
        _aux_arr_2[0] = {g1_expected_biased_normalisation};
        load_arrays_from_file(_aux_arr_2,
                              "./test/test_files/g1_expected_biased_normalisation.txt",
                              1, N);
    }
    void tearDown() {
        delete[] chA_data; delete[] chB_data;
        delete[] g1_expected_unbiased_normalisation;
    }

    void test_g1_prepare_fftw_plan() {
        cufftHandle *plans_forward; cufftHandle *plans_backward;

        G1::GPU::g1_prepare_fftw_plan(plans_forward, plans_backward);
    }

    void test_allocate_memory() {
        short *chA_data(0), *chB_data(0);
        G1::GPU::allocate_memory(&chA_data, &chB_data);

        CPPUNIT_ASSERT(chA_data != 0);
        CPPUNIT_ASSERT(chB_data != 0);
    }

    void test_g1_kernel() {
        short *chA_data, *chB_data;
        double **data_out;

        cufftHandle *plans_forward(0); cufftHandle *plans_backward(0);
        G1::GPU::g1_prepare_fftw_plan(plans_forward, plans_backward);

        CPPUNIT_ASSERT_MESSAGE("Failed to create plans", plans_forward != 0);
        CPPUNIT_ASSERT_MESSAGE("Failed to create plans", plans_backward != 0);

        G1::GPU::g1_kernel(chA_data, chB_data, data_out, plans_forward, plans_backward);
    }
};
CPPUNIT_TEST_SUITE_REGISTRATION( G1KernelGpuTest );
