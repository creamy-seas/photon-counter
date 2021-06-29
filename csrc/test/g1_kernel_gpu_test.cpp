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
    CPPUNIT_TEST( test_g1_prepare_fftw_plan );

    CPPUNIT_TEST_SUITE_END();
private:
    const static int N = 501;
    const int tau_points = 50;

    short* chA_data;
    short* chB_data;
    double** data_out;

    short *g1_test_data = new short[N];
    double *g1_expected_biased_normalisation = new double[N];
    double *g1_expected_unbiased_normalisation = new double[N];

public:
    void setUp() {
        short *_aux_arr_1[1] = {g1_test_data};
        load_arrays_from_file(_aux_arr_1,
                              "./test/test_files/g1_test_data.txt",
                              1, N);

        double *_aux_arr_2[1] = {g1_expected_unbiased_normalisation};
        load_arrays_from_file(_aux_arr_2,
                              "./test/test_files/g1_expected_unbiased_normalisation.txt",
                              1, N);
        _aux_arr_2[0] = {g1_expected_biased_normalisation};
        load_arrays_from_file(_aux_arr_2,
                              "./test/test_files/g1_expected_biased_normalisation.txt",
                              1, N);

        chA_data = new short[N];
        chB_data = new short[N];
        for (int i(0); i < N; i++) {
            chA_data[i] = g1_test_data[i];
            chB_data[i] = g1_test_data[i];
        }

        data_out = new double*[G1::no_outputs];
        for (int i(0); i < G1::no_outputs; i++)
            data_out[i] = new double[tau_points]();
    }
    void tearDown() {
        delete[] chA_data;
        delete[] chB_data;

        for (int i(0); i < G1::no_outputs; i++)
            delete[] data_out[i];
        delete[] data_out;
        delete[] g1_expected_unbiased_normalisation;
    }
    void beforeEach() {
        for (int i(0); i < G1::no_outputs; i++) {
            for (int t(0); t < tau_points; t++)
                data_out[i][t] = 0;
        }
    }

    void test_g1_prepare_fftw_plan() {
        cufftHandle *plans_forward; cufftHandle *plans_backward;
        G1::GPU::g1_prepare_fftw_plan(plans_forward, plans_backward);
    }
};
CPPUNIT_TEST_SUITE_REGISTRATION( G1KernelGpuTest );
