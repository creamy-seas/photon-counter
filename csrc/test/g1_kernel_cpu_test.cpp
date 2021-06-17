#include <cppunit/extensions/HelperMacros.h>
#include <cppunit/Exception.h>
#include <string>
#include <fftw3.h> // for all fttw related items

#include "utils.hpp"
#include "g1_kernel.hpp"

class G1KernelCpuTest : public CppUnit::TestFixture {

    // Macro for generating suite
    CPPUNIT_TEST_SUITE( G1KernelCpuTest );

    // Population with tests
    // CPPUNIT_TEST_EXCEPTION( 🐙, CppUnit::Exception );
    CPPUNIT_TEST( test_direct_unbiased_normalisation);
    CPPUNIT_TEST( test_direct_unbiased_normalisation_2_threads );
    CPPUNIT_TEST( test_direct_biased_normalisation );
    CPPUNIT_TEST( test_fftw );

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
        short *aux_arr_1[1] = {g1_test_data};
        load_arrays_from_file(aux_arr_1,
                              "./test/test_files/g1_test_data.txt",
                              1, N);

        double *aux_arr_2[1] = {g1_expected_unbiased_normalisation};
        load_arrays_from_file(aux_arr_2,
                              "./test/test_files/g1_expected_unbiased_normalisation.txt",
                              1, N);
        aux_arr_2[0] = {g1_expected_biased_normalisation};
        load_arrays_from_file(aux_arr_2,
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

    void test_direct_unbiased_normalisation() {
        int no_threads = 1;

        G1::CPU::DIRECT::g1_kernel(chA_data, chB_data, data_out, tau_points, true, no_threads);

        for (int tau(0); tau < tau_points; tau++) {
            CPPUNIT_ASSERT_DOUBLES_EQUAL_MESSAGE("Error on tau=" + std::to_string(tau),
                                                 g1_expected_unbiased_normalisation[tau],
                                                 data_out[CHAG1][tau],
                                                 0.001);
        }
    }

    void test_direct_unbiased_normalisation_2_threads() {
        int no_threads = 2;

        G1::CPU::DIRECT::g1_kernel(chA_data, chB_data, data_out, tau_points, true, no_threads);

        for (int tau(0); tau < tau_points; tau++) {
            CPPUNIT_ASSERT_DOUBLES_EQUAL_MESSAGE("Error on tau=" + std::to_string(tau),
                                                 g1_expected_unbiased_normalisation[tau], data_out[CHAG1][tau], 0.001);
        }
    }

    void test_direct_biased_normalisation() {
        int no_threads = 1;

        G1::CPU::DIRECT::g1_kernel(chA_data, chB_data, data_out, tau_points, false, no_threads);

        for (int tau(0); tau < tau_points; tau++) {
            CPPUNIT_ASSERT_DOUBLES_EQUAL_MESSAGE("Error on tau=" + std::to_string(tau),
                                                 g1_expected_biased_normalisation[tau], data_out[CHAG1][tau], 0.001);
        }
    }

    void test_fftw() {

        // Generate plans
        int time_limit = 1;
        int no_threads = 8;
        G1::CPU::FFTW::g1_prepare_fftw_plan("./dump/test-fftw-plan", time_limit, no_threads);

        // Pre-kernel setup
        double **data_out_local; fftw_complex *aux_array;
        fftw_plan *plans_forward, *plans_backward;
        G1::CPU::FFTW::g1_allocate_memory(&data_out_local, &aux_array, "./dump/test-fftw-plan",
                                          &plans_forward,
                                          &plans_backward);

        // Test
        G1::CPU::FFTW::g1_kernel(chA_data, chB_data,
                                 data_out_local, aux_array,
                                 plans_forward, plans_backward);
        for (int tau(0); tau < tau_points; tau++) {
            CPPUNIT_ASSERT_DOUBLES_EQUAL_MESSAGE("Error on tau=" + std::to_string(tau),
                                                 g1_expected_biased_normalisation[tau],
                                                 data_out_local[CHAG1][tau], 0.05);
        }

        // Post kernel
        G1::CPU::FFTW::g1_free_memory(data_out_local, aux_array, plans_forward, plans_backward);
    }
};
CPPUNIT_TEST_SUITE_REGISTRATION( G1KernelCpuTest );
