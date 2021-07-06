#include <cppunit/extensions/HelperMacros.h>
#include <cppunit/Exception.h>
#include <cmath> // for absolute

#include "g1_kernel.hpp"
#include "utils.hpp"

class G1KernelUtilsTest : public CppUnit::TestFixture {

    // Macro for generating suite
    CPPUNIT_TEST_SUITE( G1KernelUtilsTest );

    // Population with tests
    // CPPUNIT_TEST_EXCEPTION( 🐙, CppUnit::Exception );
    CPPUNIT_TEST( test_on_cpu );
    CPPUNIT_TEST( test_on_gpu );

    CPPUNIT_TEST_SUITE_END();

private:
    const int N = G1_DIGITISER_POINTS;

    short* chA_data = new short[N];
    short* chB_data = new short[N];

    const double expected_mean[3] = {
        7.950247483262897,
        -4.430522148000995,
        44702389.40396311
    };
    const double expected_variance[3] = {
        22355718.64064403,
        22346587.927357536,
        799664705005884.9
    };

public:
    void setUp(){
        short *_aux_arr_1[2] = {chA_data, chB_data};
        load_arrays_from_file(_aux_arr_1, "./test/test_files/g1_test_data.txt", 2, N);
    }
    void tearDown(){
        delete[] chA_data;
        delete[] chB_data;
    }

    void test_on_cpu(){

        double mean_list[G1::no_outputs];
        double variance_list[G1::no_outputs];
        double chA_normalised[N]; double chB_normalised[N]; double sq_normalised[N];
        double *normalised_data[G1::no_outputs];
        normalised_data[CHAG1] = chA_normalised;
        normalised_data[CHBG1] = chB_normalised;
        normalised_data[SQG1] = sq_normalised;

        G1::CPU::preprocessor(chA_data, chB_data,
                              N,
                              mean_list, variance_list,
                              normalised_data);

        // 1% tolerance
        for (int i(0); i < 3; i++) {
            CPPUNIT_ASSERT_DOUBLES_EQUAL_MESSAGE(
                "Mean", expected_mean[i], mean_list[i],
                std::abs(0.01 * mean_list[i]));
            CPPUNIT_ASSERT_DOUBLES_EQUAL_MESSAGE(
                "Variance", expected_variance[i], variance_list[i],
                0.01 * expected_variance[i]);
        }
        for (int i(0); i < 100; i++) {
            CPPUNIT_ASSERT_DOUBLES_EQUAL_MESSAGE("Normalisation", chA_data[i] - expected_mean[CHAG1], normalised_data[CHAG1][i], 0.01);
            CPPUNIT_ASSERT_DOUBLES_EQUAL_MESSAGE(
                    "Normalisation", (double)chA_data[i] * chA_data[i] + chB_data[i] * chB_data[i] - expected_mean[SQG1],
                    normalised_data[SQG1][i], 0.01 * expected_mean[SQG1]);
        }
    }

    void test_on_gpu(){

        float mean_list[G1::no_outputs]; float variance_list[G1::no_outputs];
        float chA_normalised[N]; float chB_normalised[N]; float sq_normalised[N];
        float *normalised_data[G1::no_outputs];
        normalised_data[CHAG1] = chA_normalised;
        normalised_data[CHBG1] = chB_normalised;
        normalised_data[SQG1] = sq_normalised;

        G1::GPU::g1_memory memory = G1::GPU::allocate_memory(N);
        G1::check_g1_kernel_parameters(false);
        G1::GPU::preprocessor(
            N, chA_data, chB_data,
            memory.gpu_raw_data, reinterpret_cast<float**>(memory.gpu_inout),
            memory.gpu_pp_aux, memory.gpu_mean, memory.gpu_variance,
            mean_list, variance_list, normalised_data);

        // 1% tolerance
        for (int i(0); i < 3; i++) {
            CPPUNIT_ASSERT_DOUBLES_EQUAL_MESSAGE(
                "Mean", expected_mean[i], mean_list[i],
                std::abs(0.01 * mean_list[i]));
            CPPUNIT_ASSERT_DOUBLES_EQUAL_MESSAGE(
                "Variance", expected_variance[i], variance_list[i],
                0.01 * expected_variance[i]);
        }
        for (int i(0); i < 100; i++) {
            CPPUNIT_ASSERT_DOUBLES_EQUAL_MESSAGE("Normalisation", chA_data[i] - expected_mean[CHAG1], normalised_data[CHAG1][i], 0.01);
            CPPUNIT_ASSERT_DOUBLES_EQUAL_MESSAGE(
                "Normalisation", (double)chA_data[i] * chA_data[i] + chB_data[i] * chB_data[i] - expected_mean[SQG1],
                normalised_data[SQG1][i], 0.01 * expected_mean[SQG1]);
        }

        G1::GPU::free_memory(memory);
    }
};
CPPUNIT_TEST_SUITE_REGISTRATION( G1KernelUtilsTest );
