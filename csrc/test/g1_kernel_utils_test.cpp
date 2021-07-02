#include <cppunit/extensions/HelperMacros.h>
#include <cppunit/Exception.h>

#include "g1_kernel.hpp"

class G1KernelUtilsTest : public CppUnit::TestFixture {

    // Macro for generating suite
    CPPUNIT_TEST_SUITE( G1KernelUtilsTest );

    // Population with tests
    // CPPUNIT_TEST_EXCEPTION( 🐙, CppUnit::Exception );
    CPPUNIT_TEST( test_mean_variance );
    CPPUNIT_TEST( test_on_gpu );
    CPPUNIT_TEST( test_on_gpu_irregular_arrays );

    CPPUNIT_TEST_SUITE_END();

private:
public:
    void setUp(){
    }
    void tearDown(){
    }

    void test_mean_variance(){

        const int N = 8;
        short chA_data[N] = {4, 5, 6, 7, 8, 10, 12, 12};
        short chB_data[N] = {0, 0, 0, 0, 0, 1, 0, 16384};
        int sq_data[N];

        double mean_list[G1::no_outputs];
        double var_list[G1::no_outputs];
        double chA_normalised[N]; double chB_normalised[N]; double sq_normalised[N];
        double *normalised_data[G1::no_outputs];
        normalised_data[CHAG1] = chA_normalised;
        normalised_data[CHBG1] = chB_normalised;
        normalised_data[SQG1] = sq_normalised;

        G1::CPU::preprocessor(chA_data, chB_data,
                              N,
                              mean_list, var_list,
                              normalised_data);

        CPPUNIT_ASSERT_DOUBLES_EQUAL(8.0, mean_list[CHAG1], 0.00001);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(2048.125, mean_list[CHBG1], 0.00001);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(33554504.375, mean_list[SQG1], 0.00001);

        CPPUNIT_ASSERT_DOUBLES_EQUAL(8.25, var_list[CHAG1], 0.00001);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(29359616.109375, var_list[CHBG1], 0.00001);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(7881304154573057.0, var_list[SQG1], 0.00001);

        double normalised_data_expected[3][N] = {
            {-4, -3, -2, -1,  0,  2,  4,  4},
            {-2048.125, -2048.125, -2048.125, -2048.125, -2048.125, -2047.125,
             -2048.125, 14335.875},
            {0, 0, 0, 0, 0, 0, 0, 0}
        };

        for (int i = 0; i < N; i++){
            CPPUNIT_ASSERT_EQUAL(normalised_data_expected[0][i], normalised_data[CHAG1][i]);
            CPPUNIT_ASSERT_EQUAL(normalised_data_expected[1][i], normalised_data[CHBG1][i]);
        }

    }

    void test_on_gpu_irregular_arrays(){

        int N = 262144;
        short *chA_data = new short[N];
        for (int i(0); i < N; i++)
            chA_data[i] = 1;
        short *chB_data = new short[N];

        float mean_list[G1::no_outputs]; float var_list[G1::no_outputs];

        float *chA_normalised = new float[N];
        float *chB_normalised = new float[N];
        float *sq_normalised = new float[N];
        float *normalised_data[G1::no_outputs];
        normalised_data[CHAG1] = chA_normalised;
        normalised_data[CHBG1] = chB_normalised;
        normalised_data[SQG1] = sq_normalised;

        G1::GPU::preprocessor(
            N, chA_data,
            chB_data, mean_list, var_list, normalised_data);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, mean_list[CHAG1], 0.00001);

        N = 13;
        delete[] chA_data;
        chA_data = new short[N];
        for (int i(0); i < N; i++)
            chA_data[i] = 1;

        G1::GPU::preprocessor(
            N, chA_data,
            chB_data, mean_list, var_list, normalised_data);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, mean_list[CHAG1], 0.00001);

        // CPPUNIT_ASSERT_DOUBLES_EQUAL(2048.125, mean_list[CHBG1], 0.00001);
        // CPPUNIT_ASSERT_DOUBLES_EQUAL(33554504.375, mean_list[SQG1], 0.00001);

        // CPPUNIT_ASSERT_DOUBLES_EQUAL(8.25, var_list[CHAG1], 0.00001);
        // CPPUNIT_ASSERT_DOUBLES_EQUAL(29359616.109375, var_list[CHBG1], 0.00001);
        // CPPUNIT_ASSERT_DOUBLES_EQUAL(7881304154573057.0, var_list[SQG1], 0.00001);

        // double normalised_data_expected[3][N] = {
        //     {-4, -3, -2, -1,  0,  2,  4,  4},
        //     {-2048.125, -2048.125, -2048.125, -2048.125, -2048.125, -2047.125,
        //      -2048.125, 14335.875},
        //     {0, 0, 0, 0, 0, 0, 0, 0}
        // };

        // for (int i = 0; i < N; i++){
        //     CPPUNIT_ASSERT_EQUAL(normalised_data_expected[0][i], normalised_data[CHAG1][i]);
        //     CPPUNIT_ASSERT_EQUAL(normalised_data_expected[1][i], normalised_data[CHBG1][i]);
        // }

    }

    void test_on_gpu(){

        // TODO: Remove
        G1::check_g1_kernel_parameters(true);

        const int N = 8;
        short chA_data[N] = {4, 5, 6, 7, 8, 10, 12, 12};
        short chB_data[N] = {0, 0, 0, 0, 0, 1, 0, 16384};
        int sq_data[N];

        float mean_list[G1::no_outputs]; float var_list[G1::no_outputs];

        float chA_normalised[N]; float chB_normalised[N]; float sq_normalised[N];
        float *normalised_data[G1::no_outputs];
        normalised_data[CHAG1] = chA_normalised;
        normalised_data[CHBG1] = chB_normalised;
        normalised_data[SQG1] = sq_normalised;

        G1::GPU::preprocessor(
            N, chA_data,
            chB_data, mean_list, var_list, normalised_data);

        CPPUNIT_ASSERT_DOUBLES_EQUAL(8.0, mean_list[CHAG1], 0.00001);
        // CPPUNIT_ASSERT_DOUBLES_EQUAL(2048.125, mean_list[CHBG1], 0.00001);
        // CPPUNIT_ASSERT_DOUBLES_EQUAL(33554504.375, mean_list[SQG1], 0.00001);

        // CPPUNIT_ASSERT_DOUBLES_EQUAL(8.25, var_list[CHAG1], 0.00001);
        // CPPUNIT_ASSERT_DOUBLES_EQUAL(29359616.109375, var_list[CHBG1], 0.00001);
        // CPPUNIT_ASSERT_DOUBLES_EQUAL(7881304154573057.0, var_list[SQG1], 0.00001);

        float normalised_data_expected[3][N] = {
            {-4, -3, -2, -1,  0,  2,  4,  4},
            {-2048.125, -2048.125, -2048.125, -2048.125, -2048.125, -2047.125,
             -2048.125, 14335.875},
            {0, 0, 0, 0, 0, 0, 0, 0}
        };

        for (int i = 0; i < N; i++){
            CPPUNIT_ASSERT_EQUAL(normalised_data_expected[0][i], normalised_data[CHAG1][i]);
            // CPPUNIT_ASSERT_EQUAL(normalised_data_expected[1][i], normalised_data[CHBG1][i]);
        }

    }
};
CPPUNIT_TEST_SUITE_REGISTRATION( G1KernelUtilsTest );
