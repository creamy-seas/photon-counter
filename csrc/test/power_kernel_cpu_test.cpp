#include <cppunit/extensions/HelperMacros.h>
#include <cppunit/Exception.h>

#include "power_kernel.hpp"

#ifdef TESTENV
#define NO_POINTS 9
#endif // TESTENV

class PowerCpuTest : public CppUnit::TestFixture {

    // Macro for generating suite
    CPPUNIT_TEST_SUITE( PowerCpuTest );

    // Population with tests
    // CPPUNIT_TEST_EXCEPTION( 🐙, CppUnit::Exception );
    CPPUNIT_TEST( test_power_kernel);

    CPPUNIT_TEST_SUITE_END();
private:
    int no_threads;

    short* chA_data;
    short* chB_data;
    double** data_out;
    double *expected_A_out;
    double *expected_B_out;
    double* expected_sq_out;
    unsigned int processing_mask;

public:
    void setUp(){

        chA_data = new short[24]{2, 2, 3,
                1, 2, 3,
                1, 2, 3,
                1, 2, 3,
                3, 3, 4,
                5, 5, 5,
                4, 3, 4,
                4, 3, 11};
        chB_data = new short[24]{0, 1, 2,
                3, 4, 5,
                6, 7, 9,
                10, 11, 12,
                0, 0, 0,
                1, 1, 1,
                2, 2, 2,
                3, 3, 3};

        // output arrays, chA, chB, chAsq, chBsq, SQ
        data_out = new double*[NO_OF_POWER_KERNEL_OUTPUTS];
        for (int i(0); i < NO_OF_POWER_KERNEL_OUTPUTS; i++) {
            data_out[i] = new double[3]();
        }
    }
    void tearDown(){
        delete[] chA_data;
        delete[] chB_data;

        for (int i(0); i < NO_OF_POWER_KERNEL_OUTPUTS; i++)
            delete[] data_out[i];
        delete[] data_out;
    }

    void test_power_kernel(){
        no_threads = 1;

        short *chA_background = new short[3]{1, 2, 3};
        short *chB_background = new short[3]{0, 0, 0};

        expected_A_out = new double[3]{1.625, 0.75, 1.5};
        expected_B_out = new double[3]{3.125, 3.625, 4.25};
        expected_sq_out = new double[3]{24.75, 26.625, 42.25};

        processing_mask = SQ_MASK ^ CHA_MASK ^ CHB_MASK ^ CHBSQ_MASK ^ CHASQ_MASK;

        POWER::CPU::power_kernel(
            chA_data, chB_data, data_out,
            processing_mask,
            chA_background, chB_background,
            SP_POINTS, R_POINTS, no_threads);
        for (int i(0); i < 3; i++) {
            CPPUNIT_ASSERT_EQUAL(expected_A_out[i], data_out[CHA][i]);
            CPPUNIT_ASSERT_EQUAL(expected_B_out[1], data_out[CHB][1]);
            CPPUNIT_ASSERT_EQUAL(expected_sq_out[i], data_out[SQ][i]);
        }

        delete[] expected_A_out;
        delete[] expected_B_out;
        delete[] expected_sq_out;
        delete[] chA_background;
        delete[] chB_background;
    }
};
CPPUNIT_TEST_SUITE_REGISTRATION( PowerCpuTest );
