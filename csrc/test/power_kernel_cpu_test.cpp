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
    CPPUNIT_TEST( test_power_kernel_v1_no_background );
    // CPPUNIT_TEST( test_power_kernel_v2_const_background );
    // CPPUNIT_TEST( test_power_kernel_v3_background );

    CPPUNIT_TEST_SUITE_END();
private:
    int no_threads;

    short* chA_data;
    short* chB_data;
    double** processed_data;
    double* expected_sq_out;
    unsigned int processing_mask;

public:
    void setUp(){
        // 3 output arrays, chA, chB, SQ
        processed_data = new double*[3];
    }
    void tearDown(){
        delete[] processed_data;
    }

    void test_power_kernel_v1_no_background(){
        no_threads = 4;

        chA_data = new short[12]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
        chB_data = new short[12]{0, 1, 0, 1, 0, 1, 0, 1, 0, 2, 2, 2};
        processed_data[2] = new double[3]{-1, -2, -3};
        expected_sq_out = new double[3]{(double)(1 + 17 + 49 + 104) / 4,
                (double)(5 + 25 + 65 + 125) / 4,
                (double)(9 + 37 + 81 + 148) / 4};
        processing_mask = SQ_MASK;

        CPU::power_kernel_v1_no_background(chA_data, chB_data, processed_data,
                                           processing_mask,
                                           SP_POINTS, R_POINTS, no_threads);

        for (int i(0); i < 3; i++) {
            CPPUNIT_ASSERT_EQUAL(expected_sq_out[i], processed_data[2][i]);
        }

        delete[] chA_data;
        delete[] chB_data;
        delete[] processed_data[2];
        delete[] expected_sq_out;
    }

    void test_power_kernel_v2_const_background(){
        no_threads = 10;

        chA_data = new short[12]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
        chB_data = new short[12]{0, 1, 0, 1, 0, 1, 0, 1, 0, 2, 2, 2};
        processed_data[2] = new double[3]{-1, -2, -3};
        expected_sq_out = new double[3]{(double)(0 + 10 + 36 + 85) / 4,
                (double)(2 + 16 + 50 + 104) / 4,
                (double)(4 + 26 + 64 + 125) / 4};
        short chA_const_background = 1;
        short chB_const_background = 0;

        processing_mask = SQ_MASK;

        CPU::power_kernel_v2_const_background(
            chA_data, chB_data, processed_data,
            processing_mask,
            chA_const_background, chB_const_background,
            SP_POINTS, R_POINTS, no_threads
            );
        for (int i(0); i < 3; i++) {
            CPPUNIT_ASSERT_EQUAL(expected_sq_out[i], processed_data[2][i]);
        }

        delete[] chA_data;
        delete[] chB_data;
        delete[] processed_data[2];
        delete[] expected_sq_out;
    }

    void test_power_kernel_v3_background(){
        no_threads = 1;

        chA_data = new short[12]{1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3};
        chB_data = new short[12]{0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12};
        short *chA_background = new short[3]{1, 2, 3};
        short *chB_background = new short[3]{0, 0, 0};

        processed_data[2] = new double[3]{-1, -2, -3};
        expected_sq_out = new double[3]{(double)(0 + 9 + 36 + 100) / 4,
                (double)(1 + 16 + 49 + 121) / 4,
                (double)(4 + 25 + 81 + 144) / 4};

        processing_mask = SQ_MASK;

        CPU::power_kernel_v3_background(
            chA_data, chB_data, processed_data,
            processing_mask,
            chA_background, chB_background,
            SP_POINTS, R_POINTS, no_threads
            );
        for (int i(0); i < 3; i++) {
            CPPUNIT_ASSERT_EQUAL(expected_sq_out[i], processed_data[2][i]);
        }
        delete[] chA_data;
        delete[] chB_data;
        delete[] processed_data[2];
        delete[] expected_sq_out;
        delete[] chA_background;
        delete[] chB_background;
    }
};
CPPUNIT_TEST_SUITE_REGISTRATION( PowerCpuTest );
