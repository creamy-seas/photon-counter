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
    CPPUNIT_TEST( test_power_kernel_v2_const_background );
    CPPUNIT_TEST( test_power_kernel_v3_background );

    CPPUNIT_TEST_SUITE_END();
private:
    int no_threads;

    short* chA_data;
    short* chB_data;
    float* sq_data;
    float* expected_sq_data;
public:
    void setUp(){
    }
    void tearDown(){
    }

    void test_power_kernel_v1_no_background(){
        no_threads = 4;

        chA_data = new short[12]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
        chB_data = new short[12]{0, 1, 0, 1, 0, 1, 0, 1, 0, 2, 2, 2};
        sq_data = new float[3]{-1, -2, -3};
        expected_sq_data = new float[3]{(float)(1 + 17 + 49 + 104) / 4,
                (float)(5 + 25 + 65 + 125) / 4,
                (float)(9 + 37 + 81 + 148) / 4};

        CPU::power_kernel_v1_no_background(chA_data, chB_data, sq_data,
                                           SP_POINTS, R_POINTS, no_threads);

        for (int i(0); i < 3; i++) {
            CPPUNIT_ASSERT_EQUAL(expected_sq_data[i], sq_data[i]);
        }

        delete[] chA_data;
        delete[] chB_data;
        delete[] sq_data;
        delete[] expected_sq_data;
    }

    void test_power_kernel_v2_const_background(){
        no_threads = 10;

        chA_data = new short[12]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
        chB_data = new short[12]{0, 1, 0, 1, 0, 1, 0, 1, 0, 2, 2, 2};
        sq_data = new float[3]{-1, -2, -3};
        expected_sq_data = new float[3]{(float)(0 + 10 + 36 + 85) / 4,
                (float)(2 + 16 + 50 + 104) / 4,
                (float)(4 + 26 + 64 + 125) / 4};
        short chA_const_background = 1;
        short chB_const_background = 0;

        CPU::power_kernel_v2_const_background(
            chA_data, chB_data, sq_data,
            chA_const_background, chB_const_background,
            SP_POINTS, R_POINTS, no_threads
            );
        for (int i(0); i < 3; i++) {
            CPPUNIT_ASSERT_EQUAL(expected_sq_data[i], sq_data[i]);
        }

        delete[] chA_data;
        delete[] chB_data;
        delete[] sq_data;
        delete[] expected_sq_data;
    }

    void test_power_kernel_v3_background(){
        no_threads = 1;

        chA_data = new short[12]{1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3};
        chB_data = new short[12]{0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12};
        short *chA_background = new short[3]{1, 2, 3};
        short *chB_background = new short[3]{0, 0, 0};

        sq_data = new float[3]{-1, -2, -3};
        expected_sq_data = new float[3]{(float)(0 + 9 + 36 + 100) / 4,
                (float)(1 + 16 + 49 + 121) / 4,
                (float)(4 + 25 + 81 + 144) / 4};

        CPU::power_kernel_v3_background(
            chA_data, chB_data, sq_data,
            chA_background, chB_background,
            SP_POINTS, R_POINTS, no_threads
            );
        for (int i(0); i < 3; i++) {
            CPPUNIT_ASSERT_EQUAL(expected_sq_data[i], sq_data[i]);
        }
        delete[] chA_data;
        delete[] chB_data;
        delete[] sq_data;
        delete[] expected_sq_data;
        delete[] chA_background;
        delete[] chB_background;
    }
};
CPPUNIT_TEST_SUITE_REGISTRATION( PowerCpuTest );
