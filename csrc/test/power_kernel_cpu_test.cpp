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
    unsigned int* sq_data = new unsigned int[NO_POINTS];
    unsigned int* expected_sq_data = new unsigned int[NO_POINTS];
public:
    void setUp(){
    }
    void tearDown(){
        delete[] chA_data;
        delete[] chB_data;
        delete[] sq_data;
        delete[] expected_sq_data;
    }

    void test_power_kernel_v1_no_background(){
        no_threads = 4;

        chA_data = new short[9]{1, 2, 3, 4, 5, 6, 7, 9, 10};
        chB_data = new short[9]{0, 1, 2, 3, 4, 5, 6, 7, 9};
        expected_sq_data = new unsigned int[9]{1, 5, 13, 25, 41, 61, 85, 130, 181};

        CPU::power_kernel_v1_no_background(chA_data, chB_data, sq_data,
                                           NO_POINTS, no_threads);

        for (int i(0); i < 9; i++) {
            CPPUNIT_ASSERT_EQUAL(expected_sq_data[i], sq_data[i]);
        }
    }

    void test_power_kernel_v2_const_background(){
        no_threads = 10;

        chA_data = new short[9]{1, 2, 3, 4, 5, 6, 7, 9, 10};
        chB_data = new short[9]{0, 1, 2, 3, 4, 5, 6, 7, 9};
        short chA_const_background = 1;
        short chB_const_background = 0;
        expected_sq_data = new unsigned int[9]{0, 2, 8, 18, 32, 50, 72, 113, 162};

        CPU::power_kernel_v2_const_background(
            chA_data, chB_data, sq_data,
            chA_const_background, chB_const_background,
            NO_POINTS, no_threads
            );
        for (int i(0); i < 9; i++) {
            CPPUNIT_ASSERT_EQUAL(expected_sq_data[i], sq_data[i]);
        }
    }

    void test_power_kernel_v3_background(){
        no_threads = 1;

        chA_data = new short[9]{1, 2, 3, 4, 5, 6, 7, 9, 10};
        chB_data = new short[9]{0, 1, 2, 3, 4, 5, 6, 7, 9};
        short *chA_const_background = new short[9]{1, 2, 3, 4, 5, 6, 7, 9, 10};
        short *chB_const_background = new short[9]{0,0,0,0,0,0,0,0,0};
        short* expected_chA_data = new short[9]{0,0,0,0,0,0,0,0,0};
        short* expected_chB_data = new short[9]{0, 1, 2, 3, 4, 5, 6, 7, 9};
        expected_sq_data = new unsigned int[9]{0, 1, 4, 9, 16, 25, 36, 49, 81};

        CPU::power_kernel_v3_background(
            chA_data, chB_data, sq_data,
            chA_const_background, chB_const_background,
            NO_POINTS, no_threads
            );
        for (int i(0); i < 9; i++) {
            CPPUNIT_ASSERT_EQUAL(expected_sq_data[i], sq_data[i]);
            CPPUNIT_ASSERT_EQUAL(expected_chA_data[i], chA_data[i]);
            CPPUNIT_ASSERT_EQUAL(expected_chB_data[i], chB_data[i]);
        }
    }
};
CPPUNIT_TEST_SUITE_REGISTRATION( PowerCpuTest );
