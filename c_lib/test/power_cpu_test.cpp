#include <cppunit/extensions/HelperMacros.h>
#include <cppunit/Exception.h>

#include "power.hpp"

class PowerCpuTest : public CppUnit::TestFixture {

        // Macro for generating suite
        CPPUNIT_TEST_SUITE( PowerCpuTest );

        // Population with tests
        // CPPUNIT_TEST_EXCEPTION( 🐙, CppUnit::Exception );
        CPPUNIT_TEST( test_power_no_background );

        CPPUNIT_TEST_SUITE_END();
private:
        short* chA_data;
        short* chB_data;
        unsigned int* sq_data;
        unsigned int expected_sq_data[9] = {1, 5, 13, 25, 41, 61, 85, 130, 181};
public:
        void setUp(){
                chA_data = new short[9]{1, 2, 3, 4, 5, 6, 7, 9, 10};
                chB_data = new short[9]{0, 1, 2, 3, 4, 5, 6, 7, 9};
                sq_data = new unsigned int[9];
        }
        void tearDown(){
                delete chA_data;
                delete chB_data;
        }

        void test_power_no_background(){

                CPU::power_kernel_v1(chA_data, chB_data, sq_data, 1, 9, 4);

                for (int i(0); i < 9; i++) {
                        CPPUNIT_ASSERT_EQUAL(expected_sq_data[i], sq_data[i]);
                }
        }


};
CPPUNIT_TEST_SUITE_REGISTRATION( PowerCpuTest );
