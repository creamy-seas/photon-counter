#include <cppunit/extensions/HelperMacros.h>
#include <cppunit/Exception.h>

#include <array>

#include "utils.hpp"

class UtilsTest : public CppUnit::TestFixture {

        // Macro for generating suite
        CPPUNIT_TEST_SUITE( UtilsTest );

        // Population with tests
        // CPPUNIT_TEST_EXCEPTION( 🐙, CppUnit::Exception );
        CPPUNIT_TEST( test_seconds_to_hours );
        CPPUNIT_TEST( test_float_to_string );

        CPPUNIT_TEST_SUITE_END();
private:

public:
        void setUp(){
        }
        void tearDown(){
        }

        void test_float_to_string(){
                CPPUNIT_ASSERT( float_to_string(20.226, 2) == "20.23");
        }
        void test_seconds_to_hours(){
                CPPUNIT_ASSERT( seconds_to_hours(3660) == "1h:1m");
        }
        void test_copy(){
                short arr_in[1][3] = {{1,2,3}};
                double arr_out[1][3];

                cast_arrays(arr_in, arr_out, 1, 3);
                // for (int i = 0; i < 3; i++) {
                // CPPUNIT_ASSERT ( arr_in[0][i] == arr_out[0][i]);
                // }
        }
};

CPPUNIT_TEST_SUITE_REGISTRATION( UtilsTest );
