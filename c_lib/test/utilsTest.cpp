#include <cppunit/extensions/HelperMacros.h>
#include <cppunit/Exception.h>
#include <cppunit/TestAssert.h>

#include <array>

#include "utils.hpp"

class UtilsTest : public CppUnit::TestFixture {

        // Macro for generating suite
        CPPUNIT_TEST_SUITE( UtilsTest );

        // Population with tests
        // CPPUNIT_TEST_EXCEPTION( 🐙, CppUnit::Exception );
        CPPUNIT_TEST( test_seconds_to_hours );
        CPPUNIT_TEST( test_float_to_string );
        CPPUNIT_TEST( test_copy );
        CPPUNIT_TEST( test_dump_arrays_to_file );
        CPPUNIT_TEST_EXCEPTION( test_load_arrays_from_file__fail,  std::runtime_error );

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
                short** arr_in = new short*[1];
                arr_in[0] = new short[3]{1, 2, 3};

                double** arr_out = new double*[1];
                arr_out[0] = new double[3]{4,5,6};

                double** arr_test = new double*[1];
                arr_test[0] = new double[3]{1, 2, 3};

                cast_arrays<short,double>(arr_in, arr_out, 1, 3);

                for (int i = 0; i < 3; i++) {
                        CPPUNIT_ASSERT_EQUAL( arr_out[0][i], arr_test[0][i]);
                }
        }
        void test_dump_arrays_to_file(){
                short** arr_to_dump = new short*[2];
                arr_to_dump[0] = new short[4]{1,2,3,4};
                arr_to_dump[1] = new short[4]{3,4,5,6};

                dump_arrays_to_file(arr_to_dump, 2, 4, "./test/dump-example.txt", "#Series 1\t#Series 2");

        }

        void test_load_arrays_from_file__fail(){
                short** arr_to_write = new short*[2];

                load_arrays_from_file<int>("does-not-exist.txt", 1, 1);
        }
};

CPPUNIT_TEST_SUITE_REGISTRATION( UtilsTest );
