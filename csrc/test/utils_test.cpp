#include <cppunit/extensions/HelperMacros.h>
#include <cppunit/Exception.h>
#include <cppunit/TestAssert.h>

#include "utils.hpp"

class UtilsTest : public CppUnit::TestFixture {

        // Macro for generating suite
        CPPUNIT_TEST_SUITE( UtilsTest );

        // Population with tests
        CPPUNIT_TEST( test_seconds_to_hours );
        CPPUNIT_TEST( test_float_to_string );
        CPPUNIT_TEST( test_copy );
        CPPUNIT_TEST( test_dump_arrays_to_file );
        CPPUNIT_TEST_EXCEPTION( test_load_arrays_from_file__fail,  std::runtime_error );
        CPPUNIT_TEST_EXCEPTION( test_load_arrays_from_file__too_short_y,  std::runtime_error );
        CPPUNIT_TEST_EXCEPTION( test_load_arrays_from_file__too_long_y,  std::runtime_error );
        CPPUNIT_TEST( test_load_arrays_from_file__ok );
        CPPUNIT_TEST( test_read_and_write );
        // CPPUNIT_TEST( test_load_arrays_from_file__corrupted );

        CPPUNIT_TEST_SUITE_END();
private:
        // For load_arrays_from_file
        short** arr_to_load;
        short** arr_load_example;
public:
        void setUp(){
                arr_to_load = new short*[2];
                arr_to_load[0] = new short[4]{-1};
                arr_to_load[1] = new short[4]{-1};

                arr_load_example = new short*[2];
                arr_load_example[0] = new short[4]{1, 2, 3, 4};
                arr_load_example[1] = new short[4]{3, 4, 5, 6};
        }
        void tearDown(){
                for (int x(0); x < 2; x++) {
                        delete[] arr_to_load[x];
                        delete[] arr_load_example[x];
                }
                delete[] arr_to_load;
                delete[] arr_load_example;
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

                dump_arrays_to_file(arr_to_dump, 2, 4, "./test/test_bin/dump-example.txt", "#Series 1\tSeries 2", 1);
        }

        void test_load_arrays_from_file__fail(){
                load_arrays_from_file(arr_to_load, "does-not-exist.txt", 1, 1);
        }

        void test_load_arrays_from_file__ok(){
                load_arrays_from_file(arr_to_load, "test/test_files/load_example.txt", 2, 4);

                for (int x(0); x < 2; x++) {
                        for (int y(0); y < 4; y++)
                                CPPUNIT_ASSERT_EQUAL(arr_load_example[x][y], arr_to_load[x][y]);
                }
        }
        // void test_load_arrays_from_file__corrupted(){
        //         load_arrays_from_file(arr_to_load, "test/test_files/load_example_corrupted.txt", 2, 4);

        //         for (int x(0); x < 2; x++) {
        //                 for (int y(0); y < 4; y++)
        //                         CPPUNIT_ASSERT_EQUAL(arr_load_example[x][y], arr_to_load[x][y]);
        //         }
        // }
        void test_load_arrays_from_file__too_short_y(){
                load_arrays_from_file<short>(arr_to_load, "test/test_files/load_example.txt", 2, 3);
        }
        void test_load_arrays_from_file__too_long_y(){
                load_arrays_from_file<short>(arr_to_load, "test/test_files/load_example.txt", 2, 5);
        }
        void test_read_and_write(){
                double** arr_to_dump = new double*[3];
                arr_to_dump[0] = new double[4]{1.1,2.2,3.3,4.4};
                arr_to_dump[1] = new double[4]{4.4,5.5,6.6,7.7};
                arr_to_dump[2] = new double[4]{8.8,9.9,11.11,12.12};

                dump_arrays_to_file(arr_to_dump,
                                    3,
                                    4,
                                    "./test/test_bin/dump-example.txt", "#Series 1\tSeries 2\tSeries 3",
                                    1);

                double** arr_to_load = new double*[3];
                arr_to_load[0] = new double[4];
                arr_to_load[1] = new double[4];
                arr_to_load[2] = new double[4];
                load_arrays_from_file(arr_to_load,
                                      "./test/test_bin/dump-example.txt",
                                      3, 4);

                for (int x(0); x < 3; x++) {
                        for (int y(0); y < 4; y++)
                                CPPUNIT_ASSERT_EQUAL(arr_to_dump[x][y], arr_to_load[x][y]);
                }

                for (int x(0); x < 3; x++){
                        delete[] arr_to_dump[x];
                        delete[] arr_to_load[x];
                }
                delete[] arr_to_dump;
                delete[] arr_to_load;
        }
};

CPPUNIT_TEST_SUITE_REGISTRATION( UtilsTest );
