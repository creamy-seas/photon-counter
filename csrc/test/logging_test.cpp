#include <cppunit/extensions/HelperMacros.h>
#include <limits.h> // For LONG MAX
#include "logging.hpp"

class LoggingTest : public CppUnit::TestFixture {

    // Macro for generating suite
    CPPUNIT_TEST_SUITE(  LoggingTest );

    // Population with tests
    // CPPUNIT_TEST_EXCEPTION( 🐙, CppUnit::Exception );
    CPPUNIT_TEST( test_log_error_to_file );

    CPPUNIT_TEST_SUITE_END();

public:
    void test_log_error_to_file(){
        int no_repetitions = 0;
        int max_code = 14;

        append_to_log_file("No repetitions ("
                           + std::to_string(no_repetitions)
                           + ") x 14bit Code ("
                           + std::to_string(max_code)
                           + ") x R_POINTS(number of records per point="
                           + std::to_string(R_POINTS)
                           + ") can overflow the cumulative arrays of type LONG ("
                           + std::to_string(LONG_MAX)
                           + ")");
    }
};
CPPUNIT_TEST_SUITE_REGISTRATION( LoggingTest );
