#include <cppunit/extensions/HelperMacros.h>
#include <cppunit/Exception.h>

#include "utils_gpu.hpp"

class UtilsGpuTest : public CppUnit::TestFixture {

        // Macro for generating suite
        CPPUNIT_TEST_SUITE( UtilsGpuTest );

        // Population with tests
        // CPPUNIT_TEST_EXCEPTION( 🐙, CppUnit::Exception );
        CPPUNIT_TEST( test_fetch_gpu_parameters );

        CPPUNIT_TEST_SUITE_END();
private:

public:
        void setUp(){
        }
        void tearDown(){
        }

        void test_fetch_gpu_parameters(){
                fetch_gpu_parameters();
        };
};
CPPUNIT_TEST_SUITE_REGISTRATION( UtilsGpuTest );
