#include <string>
#include <cppunit/extensions/HelperMacros.h>
#include <cppunit/Exception.h>

#include "playground.hpp"

class PlaygroundTest : public CppUnit::TestFixture {

        // Macro for generating suite
        CPPUNIT_TEST_SUITE( PlaygroundTest );

        // Population with tests
        // CPPUNIT_TEST_EXCEPTION( 🐙, CppUnit::Exception );
        CPPUNIT_TEST( test_example_gpu_func );

        CPPUNIT_TEST_SUITE_END();
private:
public:
        void test_example_gpu_func(){
                float expected_result = 5;
                // CPPUNIT_ASSERT_EQUAL(
                //         expected_result,
                //         GPU::example_gpu_func(1, 2));
        }
};
CPPUNIT_TEST_SUITE_REGISTRATION( PlaygroundTest );
