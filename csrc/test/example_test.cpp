#include <cppunit/extensions/HelperMacros.h>

class ExampleTest : public CppUnit::TestFixture {

        // Macro for generating suite
        CPPUNIT_TEST_SUITE( ExampleTest );

        // Population with tests
        CPPUNIT_TEST_EXCEPTION( test_fail, CppUnit::Exception );
        // CPPUNIT_TEST( test_fail );
        CPPUNIT_TEST( test_ok );

        CPPUNIT_TEST_SUITE_END();
private:
        // declare variables for use in tests
public:
        void setUp() {
                //
        }
        void tearDown() {
                //
        }
        void test_ok(){
                CPPUNIT_ASSERT (0 == 0);
        }
        void test_fail(){
                CPPUNIT_ASSERT (1 == 0);
        }
};

// Add suite to Factory Register which will be retrieved later on
CPPUNIT_TEST_SUITE_REGISTRATION( ExampleTest );
