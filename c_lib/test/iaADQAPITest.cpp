#include <string>
#include <exception>

#include <cppunit/Exception.h>
#include <cppunit/TestCase.h>
// #include <cppunit/TestFixture.h>
#include <cppunit/TestResult.h>
#include <cppunit/TestSuite.h>
#include <cppunit/TestCaller.h>
#include <cppunit/ui/text/TestRunner.h>
#include <cppunit/extensions/HelperMacros.h>


#include "iaADQAPI.hpp"

///////////////////////////////////////////////////////////////////////////////
//                          Very basic test example                          //
///////////////////////////////////////////////////////////////////////////////
// class iaADQAPITest : public CppUnit::TestCase {
// public:
//         iaADQAPITest(std::string name) : CppUnit::TestCase(name) {}

//         void exampleTest(){
//                 CPPUNIT_ASSERT (0 == 0);
//                 CPPUNIT_ASSERT (1 == 0);
//                 CPPUNIT_ASSERT (0 == 0);
//         }
// };


///////////////////////////////////////////////////////////////////////////////
//                              Test with suite                              //
///////////////////////////////////////////////////////////////////////////////
// class iaADQAPITest : public CppUnit::TestFixture {
// private:
//         // declare variables for use in tests
// public:
//         // define method that returns a test suite instance
//         static CppUnit::Test *suite(){

//                 // Create suite
//                 CppUnit::TestSuite *suite = new CppUnit::TestSuite( "iaADQAPITest" );

//                 // Populate with tests
//                 suite->addTest( new CppUnit::TestCaller<iaADQAPITest>(
//                                         "exampleTest",
//                                         &iaADQAPITest::exampleTest ) );
//                 return suite;
//         }

//         void setUp() {
//                 //
//         }
//         void tearDown() {
//                 //
//         }
//         void exampleTest(){
//                 CPPUNIT_ASSERT (0 == 0);
//                 CPPUNIT_ASSERT (1 == 0);
//                 CPPUNIT_ASSERT (0 == 0);
//         }
// };
// int main(void)
// {

//         CppUnit::TextUi::TestRunner runner;
//         runner.addTest( iaADQAPITest::suite() );

//         runner.run();
//         return 0;
// }


///////////////////////////////////////////////////////////////////////////////
//                              Utilising macros                             //
///////////////////////////////////////////////////////////////////////////////
class iaADQAPITest : public CppUnit::TestFixture {

        // Macro for generating suite
        CPPUNIT_TEST_SUITE( iaADQAPITest );

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


int main(void)
{
        CppUnit::TextUi::TestRunner runner;
        runner.addTest( iaADQAPITest::suite() );

        runner.run();
        return 0;
}
