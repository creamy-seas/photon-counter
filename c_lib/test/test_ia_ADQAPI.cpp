#include <string>

#include <cppunit/TestCase.h>
#include <cppunit/TestFixture.h>

#include "ia_ADQAPI.hpp"

///////////////////////////////////////////////////////////////////////////////
//                            Test with a fixture                            //
///////////////////////////////////////////////////////////////////////////////
// class test_ia_ADQAPI : public CppUnit::TestFixture {
// private:
//         // declare variables for use in tests
// public:
//         void setUp() {
//                 //
//         }
//         void tearDown() {
//                 //
//         }
// }

///////////////////////////////////////////////////////////////////////////////
//                          Very basic test example                          //
///////////////////////////////////////////////////////////////////////////////
class test_ia_ADQAPI : public CppUnit::TestCase {
public:
        test_ia_ADQAPI(std::string name) : CppUnit::TestCase(name) {}
        void runTest(){
                CPPUNIT_ASSERT (0 == 0);
                CPPUNIT_ASSERT (1 == 0);
                CPPUNIT_ASSERT (0 == 0);
        }
};

int main(void)
{
        test_ia_ADQAPI("Main test").runTest();
        return 0;
}
