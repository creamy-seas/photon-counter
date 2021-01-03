#include <cppunit/extensions/TestFactoryRegistry.h>
#include <cppunit/ui/text/TestRunner.h>

int main(void)
{
        // Create runner
        CppUnit::TextUi::TestRunner runner;

        // Retrieve registry (in the cpp files, we add a TestSuiteFactory to this registry
        // with the CPPUNIT_TEST_SUITE_REGISTRATION macro)
        CppUnit::TestFactoryRegistry &registry = (
                CppUnit::TestFactoryRegistry::getRegistry()
                );

        //  Add the test suites in the registry to the runner
        runner.addTest( registry.makeTest() );

        runner.run();
        return 0;
}
