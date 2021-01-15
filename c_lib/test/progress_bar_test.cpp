#include <thread>
#include <chrono>
#include <cppunit/extensions/HelperMacros.h>

#include "progress_bar.hpp"

class ProgressBarTest : public CppUnit::TestFixture {

        // Macro for generating suite
        CPPUNIT_TEST_SUITE( ProgressBarTest );

        // Population with tests
        // CPPUNIT_TEST_EXCEPTION( 🐙, CppUnit::Exception );
        CPPUNIT_TEST( for_loop );
        CPPUNIT_TEST( test_empty_and_clear_field );

        CPPUNIT_TEST_SUITE_END();
private:
        int n;
        ProgressBar *bar;
public:
        void setUp() {
                n = 100;
                bar = new ProgressBar(n, "Test progress bar");
        }
        void tearDown(){
                delete bar;
        }
        void for_loop(){

                bar->SetStyle("#",".");
                bar->SetFrequencyUpdate(3);

                for(int i=0;i<=n;++i){
                        bar->Progressed(i);
                        std::this_thread::sleep_for(std::chrono::milliseconds(4));
                }
        }
        void test_empty_and_clear_field(){
                ProgressBar bar_temp = ProgressBar();
                bar_temp.SetFrequencyUpdate(1000);
        }
};
CPPUNIT_TEST_SUITE_REGISTRATION( ProgressBarTest );
