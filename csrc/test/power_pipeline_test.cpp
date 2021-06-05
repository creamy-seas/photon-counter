#include <cppunit/extensions/HelperMacros.h>
#include <cppunit/Exception.h>
#include <stdexcept> //for std::runtime_error
#include <limits.h> // For LONG_MAX

#include "power_pipeline.hpp"
#include "sp_digitiser.hpp"
#include "ADQAPI.h" // For MultiRecordSetup

class PowerPipelineTest : public CppUnit::TestFixture {

    // Macro for generating suite
    CPPUNIT_TEST_SUITE( PowerPipelineTest );

    // Population with tests
    // CPPUNIT_TEST_EXCEPTION( 🐙, CppUnit::Exception );
    CPPUNIT_TEST( test_power_pipeline );
    CPPUNIT_TEST( test_power_pipeline_too_many_repetitions );

    CPPUNIT_TEST_SUITE_END();

public:
    void* adq_cu_ptr;
    short *chA_background, *chB_background;

    void setUp(){
        adq_cu_ptr = master_setup(
            NO_BLINK,
            INTERNAL_CLOCK_SOURCE_INTERNAL_10MHZ_REFFERENCE,
            TRIGGER_SOFTWARE);

        chA_background = new short[SP_POINTS]();
        chB_background = new short[SP_POINTS]();
    }
    void tearDown(){
        DeleteADQControlUnit(adq_cu_ptr);
    }

    void test_power_pipeline() {
        ADQ214_MultiRecordSetup(adq_cu_ptr, 1, R_POINTS, SP_POINTS);

        run_power_measurements(adq_cu_ptr,
                               chA_background, chB_background,
                               10, "./dump/power-pipeline-example.txt");

        ADQ214_MultiRecordClose(adq_cu_ptr, 1);
    };

    void test_power_pipeline_too_many_repetitions() {
        ADQ214_MultiRecordSetup(adq_cu_ptr, 1, R_POINTS, SP_POINTS);

        CPPUNIT_ASSERT_THROW_MESSAGE(
            "Should give warning message that LONG arrays will overflow",
            run_power_measurements(adq_cu_ptr,
                                   chA_background, chB_background,
                                   LONG_MAX,
                                   "./dump/power-pipeline-example.txt"),
            std::runtime_error);

        ADQ214_MultiRecordClose(adq_cu_ptr, 1);
    };
};
CPPUNIT_TEST_SUITE_REGISTRATION( PowerPipelineTest );
