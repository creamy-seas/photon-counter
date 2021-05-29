#include <cppunit/extensions/HelperMacros.h>

#include "ia_ADQAPI.hpp"
#include "ADQAPI.h"

class ia_ADQAPITest : public CppUnit::TestFixture {

        // Macro for generating suite
        CPPUNIT_TEST_SUITE( ia_ADQAPITest );

        // Population with tests
        // CPPUNIT_TEST_EXCEPTION( test_fail, CppUnit::Exception );
        CPPUNIT_TEST( test_GetMaxNofRecordsFromNofSamples );
        CPPUNIT_TEST( test_GetMaxNofSamplesFromNofRecords );
        CPPUNIT_TEST( test_fetch_channel_data );
    // CPPUNIT_TEST( test_setup_with_blink );

    CPPUNIT_TEST_SUITE_END();
private:
    void* adq_cu_ptr;
public:
    void setUp() {
        adq_cu_ptr = master_setup(
            NO_BLINK,
            INTERNAL_CLOCK_SOURCE_INTERNAL_10MHZ_REFFERENCE,
            TRIGGER_SOFTWARE);
    }
    void tearDown() {
        DeleteADQControlUnit(adq_cu_ptr);
    }
    void test_setup_with_blink() {
        master_setup(BLINK,
                     INTERNAL_CLOCK_SOURCE_EXTERNAL_10MHZ_REFFERENCE,
                     TRIGGER_EXTERNAL
            );
    };
    void test_GetMaxNofSamplesFromNofRecords(){
        int max_number_of_samples = 0;
        max_number_of_samples = GetMaxNofSamplesFromNofRecords(adq_cu_ptr, 1);
        CPPUNIT_ASSERT (max_number_of_samples != 0);
    }
    void test_GetMaxNofRecordsFromNofSamples(){
        int max_number_of_records = 0;
        max_number_of_records = GetMaxNofRecordsFromNofSamples(adq_cu_ptr, 1);
        CPPUNIT_ASSERT (max_number_of_records != 0);
    }
    void test_fetch_channel_data(){
        unsigned int number_of_records = 4;
        unsigned int samples_per_record = GetMaxNofRecordsFromNofSamples(adq_cu_ptr, number_of_records);

        short* buff_a = new short[samples_per_record*number_of_records];
        short* buff_b = new short[samples_per_record*number_of_records];

        for (int i(0); i < samples_per_record*number_of_records; i++) {
            buff_a[i] = -1;
            buff_b[i]= -1;
        }

        // Prepare multirecord mode
        ADQ_MultiRecordSetup(adq_cu_ptr, 1, number_of_records,  samples_per_record);

        fetch_digitiser_data(adq_cu_ptr,
                           buff_a, buff_b, 1000, 4
            );

        for (int i(0); i < samples_per_record*number_of_records; i++) {
            CPPUNIT_ASSERT( buff_a[0] != -1);
            CPPUNIT_ASSERT( buff_b[0] != -1);
        }


        ADQ_MultiRecordClose(adq_cu_ptr, 1);
        delete[] buff_a;
        delete[] buff_b;
    }
    void test_fetch_no_channel_data(){
        unsigned int number_of_records = 4;
        unsigned int samples_per_record = GetMaxNofRecordsFromNofSamples(adq_cu_ptr, number_of_records);

        short* buff_a = new short[samples_per_record*number_of_records];
        short* buff_b = new short[samples_per_record*number_of_records];

        for (int i(0); i < samples_per_record*number_of_records; i++) {
            buff_a[i] = -1;
            buff_b[i]= -1;
        }

        for (int i(0); i < samples_per_record*number_of_records; i++) {
            CPPUNIT_ASSERT( buff_a[0] == -1);
            CPPUNIT_ASSERT( buff_b[0] == -1);
        }

        delete[] buff_a;
        delete[] buff_b;
    }
};

// Add suite to Factory Register which will be retrieved later on
CPPUNIT_TEST_SUITE_REGISTRATION( ia_ADQAPITest );
