#include "power_pipeline.hpp"
#include "sp_digitiser.hpp"
#include "ADQAPI.h" // For MultiRecordSetup
#include "power_kernel.hpp"
#include "g1_kernel.hpp"

#include "logging.hpp" // TODO: remove
#include <iostream> // TODO: remove

void test_connection() {
    void* adq_cu_ptr = CreateADQControlUnit();
    ADQInfoListEntry* retList = new ADQInfoListEntry[2];
    unsigned int len = 2;

    ADQControlUnit_ListDevices(adq_cu_ptr, &retList, &len);
    std::cout << retList[0].AddressField1 << "\n";
    std::cout << retList[0].AddressField2 << "\n";
    std::cout << retList[0].DeviceInterfaceOpened << "\n";
    std::cout << retList[0].DeviceSetupCompleted << "\n";
    std::cout << retList[0].DevFile << "\n";

    std::cout << "Opening Device: " << ADQControlUnit_OpenDeviceInterface(adq_cu_ptr, 0) << "\n";
    std::cout << "Setup Device:" << ADQControlUnit_SetupDevice(adq_cu_ptr, 0) << "\n";

    ADQControlUnit_ListDevices(adq_cu_ptr, &retList, &len);
    std::cout << retList[0].AddressField1 << "\n";
    std::cout << retList[0].AddressField2 << "\n";
    std::cout << retList[0].DeviceInterfaceOpened << "\n";
    std::cout << retList[0].DeviceSetupCompleted << "\n";
    std::cout << retList[0].DevFile << "\n";

    DeleteADQControlUnit(adq_cu_ptr);
}

int main()
{
    // test_connection();

//     GPU::check_power_kernel_parameters();

//     // Genertic setup of digitiser
//     void* adq_cu_ptr = master_setup(
//         NO_BLINK,
//         INTERNAL_CLOCK_SOURCE_INTERNAL_10MHZ_REFFERENCE,
// #ifdef TESTENV
//         TRIGGER_SOFTWARE
// #else
//         TRIGGER_EXTERNAL
// #endif
//         );

//     // Setting of background data
//     short* chA_background = new short[SP_POINTS]();
//     for (int i(0); i < SP_POINTS; i++) {
//         chA_background[i] = 0;
//     }
//     short* chB_background = new short[SP_POINTS]();

//     run_power_measurements(adq_cu_ptr,
//                            chA_background, chB_background,
//                            10,
//                            "./dump/power-pipeline-example");

//     DeleteADQControlUnit(adq_cu_ptr);

    // test_connection();

    // Generate plans
    int time_limit = 100;
    int no_threads = 1;
    std::string plan_name = "./dump/bench-1-thread-plan";
    G1::CPU::FFTW::g1_prepare_fftw_plan(plan_name, time_limit, no_threads);

    // if (!fftw_init_threads()) FAIL("Failed to init threads!");
    // if (!fftw_import_wisdom_from_filename("./dump/bench-1-thread-plan_forward.wis")) FAIL("Failed to load wisdom file ");
}
