#include "power_pipeline.hpp"
#include "sp_digitiser.hpp"
#include "ADQAPI.h" // For MultiRecordSetup
#include "power_kernel.hpp"

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

    GPU::check_power_kernel_parameters();

    short* chA_background = new short[SP_POINTS]();
    short* chB_background = new short[SP_POINTS]();
    GPU::copy_background_arrays_to_gpu(chA_background, chB_background);

    void* adq_cu_ptr = master_setup(
        NO_BLINK,
        INTERNAL_CLOCK_SOURCE_INTERNAL_10MHZ_REFFERENCE,
#ifdef TESTENV
        TRIGGER_SOFTWARE
#else
        TRIGGER_EXTERNAL
#endif
        );

    ADQ214_MultiRecordSetup(adq_cu_ptr, 1, R_POINTS, SP_POINTS);

    run_power_measurements(adq_cu_ptr,
                           4,
                           "./dump/power-pipeline-example");

    ADQ214_MultiRecordClose(adq_cu_ptr, 1);

    DeleteADQControlUnit(adq_cu_ptr);

    // test_connection();
}
