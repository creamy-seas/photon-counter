#include "power_pipeline.hpp"
#include "ia_ADQAPI.hpp"
#include "ADQAPI.h" // For MultiRecordSetup
#include "power_kernel.hpp"

int main(int argc, char *argv[])
{

    GPU::fetch_kernel_parameters();

    short* chA_background = new short[SP_POINTS]();
    short* chB_background = new short[SP_POINTS]();
    GPU::copy_background_arrays_to_gpu(chA_background, chB_background);

    void* adq_cu_ptr = master_setup(
        NO_BLINK,
        INTERNAL_CLOCK_SOURCE_INTERNAL_10MHZ_REFFERENCE,
        TRIGGER_EXTERNAL);

    ADQ214_MultiRecordSetup(adq_cu_ptr, 1, R_POINTS, SP_POINTS);

    run_power_measurements(adq_cu_ptr,
                           4,
                           "./test/test_bin/power-pipeline-example");

    ADQ214_MultiRecordClose(adq_cu_ptr, 1);

    DeleteADQControlUnit(adq_cu_ptr);
}
