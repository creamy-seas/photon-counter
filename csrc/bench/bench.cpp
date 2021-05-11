#include <celero/Celero.h>
#include <cstdlib> // srand
#include <ctime> // time
#include <cmath>
#include "power_kernel.hpp"

// Macro for main
CELERO_MAIN

#ifndef SAMPLES
#define SAMPLES 10
#endif

#ifndef ITERATIONS
#define ITERATIONS 1000 // or 10000
#endif

const int digitiser_code_range = std::pow(2, 14);
short digitiser_code() {
    return ((float)std::rand() / RAND_MAX - 0.5) * digitiser_code_range;
}

class PowerKernelFixture : public celero::TestFixture
{
public:
    void setUp(__attribute__ ((unused)) const celero::TestFixture::ExperimentValue& x) override {
        // Prepare arrays before each sample is run
        chA_data = new short[TOTAL_POINTS];
        chB_data = new short[TOTAL_POINTS];

        data_out = new double*[POWER_PROCESSING_CHANNELS];
        for (int i(0); i < POWER_PROCESSING_CHANNELS; i++) {
                data_out[i] = new double[R_POINTS];
        }
        // Seed generator and populate arrays
        std::srand(std::time(0));

        for (int i(0); i < TOTAL_POINTS; i++) {
            chA_data[i] = digitiser_code();
            chB_data[i] = digitiser_code();
        }

        // Background
        chA_const_background = digitiser_code();
        chB_const_background = digitiser_code();
        chA_background = new short[SP_POINTS];
        chB_background = new short[SP_POINTS];
        for (int i(0); i < SP_POINTS; i++) {
            chA_background[i] = digitiser_code();
            chB_background[i] = digitiser_code();
        }

        // Copy background data to GPU
        GPU::copy_background_arrays_to_gpu(chA_background, chB_background);
    };

    void tearDown() override {
        delete[] chA_data;
        delete[] chB_data;
        delete[] chA_background;
        delete[] chB_background;
        for (int i(0); i < POWER_PROCESSING_CHANNELS; i++)
            delete[] data_out[i];
        delete[] data_out;
    };

    short* chA_data;
    short* chB_data;
    double** data_out;

    // Background
    short chA_const_background;
    short chB_const_background;
    short* chA_background;
    short* chB_background;

    // Allocation on GPU
    short *dev_chA_data;
    short *dev_chB_data;
    double *dev_chA_out;
    double *dev_chB_out;
    double *dev_chAsq_out;
    double *dev_chBsq_out;
    double *dev_sq_out;
};

BASELINE_F(POWER, CPU_1T_NO_BACK, PowerKernelFixture, SAMPLES, ITERATIONS)
{
    CPU::power_kernel_v1_no_background(
        chA_data, chB_data, data_out,
        SQ_MASK,
        SP_POINTS, R_POINTS, 1
        );
}

BENCHMARK_F(POWER, CPU_2T_NO_BACK, PowerKernelFixture, SAMPLES, ITERATIONS)
{
    CPU::power_kernel_v1_no_background(
        chA_data, chB_data, data_out,
        SQ_MASK,
        SP_POINTS, R_POINTS, 2
        );
}

BENCHMARK_F(POWER, CPU_8T_NO_BACK, PowerKernelFixture, SAMPLES, ITERATIONS)
{
    CPU::power_kernel_v1_no_background(
        chA_data, chB_data, data_out,
        SQ_MASK,
        SP_POINTS, R_POINTS, 8
        );
}

BENCHMARK_F(POWER, CPU_1T_NO_BACK_FULL_MASK, PowerKernelFixture, SAMPLES, ITERATIONS)
{
    CPU::power_kernel_v1_no_background(
        chA_data, chB_data, data_out,
        SQ_MASK ^ CHA_MASK ^ CHB_MASK ^ CHBSQ_MASK ^ CHASQ_MASK,
        SP_POINTS, R_POINTS, 1
        );
}


BENCHMARK_F(POWER, CPU_1T_CONST_BACK, PowerKernelFixture, SAMPLES, ITERATIONS)
{
    CPU::power_kernel_v2_const_background(
        chA_data, chB_data, data_out,
        SQ_MASK,
        chA_const_background, chB_const_background,
        SP_POINTS, R_POINTS, 1
        );
}

BENCHMARK_F(POWER, CPU_1T_CONST_BACK_FULL_MASK, PowerKernelFixture, SAMPLES, ITERATIONS)
{
    CPU::power_kernel_v2_const_background(
        chA_data, chB_data, data_out,
        SQ_MASK ^ CHA_MASK ^ CHB_MASK ^ CHBSQ_MASK ^ CHASQ_MASK,
        chA_const_background, chB_const_background,
        SP_POINTS, R_POINTS, 1
        );
}

BENCHMARK_F(POWER, CPU_1T_BACK, PowerKernelFixture, SAMPLES, ITERATIONS)
{
    CPU::power_kernel_v3_background(
        chA_data, chB_data, data_out,
        SQ_MASK,
        chA_background, chB_background,
        SP_POINTS, R_POINTS, 1
        );
}

BENCHMARK_F(POWER, CPU_1T_BACK_FULL_MASK, PowerKernelFixture, SAMPLES, ITERATIONS)
{
    CPU::power_kernel_v3_background(
        chA_data, chB_data, data_out,
        SQ_MASK ^ CHA_MASK ^ CHB_MASK ^ CHBSQ_MASK ^ CHASQ_MASK,
        chA_background, chB_background,
        SP_POINTS, R_POINTS, 1
        );
}

BENCHMARK_F(POWER, GPU_BACK, PowerKernelFixture, SAMPLES, ITERATIONS)
{
    GPU::allocate_memory_on_gpu(&dev_chA_data, &dev_chB_data,
                                &dev_chA_out, &dev_chB_out,
                                &dev_chAsq_out, &dev_chBsq_out, &dev_sq_out);
    GPU::power_kernel(
        chA_data,
        chB_data,
        data_out,
        &dev_chA_data, &dev_chB_data,
        &dev_chA_out, &dev_chB_out,
        &dev_chAsq_out, &dev_chBsq_out, &dev_sq_out
        );
    GPU::free_memory_on_gpu(&dev_chA_data, &dev_chB_data,
                            &dev_chA_out, &dev_chB_out,
                            &dev_chAsq_out, &dev_chBsq_out, &dev_sq_out);
}

///////////////////////////////////////////////////////////////////////////////
//                         Float vs double benchmark                         //
///////////////////////////////////////////////////////////////////////////////
// class TypeFixture : public celero::TestFixture
// {
// public:
//     const int points = 1000;
//     float* float_array;
//     double* double_array;

//     void setUp(__attribute__ ((unused)) const celero::TestFixture::ExperimentValue& x) override {

//         float_array = new float[points];
//         double_array = new double[points];

//         // Seed generator and populate arrays
//         std::srand(std::time(0));

//         for (int i(0); i < points; i++) {
//             float_array[i] = ((float)std::rand());
//             double_array[i] = ((double)std::rand());
//         }
//     };

//     void tearDown() override {
//         delete[] float_array;
//         delete[] double_array;
//     };
// };
// BASELINE_F(TYPE_BENCHMARK, FLOAT_MULITPLY, TypeFixture, 1000, 50000){
//     for (int i(0); i < points; i++)
//         float_array[i] = float_array[i] * float_array[i];
// }

// BENCHMARK_F(TYPE_BENCHMARK, FLOAT_ADD, TypeFixture, 1000, 50000){
//     for (int i(0); i < points; i++)
//         float_array[i] = float_array[i] + float_array[i];
// }

// BENCHMARK_F(TYPE_BENCHMARK, DOUBLE_ADD, TypeFixture, 1000, 50000){
//     for (int i(0); i < points; i++)
//         double_array[i] = double_array[i] + double_array[i];
// }

// BENCHMARK_F(TYPE_BENCHMARK, DOUBLE_MULTIPLY, TypeFixture, 1000, 50000){
//     for (int i(0); i < points; i++)
//         double_array[i] = double_array[i] * double_array[i];
// }
