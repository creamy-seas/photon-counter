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
#define ITERATIONS 10000
#endif

class PowerKernelFixture : public celero::TestFixture
{
public:

    short digitiser_code() {
        return ((float)std::rand() / RAND_MAX - 0.5) * digitiser_code_range;
    }

    void setUp(__attribute__ ((unused)) const celero::TestFixture::ExperimentValue& x) override {
        // Prepare arrays before each sample is run
        chA_data = new short[TOTAL_POINTS];
        chB_data = new short[TOTAL_POINTS];
        sq_data_gpu = new float[R_POINTS];
        sq_data_cpu = new unsigned int[TOTAL_POINTS];

        // Seed generator and populate arrays
        std::srand(std::time(0));

        for (int i(0); i < TOTAL_POINTS; i++) {
            chA_data[i] = digitiser_code();
            chB_data[i] = digitiser_code();
        }

        chA_const_background = digitiser_code();
        chB_const_background = digitiser_code();
    };

    void tearDown() override {
        delete[] chA_data;
        delete[] chB_data;
        delete[] sq_data_gpu;
        delete[] sq_data_cpu;
    };

    const int digitiser_code_range = std::pow(2, 14);

    // Same input arrays, but the GPU will make it compact
    short* chA_data;
    short* chB_data;
    unsigned int* sq_data_cpu;
    float* sq_data_gpu;

    short chA_const_background;
    short chB_const_background;

    // Allocation on GPU
    short *dev_chA_data;
    short *dev_chB_data;
    float *dev_sq_data;
};

BASELINE_F(PowerNoBack, CPU_1_thread, PowerKernelFixture, SAMPLES, ITERATIONS)
{
    CPU::power_kernel_v1_no_background(
        chA_data, chB_data, sq_data_cpu,
        TOTAL_POINTS, 1
        );
}

BENCHMARK_F(PowerNoBack, CPU_2_thread, PowerKernelFixture, SAMPLES, ITERATIONS)
{
    CPU::power_kernel_v1_no_background(
        chA_data, chB_data, sq_data_cpu,
        TOTAL_POINTS, 2
        );
}

BENCHMARK_F(PowerNoBack, CPU_4_thread, PowerKernelFixture, SAMPLES, ITERATIONS)
{
    CPU::power_kernel_v1_no_background(
        chA_data, chB_data, sq_data_cpu,
        TOTAL_POINTS, 4
        );
}

BENCHMARK_F(PowerNoBack, CPU_8_thread, PowerKernelFixture, SAMPLES, ITERATIONS)
{
    CPU::power_kernel_v1_no_background(
        chA_data, chB_data, sq_data_cpu,
        TOTAL_POINTS, 8
        );
}

BENCHMARK_F(PowerNoBack, GPU, PowerKernelFixture, SAMPLES, ITERATIONS)
{
    GPU::allocate_memory_on_gpu(&dev_chA_data, &dev_chB_data, &dev_sq_data);
    GPU::power_kernel_v1_no_background(
        chA_data,
        chB_data,
        sq_data_gpu,
        &dev_chA_data,
        &dev_chB_data,
        &dev_sq_data
        );
    GPU::free_memory_on_gpu(&dev_chA_data, &dev_chB_data, &dev_sq_data);
}

BASELINE_F(PowerConstBack, CPU_1_thread, PowerKernelFixture, SAMPLES, ITERATIONS)
{
    CPU::power_kernel_v2_const_background(
        chA_data, chB_data, sq_data_cpu,
        chA_const_background, chB_const_background,
        TOTAL_POINTS, 1
        );
}

BENCHMARK_F(PowerConstBack, GPU, PowerKernelFixture, SAMPLES, ITERATIONS)
{
    GPU::allocate_memory_on_gpu(&dev_chA_data, &dev_chB_data, &dev_sq_data);
    GPU::power_kernel_v2_const_background(
        chA_data,
        chB_data,
        sq_data_gpu,
        chA_const_background,
        chB_const_background,
        &dev_chA_data,
        &dev_chB_data,
        &dev_sq_data
        );
    GPU::free_memory_on_gpu(&dev_chA_data, &dev_chB_data, &dev_sq_data);
}
