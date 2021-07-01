#include <celero/Celero.h>
#include <cstdlib> // srand
#include <ctime> // time
#include <cmath>
#include "power_kernel.hpp"

#include "utils.hpp"
#include "sp_digitiser.hpp" // fetch_digitiser_data
#include "ADQAPI.h" // DeleteAdqControlUnit
#include "power_pipeline.hpp"
#include "g1_kernel.hpp"

CELERO_MAIN

#ifndef SAMPLES
#define SAMPLES 0 // 10 seems good, or 0 for Celero to choose for you
#endif

#ifndef ITERATIONS_PER_SAMPLE
#define ITERATIONS_PER_SAMPLE 0 // or 10000
#endif

const int digitiser_code_range = std::pow(2, 14);
short digitiser_code() {
    return ((float)std::rand() / RAND_MAX - 0.5) * digitiser_code_range;
}

///////////////////////////////////////////////////////////////////////////////
//                              Digitiser                                    //
///////////////////////////////////////////////////////////////////////////////
class DigitiserFixture : public celero::TestFixture {
public:
    void setUp(__attribute__ ((unused)) const celero::TestFixture::ExperimentValue& x) override {

        // Create pointer and set up device for multirecord
        adq_cu_ptr = master_setup(NO_BLINK,
                                  INTERNAL_CLOCK_SOURCE_INTERNAL_10MHZ_REFFERENCE,
                                  TRIGGER_EXTERNAL
                                  // TRIGGER_SOFTWARE
            );

        // Simulate with real number of points that we would typically expect
        number_of_records = 10000;//GetMaxNofRecordsFromNofSamples(adq_cu_ptr, SP_POINTS);
        buff_a = new short[SP_POINTS * number_of_records];
        buff_b = new short[SP_POINTS * number_of_records];
    };

    void tearDown() override {
        delete[] buff_a;
        delete[] buff_b;
        DeleteADQControlUnit(adq_cu_ptr);
    };

    void* adq_cu_ptr;
    unsigned int number_of_records;
    short *buff_a;
    short *buff_b;
};

// BASELINE_F(POWER, READING, DigitiserFixture, 0, 0)
// {
//     // Prepare multirecord mode
//     ADQ_MultiRecordSetup(adq_cu_ptr, 1, number_of_records,  SP_POINTS);
//     fetch_digitiser_data(adq_cu_ptr,
//                          buff_a, buff_b,
//                          SP_POINTS, number_of_records);
//     ADQ_MultiRecordClose(adq_cu_ptr, 1);
// }

///////////////////////////////////////////////////////////////////////////////
//                                Power Kernel                               //
///////////////////////////////////////////////////////////////////////////////
class PowerKernelFixture : public celero::TestFixture {
public:
    void setUp(__attribute__ ((unused)) const celero::TestFixture::ExperimentValue& x) override {
        // Seed generator for population of arrays
        std::srand(std::time(0));

        // Prepare arrays before each sample is run
        chA_data = new short[SP_POINTS * R_POINTS];
        chB_data = new short[SP_POINTS * R_POINTS];
        for (int i(0); i < SP_POINTS * R_POINTS; i++) {
            chA_data[i] = digitiser_code();
            chB_data[i] = digitiser_code();
        }

        data_out = new long*[POWER::no_outputs];
        for (int i(0); i < POWER::no_outputs; i++)
            data_out[i] = new long[SP_POINTS];

        // Background
        chA_background = new short[SP_POINTS];
        chB_background = new short[SP_POINTS];
        for (int i(0); i < SP_POINTS; i++) {
            chA_background[i] = digitiser_code();
            chB_background[i] = digitiser_code();
        }
    };

    void tearDown() override {
        delete[] chA_data;
        delete[] chB_data;
        delete[] chA_background;
        delete[] chB_background;
        for (int i(0); i < POWER::no_outputs; i++)
            delete[] data_out[i];
        delete[] data_out;
    };

    short* chA_data;
    short* chB_data;
    long** data_out;

    // Background
    short *chA_background;
    short *chB_background;
};

class PowerKernelGPUBaseFixture : public PowerKernelFixture {
public:
    short *chA_data_locked;
    short *chB_data_locked;
    short ***gpu_in; long ***gpu_out; long ***cpu_out;

    // Deriving classes will define number of streams used
    int no_streams;
    virtual int get_number_of_streams() =0;

    void setUp(__attribute__ ((unused)) const celero::TestFixture::ExperimentValue& x) override {
        PowerKernelFixture::setUp(x);

        no_streams = get_number_of_streams();

        // Validate kernel
        POWER::GPU::check_power_kernel_parameters();

        // Copy background data to GPU
        POWER::GPU::copy_background_arrays_to_gpu(chA_background, chB_background);

        // Allocate memory
        POWER::GPU::allocate_memory(&chA_data_locked, &chB_data_locked,
                                    &gpu_in, &gpu_out, &cpu_out, no_streams);
        // Copy over test input data to the locked memory
        for (int i(0); i < SP_POINTS; i++) {
            chA_data_locked[i] = chA_data[i];
            chB_data_locked[i] = chB_data[i];
        }
    };

    void tearDown() override {
        PowerKernelFixture::tearDown();
        POWER::GPU::free_memory(chA_data_locked, chB_data_locked,
                                gpu_in, gpu_out, cpu_out, no_streams);
    };
};

class PowerKernelGPU1StreamFixture : public PowerKernelGPUBaseFixture {
public:
    int get_number_of_streams() {return 1;}
};
class PowerKernelGPU2StreamFixture : public PowerKernelGPUBaseFixture {
public:
    int get_number_of_streams() {return 2;}
};
class PowerKernelGPU8StreamFixture : public PowerKernelGPUBaseFixture {
public:
    int get_number_of_streams() {return 8;}
};
class PowerKernelGPU16StreamFixture : public PowerKernelGPUBaseFixture {
public:
    int get_number_of_streams() {return 16;}
};

// BENCHMARK_F(POWER, 1T_BACK, PowerKernelFixture, SAMPLES, ITERATIONS_PER_SAMPLE) {
//     POWER::CPU::power_kernel(
//         chA_data, chB_data, data_out,
//         SQ_MASK,
//         chA_background, chB_background,
//         SP_POINTS, R_POINTS, 1
//         );
// }

// BASELINE_F(POWER, 1T_BACK_FULL_MASK, PowerKernelFixture, SAMPLES, ITERATIONS_PER_SAMPLE) {
//     POWER::CPU::power_kernel(
//         chA_data, chB_data, data_out,
//         SQ_MASK ^ CHA_MASK ^ CHB_MASK ^ CHBSQ_MASK ^ CHASQ_MASK,
//         chA_background, chB_background,
//         SP_POINTS, R_POINTS, 1
//         );
// }

// BENCHMARK_F(POWER, GPU_1ST, PowerKernelGPU1StreamFixture, SAMPLES, ITERATIONS_PER_SAMPLE) {
//     POWER::GPU::power_kernel(
//         chA_data_locked, chB_data_locked,
//         data_out,
//         gpu_in, gpu_out, cpu_out, no_streams);
// }
// BENCHMARK_F(POWER, GPU_2ST, PowerKernelGPU2StreamFixture, SAMPLES, ITERATIONS_PER_SAMPLE) {
//     POWER::GPU::power_kernel(
//         chA_data_locked, chB_data_locked,
//         data_out,
//         gpu_in, gpu_out, cpu_out, no_streams);
//     dump_arrays_to_file(data_out, 5, SP_POINTS,
//                         "./dump/bench-example.txt",
//                         "#CHA\tCHB\tCHASQ\tCHBSQ\tSQ",
//                         (double)1
//         );
// }
// BENCHMARK_F(POWER, GPU_8ST, PowerKernelGPU8StreamFixture, SAMPLES, ITERATIONS_PER_SAMPLE) {
//     POWER::GPU::power_kernel(
//         chA_data_locked, chB_data_locked,
//         data_out,
//         gpu_in, gpu_out, cpu_out, no_streams);
// }
// BENCHMARK_F(POWER, GPU_16ST, PowerKernelGPU16StreamFixture, SAMPLES, ITERATIONS_PER_SAMPLE) {
//     POWER::GPU::power_kernel(
//         chA_data_locked, chB_data_locked,
//         data_out,
//         gpu_in, gpu_out, cpu_out, no_streams);
// }
// BENCHMARK_F(POWER, FILE_WRITTING, PowerKernelGPU2StreamFixture, SAMPLES, ITERATIONS_PER_SAMPLE) {
//     dump_arrays_to_file(data_out, 5, SP_POINTS,
//                         "./dump/bench-example.txt",
//                         "#CHA\tCHB\tCHASQ\tCHBSQ\tSQ");
// }
// BENCHMARK_F(POWER, PROCESSING, PowerKernelGPU2StreamFixture, SAMPLES, ITERATIONS_PER_SAMPLE) {
//     process_digitiser_data(chA_data_locked, chB_data_locked,
//                            data_out,
//                            gpu_in, gpu_out, cpu_out, no_streams,
//                            1, "./dump/bench_processing-example");
// }

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

///////////////////////////////////////////////////////////////////////////////
//                                G1 Benchmark                               //
///////////////////////////////////////////////////////////////////////////////
class G1Kernel_CPU_DIRECT : public celero::TestFixture {
public:
    short* chA_data;
    short* chB_data;
    double** data_out;
    const int tau_points = 100;

    void setUp(__attribute__ ((unused)) const celero::TestFixture::ExperimentValue& x) override {
        // Seed generator for population of arrays
        std::srand(std::time(0));
        chA_data = new short[G1_DIGITISER_POINTS]();
        chB_data = new short[G1_DIGITISER_POINTS]();
        for (int i(0); i < G1_DIGITISER_POINTS; i++) {
            chA_data[i] = digitiser_code();
            chB_data[i] = digitiser_code();
        }

        data_out = new double*[G1::no_outputs];
        for (int i(0); i < G1::no_outputs; i++)
            data_out[i] = new double[tau_points]();
    };

    void tearDown() override {
        delete[] chA_data;
        delete[] chB_data;
        for (int i(0); i < G1::no_outputs; i++)
            delete[] data_out[i];
        delete[] data_out;
    };
};

class G1Kernel_CPU_FFTW : public celero::TestFixture {
public:
    short* chA_data;
    short* chB_data;
    double** data_out;
    fftw_complex **aux_array;
    fftw_plan *plans_forward, *plans_backward;
    const int tau_points = 100;

    virtual std::string get_plan_name() = 0;

    void setUp(__attribute__ ((unused)) const celero::TestFixture::ExperimentValue& x) override {
        chA_data = new short[G1_DIGITISER_POINTS]();
        chB_data = new short[G1_DIGITISER_POINTS]();

        // Pre-kernel setup
        G1::CPU::FFTW::g1_allocate_memory(data_out, aux_array, "./dump/bench-1-thread-plan",
                                          plans_forward,
                                          plans_backward);
    };

    void tearDown() override {
        delete[] chA_data;
        delete[] chB_data;
        G1::CPU::FFTW::g1_free_memory(data_out, aux_array, plans_forward, plans_backward);
    };
};

class G1Kernel_CPU_FFTW1Threads : public G1Kernel_CPU_FFTW {
public:
    std::string get_plan_name() override {
        return "../dump/bench-1-thread-plan";
    }
};

class G1Kernel_CPU_FFTW2Threads : public G1Kernel_CPU_FFTW {
public:
    std::string get_plan_name() override {
        return "bench-2-thread-plan";
    }
};

class G1Kernel_CPU_FFTW4Threads : public G1Kernel_CPU_FFTW {
public:
    std::string get_plan_name() override {
        return "bench-4-thread-plan";
    }
};

class G1Kernel_CPU_FFTW8Threads : public G1Kernel_CPU_FFTW {
public:
    std::string get_plan_name() override {
        return "bench-8-thread-plan";
    }
};

class G1Kernel_GPU : public celero::TestFixture {
public:
    cufftHandle *plans_forward; cufftHandle *plans_backward;
    short* chA_data, *chB_data;
    cufftReal **gpu_inout;
    cufftComplex **gpu_aux;
    float **cpu_out;

    float **preprocessed_data;
    float variance_list[3] = {1, 1, 1};

    void setUp(__attribute__ ((unused)) const celero::TestFixture::ExperimentValue& x) override {

        G1::GPU::g1_prepare_fftw_plan(plans_forward, plans_backward);

        preprocessed_data = new float*[G1::no_outputs];
        for (int i(0); i < G1::no_outputs; i++)
            preprocessed_data[i] = new float[G1_DIGITISER_POINTS];

        G1::GPU::allocate_memory(chA_data, chB_data, gpu_inout, gpu_aux, cpu_out);

        // Validate kernel
        G1::check_g1_kernel_parameters(false);
    };

    void tearDown() override {
        G1::GPU::free_memory(chA_data, chB_data, gpu_inout, gpu_aux, cpu_out);
    };
};


// BASELINE_F(G1, READING, DigitiserFixture, 0, 0)
// {
//     // Prepare multirecord mode
//     ADQ_MultiRecordSetup(adq_cu_ptr, 1, 1, G1_DIGITISER_POINTS);
//     fetch_digitiser_data(adq_cu_ptr,
//                          buff_a, buff_b,
//                          G1_DIGITISER_POINTS, 1);
//     ADQ_MultiRecordClose(adq_cu_ptr, 1);
// }

// BENCHMARK_F(G1, DIRECT_1T, G1Kernel_CPU_DIRECT, 0, 0) {
//     int no_threads = 1;
//     G1::CPU::DIRECT::g1_kernel(chA_data, chB_data, data_out, tau_points, false, no_threads);
// }
// BENCHMARK_F(G1, DIRECT_2T, G1Kernel_CPU_DIRECT, 0, 0) {
//     int no_threads = 2;
//     G1::CPU::DIRECT::g1_kernel(chA_data, chB_data, data_out, tau_points, false, no_threads);
// }
// BENCHMARK_F(G1, DIRECT_4T, G1Kernel_CPU_DIRECT, 0, 0) {
//     int no_threads = 4;
//     G1::CPU::DIRECT::g1_kernel(chA_data, chB_data, data_out, tau_points, false, no_threads);
// }
// BENCHMARK_F(G1, DIRECT_8T, G1Kernel_CPU_DIRECT, 0, 0) {
//     int no_threads = 8;
//     G1::CPU::DIRECT::g1_kernel(chA_data, chB_data, data_out, tau_points, false, no_threads);
// }
// BENCHMARK_F(G1, DIRECT_16T, G1Kernel_CPU_DIRECT, 0, 0) {
//     int no_threads = 16;
//     G1::CPU::DIRECT::g1_kernel(chA_data, chB_data, data_out, tau_points, false, no_threads);
// }
BASELINE_F(G1, FFTW_1T, G1Kernel_CPU_FFTW1Threads, 0, 0)
// BENCHMARK_F(G1, FFTW_1T, G1Kernel_CPU_FFTW1Threads, 0, 0)
{
    G1::CPU::FFTW::g1_kernel(chA_data, chB_data,
                             data_out, aux_array,
                             plans_forward, plans_backward);
}
// BENCHMARK_F(G1, FFTW_2T, G1Kernel_CPU_FFTW2Threads, 0, 0) {
//     G1::CPU::FFTW::g1_kernel(chA_data, chB_data,
//                              data_out, aux_array,
//                              plans_forward, plans_backward);
// }
// BENCHMARK_F(G1, FFTW_4T, G1Kernel_CPU_FFTW4Threads, 0, 0) {
//     G1::CPU::FFTW::g1_kernel(chA_data, chB_data,
//                              data_out, aux_array,
//                              plans_forward, plans_backward);
// }
// BASELINE_F(G1, FFTW_8T, G1Kernel_CPU_FFTW8Threads, 0, 0)
// BENCHMARK_F(G1, FFTW_8T, G1Kernel_CPU_FFTW8Threads, 0, 0)
// {
//     G1::CPU::FFTW::g1_kernel(chA_data, chB_data,
//                              data_out, aux_array,
//                              plans_forward, plans_backward);
// }
BENCHMARK_F(G1, GPU, G1Kernel_GPU, 0, 0)
// BENCHMARK_F(G1, FFTW_1T, G1Kernel_CPU_FFTW1Threads, 0, 0)
{
    G1::GPU::g1_kernel(preprocessed_data, variance_list, gpu_inout, gpu_aux, cpu_out, plans_forward, plans_backward);
}
