#include <celero/Celero.h>

#include <random>
#include "power_kernel.hpp"

// Macro for main
CELERO_MAIN

std::random_device RandomDevice;
std::uniform_int_distribution<int> UniformDistribution(0, 1024);

///
/// In reality, all of the "Complex" cases take the same amount of time to run.
/// The difference in the results is a product of measurement error.
///
/// Interestingly, taking the sin of a constant number here resulted in a
/// great deal of optimization in clang and gcc.
///
BASELINE(Power, CPU, 10, 100)
{
    // celero::DoNotOptimizeAway(static_cast<float>(sin(UniformDistribution(RandomDevice))));
    celero::DoNotOptimizeAway(GPU::fetch_kernel_parameters());
}

///
/// Run a test consisting of 1 sample of 710000 operations per measurement.
/// There are not enough samples here to likely get a meaningful result.
///
// BENCHMARK(Power, Complex1, 1, 710000)
// {
//         celero::DoNotOptimizeAway(static_cast<float>(sin(fmod(UniformDistribution(RandomDevice), 3.14159265))));
// }

// ///
// /// Run a test consisting of 30 samples of 710000 operations per measurement.
// /// There are not enough samples here to get a reasonable measurement
// /// It should get a Baseline number lower than the previous test.
// ///
// BENCHMARK(Power, Complex2, 30, 710000)
// {
//         celero::DoNotOptimizeAway(static_cast<float>(sin(fmod(UniformDistribution(RandomDevice), 3.14159265))));
// }

// ///
// /// Run a test consisting of 60 samples of 710000 operations per measurement.
// /// There are not enough samples here to get a reasonable measurement
// /// It should get a Baseline number lower than the previous test.
// ///
// BENCHMARK(Power, Complex3, 60, 710000)
// {
//         celero::DoNotOptimizeAway(static_cast<float>(sin(fmod(UniformDistribution(RandomDevice), 3.14159265))));
// }
