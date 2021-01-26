# Project setup #

1. Create `.dir-locals.el` listing location of standard libraries and header files.

```elisp
((nil . (
         (company-clang-arguments . ("-I/usr/include"
                                     "-I/home/antonov/photon_counting/c_lib/includes"
                                     "-std=c++11"))
         (flycheck-clang-include-path . ("/home/antonov/photon_counting/c_lib/includes"))
         (flycheck-clang-definitions . ("LINUX")) ;; -D flags
         (flycheck-clang-language-standard . ("-std=c++11"))
         )))
```

2. Run `ggtags-create-tags` in an c/cpp project. or `gtags`

3. Install `cppunit, gcov, lcov` lib

```shell
sudo apt-get install libcppunit-dev lcov
```

4. Install `Celeto` for benchmarking
```shell
git submodule update --init --recursive
cd Celero
mkdir build && cd build && cmake ..
make
```

# Test coverage

https://quantum-optics-ride.gitlab.io/experimental/photon-counting/

# Links #
- https://stackoverflow.com/questions/242894/cuda-driver-api-vs-cuda-runtime
- https://stackoverflow.com/questions/17278932/cuda-shared-library-linking-undefined-reference-to-cudaregisterlinkedbinary

# Cuda notes #
- Use cuda-11.0
- `nvcc` will automatically lookup relevant headers and libraries, so it can be used for compilation. It can even pass the non-gpu code to the standard compuler. But it will not be able to inject `-fprofile-arcs` and `-ftest-coverage` so it is better to use it for only building object files and not the total compilation.

# Build #
- `gcc 4.8.5`


# Implementation Table #

## SP-Digitiser Setup ##

| `sp_digitizer.py` (Python) | `ia_ADQAPI.cpp` (C++)           |
|:---------------------------|---------------------------------|
| - extensive setup          | - essential setup                 |
| - preprocessing            | - wraps some functions for python |
| 🏆                         | Required only for tests         |

## GPU Utils ##

| `gpu_info.py` (Python)    | `utils_gpu.cpp` (C++)         |
|:--------------------------|-------------------------------|
| - `fetch_gpu_parameters`  | `fetch_gpu_parameters`        |
| - `verify_gpu_allocation` |                               |
| - `allocate_threads`      | `allocate_threads`            |
| 🏆                        | Use macro to allocate threads |

## Power ##

| `power_kernel.py` (Python) | `power_gpu.cu/cpp` (C++) |
|:---------------------------|--------------------------|
| - `verify_gpu_allocation`  |                          |
| - customizeable `kernel`   | - parallel `kernel`      |
| use to check allocated GPU | 🏆                       |

## Power pipeline ##

| `sp_digitiser_power.py` (Python) | (C++)                  |
|:---------------------------------|------------------------|
| - allocation of arrays           | - allocation of arrays |
| - call C++ lib                   |                        |
| 🏆                               | 🏆                     |

## Correlation ##

| (Python) | (C++) |
|:---------|-------|
|          |       |

## Correlation Pipeline ##

| (Python) | (C++) |
|:---------|-------|
|          |       |
