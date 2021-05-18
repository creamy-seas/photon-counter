# Project setup #

1. Create `.dir-locals.el` listing location of standard libraries and header files.

```elisp
((nil
  (company-clang-arguments . ("-I/usr/include"
                              "-I/home/antonov/photon_counting/csrc/include"
                              "-I/usr/local/cuda-11.0/include"
                              "-std=c++11"))
  (flycheck-clang-include-path . ("/home/antonov/photon_counting/csrc/include"
                                  "/usr/local/cuda-11.0/include"
                                  ))
  (flycheck-clang-definitions . ("LINUX")) ;; -D flags
  )
 (c++-mode
  (flycheck-clang-language-standard . "c++11")))
```

2. Run `ggtags-create-tags` in an c/cpp project. or `gtags`

3. Install `cppunit, gcov, lcov` lib

```shell
sudo apt-get install libcppunit-dev lcov
```

4. Install `Celero` for benchmarking
```shell
git submodule update --init --recursive
cd Celero
mkdir build && cd build && cmake ..
make
```

# Test coverage

```shell
cd csrc && make test
```

https://quantum-optics-ride.gitlab.io/photon-counting/

# Benchmarking #
`celero` will run N samples, where each sample will run M operations.

```shell
cd csrc && make bench
```

# Links #
- https://stackoverflow.com/questions/242894/cuda-driver-api-vs-cuda-runtime
- https://stackoverflow.com/questions/17278932/cuda-shared-library-linking-undefined-reference-to-cudaregisterlinkedbinary

# Choice #
I have though about c over cpp - the first being less bloated

| C    | Cpp                                                    |
|------|--------------------------------------------------------|
| Slim | Will be required for unit tests                        |
|      | Can overload functions - nice for consistent interface |
|      | Can throw exceptions                                   |
|      | Threads?                                               |

# Python-c notes #
- Export with C linkage to not mangle the function names. **This will prevent function overloading!** (function with the same name are not allowed)
```text
#ifdef __cplusplus
extern "C" {
#endif

        namespace GPU {
                float example_gpu_func(
                        short a,
                        short b);
        }

#ifdef __cplusplus
}
#endif
```
- Load in python and define return types if they are not integers
```python
import ctypes
from ctypes import cdll

ia_ADQAPI = cdll.LoadLibrary("./bin/ia_1488.so")
ia_ADQAPI.example_gpu_func.restype = ctypes.c_float

print(ia_ADQAPI.example_gpu_func(10,2))
```

# Cuda notes #
- Use cuda-11.0
- `nvcc` will automatically lookup relevant headers and libraries, so it can be used for compilation. It can even pass the non-gpu code to the standard compuler. But it will not be able to inject `-fprofile-arcs` and `-ftest-coverage` so it is better to use it for only building object files and not the total compilation.

# Tools #
- Check the objects in a library file
```shell
nm ./bin/ia_1488.so
```

- Check dependenices of library
```shell
ldd ./bin/ia_1488.so
```

# Build #
- `gcc 4.8.5`


# Implementation Table #
> - 🏆 indicates the language of choice. Something in C++ will be wrapped for calling from python;
> - `funcA` indicates a function to use

## SP-Digitiser Setup ##
> Complete

| `sp_digitizer.py` (Python) | `ia_ADQAPI.cpp` (C++)                 |
|:---------------------------|---------------------------------------|
| 🏆 extensive setup         | essential setup only for tests        |
|                            | 🏆 `fetch_channel_data` to fetch data |

## GPU Utils ##
> Complete

| `gpu_utils.py` (Python) | `gpu_utils.hpp` (C++)  |
|:------------------------|------------------------|
| `fetch_gpu_parameters`  | `fetch_gpu_parameters` |
| `verify_gpu_allocation` |                        |
| `allocate_threads`      |                        |
| 🏆                      | Only for development   |

## Power ##
> Complete
- [ ] Safety checks to ensure that correct allocation is made

| `power_kernel.py` (Python)                                        | `power_kernel.hpp` (C++)                                                 |
|:------------------------------------------------------------------|--------------------------------------------------------------------------|
| "Out-of-the-box" `kernel` with little customisability             | 🏆 `kernel` that can be used in parallel processes                       |
| 🏆 use `check_kernel_parameters` to check the kernel built in C++ | feed `fetch_kernel_parameters` into the python `check_kernel_parameters` |

## Power pipeline ##
> In progress

| `sp_digitiser_power.py` (Python) | (C++)                                              |
|:---------------------------------|----------------------------------------------------|
| allocation of arrays             | allocation of arrays                               |
| call in order:                   | 3 threads (read, process, save) in single function |
| - C++: `fetch_channel_data`      |                                                    |
| - C++: `power_kernel_vXXXX`      |                                                    |
| - Python: Post processing        |                                                    |
|                                  | 🏆 just need to make sure that data dumping safe   |

## Correlation ##

| (Python) | (C++) |
|:---------|-------|
|          |       |

## Correlation Pipeline ##

| (Python) | (C++) |
|:---------|-------|
|          |       |

# Colours #

| Colour | Meaning             |
|--------|---------------------|
| BLUE   | Information         |
| GREEN  | Something compelted |
| YELLOW | Irrelevant/debug    |
| RED    | Error               |
