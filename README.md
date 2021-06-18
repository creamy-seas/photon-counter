Overview {#mainpage}
==========

[![Made withJupyter](https://img.shields.io/badge/Made%20with-Jupyter-orange?style=for-the-badge&logo=Jupyter)](https://jupyter.org/try)

[![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-380/)

![C++](https://img.shields.io/badge/C++-Solutions-blue.svg?style=flat&logo=c%2B%2B)

[Click here](./CONTRIBUTING.md) for installation and development notes.

# Implementation Table #
> - 🏆 indicates the language of choice. Something in C++ will be wrapped for calling from python;
> - `funcA` indicates a function to use

## SP-Digitiser Setup ##
> Complete

| `sp_digitizer.py` (Python) | `libia.cpp` (C++)                       |
|:---------------------------|-----------------------------------------|
| 🏆 extensive setup         | essential setup only for tests          |
|                            | 🏆 `fetch_digitiser_data` to fetch data |

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

| `power_kernel.py` (Python)                                        | `power_kernel.hpp` (C++)                                                       |
|:------------------------------------------------------------------|--------------------------------------------------------------------------------|
| "Out-of-the-box" `kernel` with little customisability             | 🏆 `kernel` that can be used in parallel processes                             |
| 🏆 use `check_kernel_parameters` to check the kernel built in C++ | feed `check_power_kernel_parameters` into the python `check_kernel_parameters` |

## Power pipeline ##
> Complete

| `sp_digitiser_power.py` (Python)                 | `power_pipeline.hpp` (C++)                            |
|:-------------------------------------------------|-------------------------------------------------------|
| 🏆 Compile + Run + Load latest log file and plot | 🏆 2 threads (read, process) dumping to rotating logs |

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
| GREEN  | Something completed |
| YELLOW | Irrelevant/debug    |
| RED    | Error               |
