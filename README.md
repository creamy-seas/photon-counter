Overview {#mainpage}
==========

[![Made withJupyter](https://img.shields.io/badge/Made%20with-Jupyter-orange?style=for-the-badge&logo=Jupyter)](https://jupyter.org/try) [![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-380/) ![C++](https://img.shields.io/badge/C++-Solutions-blue.svg?style=flat&logo=c%2B%2B)

[Click here](./CONTRIBUTING.md) for installation and development notes.

# Implementation Table #
> - 🏆 indicates the language of choice. Something in C++ will be wrapped for calling from python;
> - `funcA` indicates a function to use

|                           | (Python)                                         | (C++)                                                 |
|---------------------------|:-------------------------------------------------|-------------------------------------------------------|
| **Digitiser Tools**       | `sp_digitiser.py`                                | `sp_digitiser.hpp`                                    |
| Setup of digitiser        | 🏆 extensive setup using dictionaries            | essential setup only for tests                        |
| Fetch data from digitser  |                                                  | 🏆 `fetch_digitiser_data` to fetch data               |
|                           |                                                  |                                                       |
| **Generic GPU Utilities** | `gpu_utils.py`                                   | `utils_gpu.hpp`                                       |
|                           | 🏆 `fetch_gpu_parameters`                        |                                                       |
|                           | 🏆  `check_gpu_allocation`                       |                                                       |
|                           | 🏆  `allocate_threads`                           |                                                       |
|                           |                                                  |                                                       |
| **Power Kernel**          | `power_kernel.py`                                | `power_kernel.hpp`                                    |
|                           | Kernel with little customisability               | 🏆 Kernel using constant memory, streams etc          |
| **Power Pipeline**        | `power_pipeline.py`                              | `power_pipeline.hpp`                                  |
|                           | 🏆 Compile + Run + Load latest log file and plot | 🏆 2 threads (read, process) dumping to rotating logs |
|                           |                                                  |                                                       |
| **G1 Kernel**             |                                                  | `g1_kernel.hpp`                                       |
|                           |                                                  | 🏆 Kernel to process single record of data            |
|                           | `g1_pipeline.py`                                   | `g1_pipeline.hpp`                                     |
|                           |                                                  | 🏆 2 threads (read, process) dumping to rotating logs |
| **G2 Kernel**             |                                                  | `g2_kernel.hpp`                                       |
|                           |                                                  | TODO                                                  |


# Colours #

| Colour | Meaning             |
|--------|---------------------|
| BLUE   | Information         |
| GREEN  | Something completed |
| YELLOW | Irrelevant/debug    |
| RED    | Error               |
