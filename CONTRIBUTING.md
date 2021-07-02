# Project setup #

- `gcc 4.8.5`
- `doxygen 1.8.5`

1. Create `.dir-locals.el` listing location of standard libraries and header files.

```elisp
((nil
  (company-clang-arguments . ("-I/usr/include"
                              "-I/home/antonov/photon_counting/csrc/include"
                              "-I/usr/local/cuda-11.0/include"
                              "-I/home/antonov/photon_counting/Celero/include"
                              "-std=c++11"))
  (flycheck-clang-include-path . ("/home/antonov/photon_counting/csrc/include"
                                  "/usr/local/cuda-11.0/include"
                                  "/home/antonov/photon_counting/Celero/include"
                                  ))
  (flycheck-clang-definitions . ("LINUX" "TESTENV" "R_POINTS=254200" "SP_POINTS=100" "THREADS_PER_BLOCK=1024")) ;; -D flags
  (c-basic-offset . 4)
  )
 (c++-mode
  (flycheck-clang-language-standard . "c++11"))
 (cuda-mode
  (c-basic-offset . 4)))
```

2. Run `ggtags-create-tags` in an c/cpp project. or `gtags`

3. Install `cppunit, gcov, lcov, latex, dvipsk` libraries

```shell
sudo apt-get install libcppunit-dev lcov texlive
```

4. Install `Celero` for benchmarking
```shell
git submodule update --init --recursive
cd Celero
mkdir build && cd build && cmake ..
make
```

5. Install [`fftw3`](http://fftw.org/) for efficient computation of correlation functions.
```shell
wget http://fftw.org/fftw-3.3.9.tar.gz
tar -xvf fftw-3.3.9.tar.gz
# Read the README
./configure --enable-shared --enable-threads
make
sudo make install
```

# Testing

For CPP
```shell
cd csrc && make test
```

For Python
```
cd python_app && nosetests
```

# Benchmarking #
`celero` will run N samples, where each sample will run M operations.

```shell
cd csrc && make bench
```


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

libia = cdll.LoadLibrary("./bin/libia.so")
libia.example_gpu_func.restype = ctypes.c_float

print(libia.example_gpu_func(10,2))
```

# Cuda notes #
- Use cuda-11.0
- `nvcc` will automatically lookup relevant headers and libraries, so it can be used for compilation. It can even pass the non-gpu code to the standard compuler. But it will not be able to inject `-fprofile-arcs` and `-ftest-coverage` so it is better to use it for only building object files and not the total compilation.
- Quadro P6000 GPU has Pascal Architecture with compute capability `6.1` (`sm_61`). See [this page](https://en.wikipedia.org/wiki/CUDA) or the `cudaGetDeviceProperties` function.

# Digitiser Notes #
- ADQ214 Digitiser
- Can run in **MultiRecord** mode only (no reading and streaming capabilities) see *14-1171-B Acquisition modes app note.pdf*
- We do not want to waveform average, as we would not be able to extract chA² or chB²

From *10-0569-ADQ214-datasheet.pdf*
- Trigger accurancy is 625ps (1/4 of 2.5ns sampling rate)
- 790Mb/s transfer rate, meaning
- 4.4Vpp input voltage on chA and chB
- 3V trigger input

Attenuation of `χdB` will lower voltage signal by `√(10^(χ/10))`.

| Keyisght Signal | Output | Attenuator (dB) | Output |
|:---------------:|:------:|:---------------:|:------:|
| CH1             | 4Vpp   | 6               | 2Vpp   |
| CH2             | 4Vpp   | 1               | 3.5Vpp |
| Trigger         | 5V     | 10              | 1.6V   |


# Tools #
- Check the objects in a library file
```shell
nm ./bin/ia_1488.so
```

- Check dependenices of library
```shell
ldd ./bin/ia_1488.so
```

- Warning: https://www.hpc.dtu.dk/?page_id=1180

- Libraries are normally in `/usr/lib` and the header files in `/usr/include`.

# Links #
<details>
<summary>Click this to collapse/fold.</summary>

- https://stackoverflow.com/questions/242894/cuda-driver-api-vs-cuda-runtime
- https://stackoverflow.com/questions/17278932/cuda-shared-library-linking-undefined-reference-to-cudaregisterlinkedbinary

</details>
