import ctypes
from ctypes import cdll

libia = cdll.LoadLibrary("./bin/libia.so")
libia.example_gpu_func.restype = ctypes.c_float

print(libia.example_gpu_func(10,2))
