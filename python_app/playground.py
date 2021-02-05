import ctypes
from ctypes import cdll

ia_ADQAPI = cdll.LoadLibrary("./bin/ia_1488.so")
ia_ADQAPI.example_gpu_func.restype = ctypes.c_float

print(ia_ADQAPI.example_gpu_func(10,2))
