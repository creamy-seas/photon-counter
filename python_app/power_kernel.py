from typing import List, Callable, Tuple, Optional, Dict

import numpy as np
from numba import cuda
from numba.cuda.cudadrv.devicearray import DeviceNDArray

from python_app.utils import gpu_info


class PowerKernel:
    def __init__(self, parameter_dict: Dict):

        self.verify_gpu_allocation(parameter_dict)

        self.kernel = self.kernel_wrapper(
            parameter_dict["NP_POINTS"],
            parameter_dict["R_POINTS"],
            parameter_dict["PROCESSING_ARRAY_TYPE"],
        )

    @staticmethod
    def verify_gpu_allocation(parameter_dict: Dict):
        """Checks that allowed threads, blocks, memory"""

        # Unpack parameters ###################################################
        BLOCKS = parameter_dict["BLOCKS"]
        THREADS_PER_BLOCK = parameter_dict["THREADS_PER_BLOCK"]
        NP_POINTS = parameter_dict["NP_POINTS"]
        R_POINTS = parameter_dict["R_POINTS"]
        PROCESSING_ARRAY_TYPE = parameter_dict["PROCESSING_ARRAY_TYPE"]
        INPUT_ARRAY_TYPE = parameter_dict["INPUT_ARRAY_TYPE"]
        OUTPUT_ARRAY_TYPE = parameter_dict["OUTPUT_ARRAY_TYPE"]

        # Evaluate size of arrays to store ####################################
        in_arrays_in_bytes = 2 * (
            NP_POINTS * R_POINTS * np.dtype(INPUT_ARRAY_TYPE).itemsize
        )
        out_array_in_bytes = NP_POINTS * np.dtype(OUTPUT_ARRAY_TYPE).itemsize
        storage_array_in_bytes = NP_POINTS * np.dtype(PROCESSING_ARRAY_TYPE).itemsize

        # Pass it to the global verifier ######################################
        gpu_info.verify_gpu_allocation(
            **{
                "grid_dim_x": BLOCKS,
                "block_dim_x": THREADS_PER_BLOCK,
                "global_memory": in_arrays_in_bytes
                + out_array_in_bytes
                + storage_array_in_bytes,
                "shared_memory_per_block": storage_array_in_bytes,
            }
        )

    def kernel_wrapper(
        self, NP_POINTS: int, R_POINTS: int, PROCESSING_ARRAY_TYPE: type
    ):
        @cuda.jit(device=True)
        def reduction_sum(processing_array):
            """Reduce the array by summing up the total into the first cell"""

            i = min(cuda.blockDim.x // 2, R_POINTS // 2)

            while i != 0:

                r_coordinate = cuda.threadIdx.x
                while r_coordinate < R_POINTS:

                    if r_coordinate < i:
                        processing_array[r_coordinate] = r_coordinate + i
                        # a = processing_array[r_coordinate + i]
                    r_coordinate += cuda.blockDim.x
                cuda.syncthreads()
                i //= 2

        @cuda.jit
        def kernel(
            a_array: List[int],
            b_array: List[int],
            array_out: DeviceNDArray,
        ):
            """
            a/b_array:        Sequentially measured voltages
            array_out:          allocate either with cuda.device_array or passing in a numpy array

            __ Logic ___
            a1 a2 a3 a4 ... b1 b2 b3 b4 ... c1 c2 c3 c4 ...

            will be mapped to a 2D array

            a1 a2 a3 -> main_axis (np_coordinate)
            b1 b2 b3 ...
            c1 c2 c3 ...
            d1 d2 d3 ...
            e1 e2 e3 ...
            f1 f2 f3 ...
            g1 g2 g3 ...

            |
            repetition-axis (r_coordinate)
            """

            processing_array = cuda.shared.array(
                shape=R_POINTS, dtype=PROCESSING_ARRAY_TYPE
            )

            np_coordinate = cuda.blockIdx.x
            while np_coordinate < NP_POINTS:

                r_coordinate = cuda.threadIdx.x
                while r_coordinate < R_POINTS:

                    coordinate = r_coordinate + np_coordinate * NP_POINTS
                    # array_out[np_coordinate] = coordinate

                    processing_array[r_coordinate] = (
                        a_array[coordinate] * a_array[coordinate]
                        + b_array[coordinate] * b_array[coordinate]
                    )

                    # Once thread has completed, shift the
                    # row index by the number of allocated
                    # threads and continue summation
                    r_coordinate += cuda.blockDim.x

                # Ensure that all threads have completed execution
                cuda.syncthreads()

                # Summation
                reduction_sum(processing_array)
                array_out[np_coordinate] = float(processing_array[2])  # / R_POINTS

                # Shift by number of allocated blocks along main-axis
                np_coordinate += cuda.gridDim.x

        return kernel
