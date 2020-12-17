import json
from typing import List, Callable, Tuple, Optional, Dict

import numpy as np
from numba import cuda
from numba.cuda.cudadrv.devicearray import DeviceNDArray

from python_app.utils import gpu_info
from python_app.utils.terminal_colour import TerminalColour


class PowerKernel:
    HEADING = (
        TerminalColour.CBLUE
        + TerminalColour.BOLD
        + "POWER-KERNEL"
        + TerminalColour.ENDC
        + ":"
    )
    LOG_TEMPLATE = f"{HEADING:<65}{{info}}"

    @classmethod
    def log(cls, message: str):
        print(cls.LOG_TEMPLATE.format(info=str(message)))

    def __init__(self, power_kernel_parameters: Dict):

        self.verify_gpu_allocation(power_kernel_parameters)
        self.log(
            f"""Loaded following parameters:
{json.dumps(power_kernel_parameters, indent=4, default=lambda o: str(o))}"""
        )

        self.kernel = self.kernel_wrapper(
            power_kernel_parameters["NP_POINTS"],
            power_kernel_parameters["R_POINTS"],
            power_kernel_parameters["PROCESSING_ARRAY_TYPE"],
        )

    @staticmethod
    def verify_gpu_allocation(power_kernel_parameters: Dict):
        """Checks that allowed threads, blocks, memory"""

        # Unpack parameters ###################################################
        BLOCKS = power_kernel_parameters["BLOCKS"]
        THREADS_PER_BLOCK = power_kernel_parameters["THREADS_PER_BLOCK"]
        NP_POINTS = power_kernel_parameters["NP_POINTS"]
        R_POINTS = power_kernel_parameters["R_POINTS"]
        PROCESSING_ARRAY_TYPE = power_kernel_parameters["PROCESSING_ARRAY_TYPE"]
        INPUT_ARRAY_TYPE = power_kernel_parameters["INPUT_ARRAY_TYPE"]
        OUTPUT_ARRAY_TYPE = power_kernel_parameters["OUTPUT_ARRAY_TYPE"]

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

        assert (
            R_POINTS & (R_POINTS - 1) == 0
        ) and R_POINTS != 0, f"(R_POINTS={R_POINTS}) is not a power of 2! This would make the implementaton of summation inefficient, so change it"
        assert (
            R_POINTS != 1
        ), f"(R_POINTS={R_POINTS}) is an edge case - try setting it to at least 2"

    def kernel_wrapper(
        self, NP_POINTS: int, R_POINTS: int, PROCESSING_ARRAY_TYPE: type
    ):
        i_default = R_POINTS // 2

        @cuda.jit(device=True)
        def reduction_sum(cache_array):
            """Reduce the array by summing up the total into the first cell"""

            i = i_default

            while i != 0:
                r_coordinate = cuda.threadIdx.x
                while r_coordinate < R_POINTS:
                    if r_coordinate < i:
                        cache_array[r_coordinate] += cache_array[r_coordinate + i]
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

            cache_array = cuda.shared.array(
                shape=(R_POINTS), dtype=PROCESSING_ARRAY_TYPE
            )

            np_coordinate = cuda.blockIdx.x
            while np_coordinate < NP_POINTS:

                r_coordinate = cuda.threadIdx.x
                while r_coordinate < R_POINTS:
                    # coordinate = r_coordinate + np_coordinate * R_POINTS
                    coordinate = r_coordinate * NP_POINTS + np_coordinate

                    cache_array[r_coordinate] = (
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
                reduction_sum(cache_array)
                array_out[np_coordinate] = float(cache_array[0]) / R_POINTS

                # Shift by number of allocated blocks along main-axis
                np_coordinate += cuda.gridDim.x

        return kernel
