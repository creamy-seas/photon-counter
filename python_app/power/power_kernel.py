from typing import List, Callable, Tuple, Optional

import numpy as np
from numba import cuda
from numba.cuda.cudadrv.devicearray import DeviceNDArray

from python_app.utils import gpu_info


class PowerKernel:
    def __init__(
        self,
        np_points: int,
        r_points: int,
        in_dtype: type,
        out_dtype: type,
        processing_dtype: type,
    ):
        self.NP_POINTS = np_points
        self.R_POINTS = r_points

        self.in_dtype = in_dtype
        self.out_dtype = out_dtype
        self.processing_dtype = processing_dtype

        self.kernel = self.kernel_wrapper()

    def verify_gpu_allocation(self):
        """Checks that allowed threads, blocks, memory"""

        in_arrays_in_bytes = 2 * (
            self.NP_POINTS * self.R_POINTS * np.dtype(self.in_dtype).itemsize
        )
        out_array_in_bytes = self.NP_POINTS * np.dtype(self.out_dtype).itemsize
        storage_array_in_bytes = (
            self.NP_POINS * np.dtype(self.processing_dtype).itemsize
        )

        gpu_info.verify_gpu_allocation(
            {
                "grid_dim_x": self.NP_POINTS,
                "block_dim_x": self.R_POINTS,
                "global_memory": 1,
            }
        )

    def kernel_wrapper(self):
        # Define it for when the kernel is defined
        NP_POINTS = self.NP_POINTS
        R_POINTS = self.R_POINTS

        @cuda.jit(target="gpu")
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

            a1 a2 a3 -> main_axis (np_coordinate, short)
            b1 b2 b3 ...
            c1 c2 c3 ...
            d1 d2 d3 ...
            e1 e2 e3 ...
            f1 f2 f3 ...
            g1 g2 g3 ...

            |
            repetition-axis (r_coordinate, long)
            """

            thread_array = cuda.shared.array(shape=(R_POINTS), dtype=np.int16)

            np_coordinate = cuda.block.x
            while np_coordinates < NP_POINTS:
                r_coordinate = cuda.threadIdx.x
                while r_coordinate < R_POINTS:
                    coordinate = r_coordinate + np_coordinate * NP_POINTS

                    # Once thread has completed, shift the
                    # index by the number of allocated threads
                    r_coordinate += cuda.blockDim.x
                # Shift by number of allocated blocks along main-axis
                np_coordinate += cuda.gridDim.x

            # Traverse over the full grid
            while phi01_idx < NUMBER_OF_PHI_POINTS:
                while phi02_idx < NUMBER_OF_PHI_POINTS:
                    while phi03_idx < NUMBER_OF_PHI_POINTS:
                        array_out[L][R][phi01_idx][phi02_idx][
                            phi03_idx
                        ] = potential_function_cuda(
                            (
                                phixx_array[phi01_idx],
                                phixx_array[phi02_idx],
                                phixx_array[phi03_idx],
                            ),
                            lr_array[L_offset],
                            lr_array[R_offset],
                            alpha,
                        )

                        phi03_idx += cuda.blockDim.z
                    phi03_idx = cuda.threadIdx.z
                    phi02_idx += cuda.blockDim.y
                phi02_idx = cuda.threadIdx.y
                phi01_idx += cuda.blockDim.x
            cuda.syncthreads()

        return kernel
