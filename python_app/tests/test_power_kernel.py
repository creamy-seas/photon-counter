import logging
import unittest
from unittest.mock import Mock
from unittest.mock import patch

import numpy as np
from numba import cuda

from python_app.power_kernel import PowerKernel


class TestPowerKernel(unittest.TestCase):
    def setUp(self):
        numba_logger = logging.getLogger("numba")
        numba_logger.setLevel(logging.WARNING)

    def tearDown(self):
        numba_logger = logging.getLogger("numba")
        numba_logger.setLevel(logging.DEBUG)

    # def test__small_size(self):

    #     NP_POINTS = 1
    #     R_POINTS = 5

    #     # init ################################################################
    #     parameter_dict = {
    #         "NP_POINTS": NP_POINTS,
    #         "BLOCKS": NP_POINTS,
    #         "R_POINTS": R_POINTS,
    #         "THREADS_PER_BLOCK": 1024,
    #         "PROCESSING_ARRAY_TYPE": np.int32,
    #         "INPUT_ARRAY_TYPE": np.int16,
    #         "OUTPUT_ARRAY_TYPE": np.float32,
    #     }
    #     sut = PowerKernel(parameter_dict)

    #     # data ################################################################
    #     total_points = parameter_dict["NP_POINTS"] * parameter_dict["R_POINTS"]
    #     a_array = np.array([1] * total_points)
    #     b_array = np.array([2] * total_points)
    #     DEVICE_out_array = cuda.device_array(
    #         shape=(parameter_dict["NP_POINTS"]),
    #         dtype=parameter_dict["OUTPUT_ARRAY_TYPE"],
    #     )

    #     # run #################################################################
    #     sut.kernel[(parameter_dict["BLOCKS"], parameter_dict["THREADS_PER_BLOCK"])](
    #         cuda.to_device(a_array), cuda.to_device(b_array), DEVICE_out_array
    #     )

    #     # test ################################################################
    #     expected = np.array([5.0] * parameter_dict["NP_POINTS"])
    #     self.assertTrue(np.all(expected - DEVICE_out_array.copy_to_host() == 0))

    # def test__large_size(self):

    #     NP_POINTS = 10000
    #     R_POINTS = 5000

    #     # init ################################################################
    #     parameter_dict = {
    #         "NP_POINTS": NP_POINTS,
    #         "BLOCKS": NP_POINTS,
    #         "R_POINTS": R_POINTS,
    #         "THREADS_PER_BLOCK": 1024,
    #         "PROCESSING_ARRAY_TYPE": np.int32,
    #         "INPUT_ARRAY_TYPE": np.int16,
    #         "OUTPUT_ARRAY_TYPE": np.float32,
    #     }
    #     sut = PowerKernel(parameter_dict)

    #     # data ################################################################
    #     total_points = parameter_dict["NP_POINTS"] * parameter_dict["R_POINTS"]
    #     a_array = np.array([1] * total_points)
    #     b_array = np.array([2] * total_points)
    #     DEVICE_out_array = cuda.device_array(
    #         shape=(parameter_dict["NP_POINTS"]),
    #         dtype=parameter_dict["OUTPUT_ARRAY_TYPE"],
    #     )

    #     # run #################################################################
    #     sut.kernel[(parameter_dict["BLOCKS"], parameter_dict["THREADS_PER_BLOCK"])](
    #         cuda.to_device(a_array), cuda.to_device(b_array), DEVICE_out_array
    #     )

    #     # test ################################################################
    #     expected = np.array([5.0] * parameter_dict["NP_POINTS"])
    #     self.assertTrue(np.all(expected - DEVICE_out_array.copy_to_host() == 0))

    # def test__specific_array_even_r(self):

    #     a_array = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    #     b_array = np.array([11, 12, 13, 14, 15, 16, 17, 18, 19, 20])

    #     NP_POINTS = 5
    #     R_POINTS = 2

    #     # init ################################################################
    #     parameter_dict = {
    #         "NP_POINTS": NP_POINTS,
    #         "BLOCKS": NP_POINTS,
    #         "R_POINTS": R_POINTS,
    #         "THREADS_PER_BLOCK": 1024,
    #         "PROCESSING_ARRAY_TYPE": np.int32,
    #         "INPUT_ARRAY_TYPE": np.int16,
    #         "OUTPUT_ARRAY_TYPE": np.float32,
    #     }
    #     sut = PowerKernel(parameter_dict)

    #     # data ################################################################
    #     DEVICE_out_array = cuda.device_array(
    #         shape=(parameter_dict["NP_POINTS"]),
    #         dtype=parameter_dict["OUTPUT_ARRAY_TYPE"],
    #     )

    #     # run #################################################################
    #     sut.kernel[(parameter_dict["BLOCKS"], parameter_dict["THREADS_PER_BLOCK"])](
    #         cuda.to_device(a_array), cuda.to_device(b_array), DEVICE_out_array
    #     )

    #     # test ################################################################
    #     expected = np.array([207, 243, 283, 327, 375])
    #     self.assertTrue(np.all(expected - DEVICE_out_array.copy_to_host() == 0))

    def test__specific_array_odd_r(self):

        a_array = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        b_array = np.array([11, 12, 13, 14, 15, 16, 17, 18, 19, 20])

        NP_POINTS = 2
        R_POINTS = 5

        # init ################################################################
        parameter_dict = {
            "NP_POINTS": NP_POINTS,
            "BLOCKS": NP_POINTS,
            "R_POINTS": R_POINTS,
            "THREADS_PER_BLOCK": 1024,
            "PROCESSING_ARRAY_TYPE": np.int32,
            "INPUT_ARRAY_TYPE": np.int16,
            "OUTPUT_ARRAY_TYPE": np.float32,
        }
        sut = PowerKernel(parameter_dict)

        # data ################################################################
        DEVICE_out_array = cuda.device_array(
            shape=(parameter_dict["NP_POINTS"]),
            dtype=parameter_dict["OUTPUT_ARRAY_TYPE"],
        )

        # run #################################################################
        sut.kernel[(parameter_dict["BLOCKS"], parameter_dict["THREADS_PER_BLOCK"])](
            cuda.to_device(a_array), cuda.to_device(b_array), DEVICE_out_array
        )

        # test ################################################################
        assert False, DEVICE_out_array.copy_to_host()
        expected = np.array([266, 308])
        self.assertTrue(np.all(expected - DEVICE_out_array.copy_to_host() == 0))

    # def test__specific_array_low_threads_laow_blocks(self):

    #     a_array = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    #     b_array = np.array([11, 12, 13, 14, 15, 16, 17, 18, 19, 20])

    #     NP_POINTS = 5
    #     R_POINTS = 2

    #     # init ################################################################
    #     parameter_dict = {
    #         "NP_POINTS": NP_POINTS,
    #         "BLOCKS": 2,
    #         "R_POINTS": R_POINTS,
    #         "THREADS_PER_BLOCK": 1,
    #         "PROCESSING_ARRAY_TYPE": np.int32,
    #         "INPUT_ARRAY_TYPE": np.int16,
    #         "OUTPUT_ARRAY_TYPE": np.float32,
    #     }
    #     sut = PowerKernel(parameter_dict)

    #     # data ################################################################
    #     DEVICE_out_array = cuda.device_array(
    #         shape=(parameter_dict["NP_POINTS"]),
    #         dtype=parameter_dict["OUTPUT_ARRAY_TYPE"],
    #     )

    #     # run #################################################################
    #     sut.kernel[(parameter_dict["BLOCKS"], parameter_dict["THREADS_PER_BLOCK"])](
    #         cuda.to_device(a_array), cuda.to_device(b_array), DEVICE_out_array
    #     )

    #     # test ################################################################
    #     expected = np.array([207, 243, 283, 327, 375])
    #     self.assertTrue(np.all(expected - DEVICE_out_array.copy_to_host() == 0))
