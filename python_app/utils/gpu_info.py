import json
from typing import List, Callable, Tuple, Optional

import numba as nb
from numba import cuda


def fetch_gpu_parameters(verbose=False):
    if cuda.is_available():
        device = cuda.get_current_device()
        parameters = {
            # Device Parameters
            "name": device.name.decode("utf-8"),
            "clock_rate": device.CLOCK_RATE,
            "compute_capability": device.compute_capability,
            "shared_memory_per_block": device.MAX_SHARED_MEMORY_PER_BLOCK,
            "threads_per_block": device.MAX_THREADS_PER_BLOCK,
            "block_dim_x": device.MAX_BLOCK_DIM_X,
            "block_dim_y": device.MAX_BLOCK_DIM_Y,
            "block_dim_z": device.MAX_BLOCK_DIM_Z,
            "grid_dim_x": device.MAX_GRID_DIM_X,
            "grid_dim_y": device.MAX_GRID_DIM_Y,
            "grid_dim_z": device.MAX_GRID_DIM_Z,
            "global_memory": 25637224448,
        }

        if verbose:
            print(f"🦑 Found device {parameters['name']}")
            print(json.dumps(parameters, indent=4))

            print(
                f"🦑 shared_memory_per_block int16: {device.MAX_SHARED_MEMORY_PER_BLOCK // nb.int16.bitwidth}"
            )
            print(
                f"🦑 shared_memory_per_block float32: {device.MAX_SHARED_MEMORY_PER_BLOCK // nb.float32.bitwidth}"
            )
            print(
                f"🦑 global_memory int16: {parameters['global_memory'] // nb.int16.bitwidth}"
            )
            print(
                f"🦑 global_memory float32: {parameters['global_memory'] // nb.float32.bitwidth}"
            )

        return parameters
    raise RuntimeError("Missing GPU")


def verify_gpu_allocation(**kwargs):
    """Compares supplied parameters to cuda ones to ensure that they fit"""

    gpu_parameters = fetch_gpu_parameters()

    total_threads_in_block = 1
    direct_comparisson = [
        "grid_dim_x",
        "grid_dim_y",
        "grid_dim_z",
        "block_dim_x",
        "block_dim_y",
        "block_dim_z",
        "global_memory",
        "shared_memory_per_block",
    ]
    for key in filter(lambda x: x in kwargs, kwargs):
        if kwargs[key] > gpu_parameters[key]:
            raise RuntimeError(
                f"Parameter ({key} = {kwargs[key]}) larger than allowed ({key} = {gpu_parameters[key]})"
            )
        if key in ["block_dim_x", "block_dim_y", "block_dim_z"]:
            total_threads_in_block *= kwargs[key]

    if total_threads_in_block > gpu_parameters["threads_per_block"]:
        raise RuntimeError(
            f"Too many threads allocated ({kwargs[key]} > gpu_parameters['threads_per_block'])"
        )


def allocate_max_threads(
    user_defined_number: Optional[int] = None, verbose=False
) -> Tuple[int, int, int]:
    gpu_info = fetch_gpu_parameters()
    if verbose:
        print(
            f"""Thread parameters:
    > Max threads per block: {gpu_info['threads_per_block']}
    > Max threads in x: {gpu_info['block_dim_x']}
    > Max threads in y: {gpu_info['block_dim_y']}
    > Max threads in z: {gpu_info['block_dim_z']}"""
        )
    max_threads_approximation = int(gpu_info["threads_per_block"] ** (1 / 3))
    if user_defined_number is not None:
        max_threads_approximation = user_defined_number

    max_thread_allocation = (
        min(max_threads_approximation, gpu_info["block_dim_x"]),
        min(max_threads_approximation, gpu_info["block_dim_y"]),
        min(max_threads_approximation, gpu_info["block_dim_z"]),
    )
    print(f"🐳 {'Allocating':<20} THREADS_PER_BLOCK = {max_thread_allocation}")

    return max_thread_allocation
