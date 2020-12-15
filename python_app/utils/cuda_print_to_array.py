"""
Set of functions that places a value into an array allocated on the device

88  88  88
88  a   88
88  88  88
"""
from numba import cuda
from numba.cuda.cudadrv.devicearray import DeviceNDArray


@cuda.jit(device=True)
def dump_to_cell(
    x: int,
    y: int,
    value,
    array_to_dump_to: DeviceNDArray,
):
    for xidx in range(x - 1, x + 2):
        if xidx >= 0 and xidx < len(array_to_dump_to):
            for yidx in range(y - 1, y + 2):
                if yidx >= 0 and yidx < len(array_to_dump_to):
                    array_to_dump_to[xidx][yidx] = 88
    array_to_dump_to[x][y] = value


@cuda.jit(device=True)
def dump_thread_information(array_to_dump_to: DeviceNDArray):
    dump_to_cell(1, 3, cuda.threadIdx.x, array_to_dump_to)
    dump_to_cell(3, 3, cuda.threadIdx.y, array_to_dump_to)
    dump_to_cell(5, 3, cuda.threadIdx.z, array_to_dump_to)
    dump_to_cell(1, 1, cuda.blockDim.x, array_to_dump_to)
    dump_to_cell(3, 1, cuda.blockDim.y, array_to_dump_to)
    dump_to_cell(5, 1, cuda.blockDim.z, array_to_dump_to)
