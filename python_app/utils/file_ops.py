"""
Common file operations
"""
import numpy as np

from python_app.utils.terminal_colour import TerminalColour

HEADING = (
    TerminalColour.CGREEN
    + TerminalColour.UNDERLINE
    + "FILE-OPERATIONS"
    + TerminalColour.ENDC
    + ":"
)
LOG_TEMPLATE = f"{HEADING:<31}{{}}"


def load_chA_chB_arrays(
    filename: str, numpy_data_type=np.short
) -> (np.array, np.array):
    """
    Arrays are read in from the target file, expecting:
    - chA in column 0
    - chB in column 1
    """
    CHA_COL = 0
    CHB_COL = 1

    raw_data = np.loadtxt(filename).transpose()
    SP_POINTS = raw_data.shape[1]

    print(
        LOG_TEMPLATE.format(
            f"Loaded chA and chB arrays of length SP_POINTS={SP_POINTS}."
        )
    )

    chA_array = np.zeros(SP_POINTS, dtype=numpy_data_type)
    chB_array = np.zeros(SP_POINTS, dtype=numpy_data_type)
    for sp in range(SP_POINTS):
        chA_array[sp] = raw_data[CHA_COL][sp]
        chB_array[sp] = raw_data[CHB_COL][sp]

    return (chA_array, chB_array)
