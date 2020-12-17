"""
Class is responsible for setting up the digitiser and interacting with it
"""
import ctypes
from ctypes import cdll
from typing import Dict

from python_app.utils.terminal_colour import TerminalColour

try:
    ADQAPI = cdll.LoadLibrary("libadq.so")
    ia_ADQAPI = cdll.LoadLibrary("../build/ia_ADQAPI.so")

    ADQAPI.CreateADQControlUnit.restype = ctypes.c_void_p
    ADQAPI.ADQ214_GetRevision.restype = ctypes.c_void_p
    ADQAPI.ADQControlUnit_FindDevices.argtypes = [ctypes.c_void_p]
except Exception as err:
    print("Failed to load the ADQ library - see below for details:")
    print(err)


class SpDigitiser:
    HEADING = (
        TerminalColour.CROMAN
        + TerminalColour.UNDERLINE
        + "SP-DIGITISER"
        + TerminalColour.ENDC
        + ":"
    )
    LOG_TEMPLATE = f"{HEADING:<65}{{info}}"

    TRIGGER_EXTERNAL = 1
    TRIGGER_INTERNAL = 0

    @classmethod
    def log(cls, message: str):
        print(cls.LOG_TEMPLATE.format(info=str(message)))

    def __init__(self, sp_digitiser_parameters: Dict):
        """
        record                  is taken at every trigger
        samples_per_record      number of samples taken for every record
        """

        # 1. Create control unit and attach devices to it
        self.adq_cu_ptr = ctypes.c_void_p(ADQAPI.CreateADQControlUnit())
        no_of_devices = int(ADQAPI.ADQControlUnit_FindDevices(self.adq_cu_ptr))
        assert no_of_devices > 0, "Failed to find the SP-DIGITISER"

    def __del__(self):
        """Safe deallocation of pointer to disconnect the device"""
        ADQAPI.DeleteADQControlUnit(self.adq_cu_ptr)
        self.log("🕱 Destructor activated")

    def blink(self):
        self.log("Blinking device!")
        ADQAPI.ADQ214_Blink(self.adq_cu_ptr, 1)
        self.log("Blinking finished")
