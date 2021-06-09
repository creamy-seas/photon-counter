"""
Establishing connection and setting up of the SP digitiser ADQ214 for data acquisition.

**Default Offsets**
channelA_offset = -208
channelB_offset = -143
"""

import ctypes
from typing import Dict

from python_app.utils.terminal_colour import TerminalColour
import python_app.utils.library_manager as library_manager

###############################################################################
#                                 Library load                                #
###############################################################################
try:
    ADQAPI = ctypes.cdll.LoadLibrary("libadq.so")
    ADQAPI.CreateADQControlUnit.restype = ctypes.c_void_p
    ADQAPI.ADQ214_GetRevision.restype = ctypes.c_void_p
    ADQAPI.ADQControlUnit_FindDevices.argtypes = [ctypes.c_void_p]
except Exception as err:
    raise RuntimeError(
        f"Failed to load the ADQ library (for the digitiser) - please install it by following instructions in README.md"
        + "\n"
        + "{err}"
    )

try:
    ADQAPIia = ctypes.cdll.LoadLibrary("./csrc/bin/ADQAPIia.so")
except OSError as err:
    print("Rebuilding digitiser library")
    library_manager.build_library({}, "libadq")
    ADQAPIia = ctypes.cdll.LoadLibrary("./csrc/bin/ADQAPIia.so")

###############################################################################
#                               Class definition                              #
###############################################################################
class SpDigitiser:
    HEADING = (
        TerminalColour.CROMAN
        + TerminalColour.UNDERLINE
        + "SP-DIGITISER"
        + TerminalColour.ENDC
        + ":"
    )
    LOG_TEMPLATE = f"{HEADING:<31}{{info}}"
    SAMPLES_PER_RECORD_TO_TRIGGER_FREQUENCY_KHZ = {
        (0, 1600),
        (16, 1300),
        (64, 860),
        (256, 340),
        (1024, 92),
        (4096, 6.1),
    }

    TRIGGER_SOFTWARE = 1
    TRIGGER_EXTERNAL = 2

    INTERNAL_CLOCK_SOURCE_INTERNAL_10MHZ_REFFERENCE = 0
    INTERNAL_CLOCK_SOURCE_EXTERNAL_10MHZ_REFFERENCE = 1

    LOW_FREQUENCY_MODE = 0  # external clock range 35-240MHz
    HIGH_FREQUENCY_MODE = 1  # external clock range 240-550MHz

    PACKED_14_BIT_MODE = 0  # faster
    UNPACKED_14_BIT_MODE = 1

    @classmethod
    def log(cls, message: str):
        print(cls.LOG_TEMPLATE.format(info=str(message)))

    def __init__(self, sp_digitiser_parameters: Dict = None):
        """
        @param r_points number of repetition measuements (aka r_points)
        @param sp_points number of samples taken at every trigger (aka samples_per_record)
        """
        self.sp_digitiser_parameters = sp_digitiser_parameters

        # 1. Create control unit and attach devices to it
        self.adq_cu_ptr = ctypes.c_void_p(ADQAPI.CreateADQControlUnit())
        try:
            no_of_devices = int(ADQAPI.ADQControlUnit_FindDevices(self.adq_cu_ptr))
            assert (
                no_of_devices > 0
            ), "No devices found! Make sure all programs refferencing devices are closed and that the box is switched on. When rebooting, turn the pc on after the digitiser."

            # 2. Set parameters if supplied
            if self.sp_digitiser_parameters:
                self.check_parameters()
                self.parameter_setup()
        except KeyError as err:
            self.__del__()
            raise RuntimeError(f"Missing a parameter: {err}")
        except Exception as err:
            self.__del__()
            raise err

    def __del__(self):
        """Safe disconnection from the device"""
        ADQAPI.DeleteADQControlUnit(self.adq_cu_ptr)
        self.log("🕱 Disconnected from digitiser.")

    def get_max_noSamples_from_noRecords(self, r_points: int) -> int:
        return ADQAPIia.GetMaxNofSamplesFromNofRecords(self.adq_cu_ptr, r_points)

    def get_max_noRecords_from_noSamples(self, sp_points: int) -> int:
        return ADQAPIia.GetMaxNofRecordsFromNofSamples(self.adq_cu_ptr, sp_points)

    def check_parameters(self):
        sp_points = self.sp_digitiser_parameters["sp_points"]
        r_points = self.sp_digitiser_parameters["r_points"]

        max_samples = self.get_max_noSamples_from_noRecords(r_points)
        max_records = self.get_max_noRecords_from_noSamples(sp_points)

        # Verify samples
        #         self.log("\n" +
        #             f"Max Samples for (r_points={r_points}): {max_samples}" +
        #             "\n" +
        #             f"Max Records for (sp_points={sp_points}): {max_records}"
        #         )
        if sp_points > max_samples or r_points > max_records:
            raise RuntimeError(
                "Invalid parameters for r_points/sp_points to digitiser readout."
            )

        # Derive trigger frequency
        for (
            _spr,
            _trigger_frequency,
        ) in self.SAMPLES_PER_RECORD_TO_TRIGGER_FREQUENCY_KHZ:
            if sp_points > _spr:
                trigger_frequency = _trigger_frequency

        self.log(f"Max Trigger frequency: {trigger_frequency}kHz")
        self.sp_digitiser_parameters["trigger_frequency"] = trigger_frequency

    def parameter_setup(self):
        # a - delay after trigger
        assert ADQAPI.ADQ214_SetTriggerHoldOffSamples(
            self.adq_cu_ptr, 1, self.sp_digitiser_parameters["delay"]
        )

        # b - if exetrnal clock source, connect it to the front panel
        assert ADQAPI.ADQ214_SetClockSource(
            self.adq_cu_ptr, 1, self.sp_digitiser_parameters["clock_source"]
        )
        if self.sp_digitiser_parameters["clock_source"]:
            self.log(f"{TerminalColour.WARNING}External clock source used!")

        # c - range of the external clock refference
        assert ADQAPI.ADQ214_SetClockFrequencyMode(
            self.adq_cu_ptr, 1, self.sp_digitiser_parameters["frequency_mode"]
        )

        # d - Synthesise 400MHz sampling signal from the f_clock=10MHz
        # (phase locked loop samples at f_clock*80/divider_value, so in this case its 400MHz. That is the sampling frequency)
        assert ADQAPI.ADQ214_SetPllFreqDivider(self.adq_cu_ptr, 1, 2)

        # e - Set the trigger type
        assert ADQAPI.ADQ214_SetTriggerMode(
            self.adq_cu_ptr, 1, self.sp_digitiser_parameters["trigger_type"]
        )
        if self.sp_digitiser_parameters["trigger_type"]:
            self.log(f"{TerminalColour.WARNING}External trigger used!")

        # f - Set the data format to 14 bit unpacked, to map 1to1 the collected data memory inefficiently, but quickly
        assert ADQAPI.ADQ214_SetDataFormat(self.adq_cu_ptr, 1, self.PACKED_14_BIT_MODE)

        # g - Offset found by taking measurements of the 2 channels
        assert ADQAPI.ADQ214_SetGainAndOffset(
            self.adq_cu_ptr,
            1,
            1,
            int(self.sp_digitiser_parameters["channelA_gain"] * 1024),
            self.sp_digitiser_parameters["channelA_offset"],
        )
        assert ADQAPI.ADQ214_SetGainAndOffset(
            self.adq_cu_ptr,
            1,
            2,
            int(self.sp_digitiser_parameters["channelB_gain"] * 1024),
            self.sp_digitiser_parameters["channelB_offset"],
        )

    def blink(self):
        self.log("Blinking device!")
        assert ADQAPI.ADQ214_Blink(self.adq_cu_ptr, 1)
        self.log("Blinking finished")
