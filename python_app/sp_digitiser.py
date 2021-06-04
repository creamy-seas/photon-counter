"""
Establishing connection and setting up of the SP digitiser ADQ214 for data acquisition.

**Default Offsets**
channelA_offset = -208
channelB_offset = -143
"""

import ctypes
from typing import Dict

from python_app.utils.terminal_colour import TerminalColour

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
        + "\n" + "{err}")

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

    TRIGGER_EXTERNAL = 1
    TRIGGER_SOFTWARE = 0

    INTERNAL_CLOCK_SOURCE_INTERNAL_10MHZ_REFFERENCE = 0
    INTERNAL_CLOCK_SOURCE_EXTERNAL_10MHZ_REFFERENCE = 1

    LOW_FREQUENCY_MODE = 0 # external clock range 35-240MHz
    HIGH_FREQUENCY_MODE = 1 # external clock range 240-550MHz

    PACKED_14_BIT_MODE = 0 # faster
    UNPACKED_14_BIT_MODE = 1

    @classmethod
    def log(cls, message: str):
        print(cls.LOG_TEMPLATE.format(info=str(message)))

    def __init__(self, sp_digitiser_parameters: Dict, libia: ctypes.CDLL):
        """
        @param r_points number of repetition measuements (aka r_points)
        @param sp_points number of samples taken at every trigger (aka samples_per_record)
        """

        self.sp_digitiser_parameters = sp_digitiser_parameters
        self.libia = libia

        # 1. Create control unit and attach devices to it
        self.adq_cu_ptr = ctypes.c_void_p(ADQAPI.CreateADQControlUnit())
        no_of_devices = int(ADQAPI.ADQControlUnit_FindDevices(self.adq_cu_ptr))
        assert no_of_devices > 0, "Failed to find the SP-DIGITISER"

        # 2. Set parameters
        try:
            self.check_parameters()
            self.parameter_setup()
        except KeyError as err:
            raise RuntimeError(f"Missing a parameter: {err}")

    def __del__(self):
        """Safe deallocation of pointer to disconnect the device"""
        ADQAPI.DeleteADQControlUnit(self.adq_cu_ptr)
        self.log("🕱 Disconnected from digitiser.")

    def get_max_samples_per_record(self, r_points: int) -> int:
        return self.libia.GetMaxNofSamplesFromNofRecords(
            self.adq_cu_ptr, r_points
        )

    def get_max_number_of_records(self, sp_points: int) -> int:
        return self.libia.GetMaxNofRecordsFromNofSamples(
            self.adq_cu_ptr, sp_points
        )

    def check_parameters(self):
        sp_points = self.sp_digitiser_parameters["sp_points"]
        r_points = self.sp_digitiser_parameters["r_points"]

        max_samples = self.get_max_samples_per_record(r_points)
        max_records = self.get_max_number_of_records(sp_points)

        # Verify samples
        self.log("\n" +
            f"Max Samples for (r_points={r_points}): {max_samples}" +
            "\n" +
            f"Max Records for (sp_points={sp_points}): {max_records}"
        )

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

        self.log(f"Trigger frequency: {trigger_frequency}kHz")
        self.sp_digitiser_parameters["trigger_frequency"] = trigger_frequency

    def parameter_setup(self):
        # a - delay after trigger
        ADQAPI.ADQ214_SetTriggerHoldOffSamples(
            self.adq_cu_ptr, 1, self.sp_digitiser_parameters["delay"]
        )

        # b - if exetrnal clock source, connect it to the front panel
        ADQAPI.ADQ214_SetClockSource(
            self.adq_cu_ptr, 1, self.sp_digitiser_parameters["clock_source"]
        )

        # c - range of the external clock refference
        ADQAPI.ADQ214_SetClockFrequencyMode(self.adq_cu_ptr, 1, self.sp_digitiser_parameters["frequency_mode"])

        # d - Synthesise 400MHz sampling signal from the f_clock=10MHz
        # (phase locked loop samples at f_clock*80/divider_value, so in this case its 400MHz. That is the sampling frequency)
        ADQAPI.ADQ214_SetPllFreqDivider(self.adq_cu_ptr, 1, 2)

        # e - Set the trigger type
        ADQAPI.ADQ214_SetTriggerMode(
            self.adq_cu_ptr, 1, self.sp_digitiser_parameters["trigger_type"]
        )

        # f - Set the data format to 14 bit unpacked, to map 1to1 the collected data memory inefficiently, but quickly
        ADQAPI.ADQ214_SetDataFormat(self.adq_cu_ptr, 1, self.PACKED_14_BIT_MODE)

        # g - Offset found by taking measurements of the 2 channels
        ADQAPI.ADQ214_SetGainAndOffset(
            self.adq_cu_ptr,
            1,
            1,
            int(self.sp_digitiser_parameters["channelA_gain"] * 1024),
            int(
                -self.sp_digitiser_parameters["channelA_offset"]
                / self.sp_digitiser_parameters["channelA_gain"]
                / 4
            ),
        )
        ADQAPI.ADQ214_SetGainAndOffset(
            self.adq_cu_ptr,
            1,
            2,
            int(self.sp_digitiser_parameters["channelB_gain"] * 1024),
            int(
                -self.sp_digitiser_parameters["channelB_offset"]
                / self.sp_digitiser_parameters["channelB_gain"]
                / 4
            ),
        )

        # h - setup multirecord mode ##########################################
        ADQAPI.ADQ214_MultiRecordSetup(
            self.adq_cu_ptr,
            1,
            self.sp_digitiser_parameters["r_points"],
            self.sp_digitiser_parameters["sp_points"],
        )

    def blink(self):
        self.log("Blinking device!")
        ADQAPI.ADQ214_Blink(self.adq_cu_ptr, 1)
        self.log("Blinking finished")
