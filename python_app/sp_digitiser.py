"""
Class is responsible for setting up the digitiser and interacting with it
channelA_offset = -208
channelB_offset = -143
"""

import ctypes
from ctypes import cdll
from typing import Dict

from python_app.utils.terminal_colour import TerminalColour

###############################################################################
#                                 Library load                                #
###############################################################################
try:
    ADQAPI = cdll.LoadLibrary("libadq.so")
except Exception as err:
    raise RuntimeError(f"Failed to load the ADQ library - see below for details: {err}")
try:
    ia_ADQAPI = cdll.LoadLibrary("./bin/ia_1488.so")
except Exception as err:
    raise RuntimeError(
        f"Failed to load the custom photon-counting library - make sure that it has been built!: {err}"
    )

ADQAPI.CreateADQControlUnit.restype = ctypes.c_void_p
ADQAPI.ADQ214_GetRevision.restype = ctypes.c_void_p
ADQAPI.ADQControlUnit_FindDevices.argtypes = [ctypes.c_void_p]

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
    TRIGGER_INTERNAL = 0

    @classmethod
    def log(cls, message: str):
        print(cls.LOG_TEMPLATE.format(info=str(message)))

    def __init__(self, sp_digitiser_parameters: Dict):
        """
        record                  is taken at every trigger
        samples_per_record      number of samples taken for every record
        """
        self.sp_digitiser_parameters = sp_digitiser_parameters

        # 1. Create control unit and attach devices to it
        self.adq_cu_ptr = ctypes.c_void_p(ADQAPI.CreateADQControlUnit())
        no_of_devices = int(ADQAPI.ADQControlUnit_FindDevices(self.adq_cu_ptr))
        assert no_of_devices > 0, "Failed to find the SP-DIGITISER"

        # 2. Set parameters ###################################################
        try:
            self.parameter_preprocessing()
            self.parameter_setup()
        except KeyError as err:
            raise RuntimeError(f"Missing a parameter: {err}")

    def __del__(self):
        """Safe deallocation of pointer to disconnect the device"""
        ADQAPI.DeleteADQControlUnit(self.adq_cu_ptr)
        self.log("🕱 Destructor activated")

    def get_max_samples_per_record(self, number_of_records: int) -> int:
        return ia_ADQAPI.GetMaxNofSamplesFromNofRecords(
            self.adq_cu_ptr, number_of_records
        )

    def get_max_number_of_records(self, samples_per_record: int) -> int:
        return ia_ADQAPI.GetMaxNofRecordsFromNofSamples(
            self.adq_cu_ptr, samples_per_record
        )

    def parameter_preprocessing(self):
        samples_per_record = self.sp_digitiser_parameters["samples_per_record"]
        number_of_records = self.sp_digitiser_parameters["number_of_records"]

        max_samples = self.get_max_samples_per_record(number_of_records)
        max_records = self.get_max_number_of_records(samples_per_record)

        # Verify samples ######################################################
        self.log(
            f"""
                Max Samples for (number_of_records={number_of_records}): {max_samples}
                Max Records for (samples_per_record={samples_per_record}): {max_records}"""
        )

        if samples_per_record > max_samples or number_of_records > max_records:
            raise RuntimeError(
                "Invalid parameters for number_of_records/samples_per_record to collect:"
            )

        # Derive trigger frequency ############################################
        for (
            _spr,
            _trigger_frequency,
        ) in self.SAMPLES_PER_RECORD_TO_TRIGGER_FREQUENCY_KHZ:
            if samples_per_record > _spr:
                trigger_frequency = _trigger_frequency

        self.log(f"Trigger frequency: {trigger_frequency}kHz")
        self.sp_digitiser_parameters["trigger_frequency"] = trigger_frequency

    def parameter_setup(self):
        # a - delay after trigger #############################################
        ADQAPI.ADQ214_SetTriggerHoldOffSamples(
            self.adq_cu_ptr, 1, self.sp_digitiser_parameters["delay"]
        )

        # b - internal clock source with external 10MHz refference ############
        INTERNAL_CLOCK_SOURCE_EXTERNAL_10MHZ_REFFERECNCE = 1
        ADQAPI.ADQ214_SetClockSource(
            self.adq_cu_ptr, 1, INTERNAL_CLOCK_SOURCE_EXTERNAL_10MHZ_REFFERECNCE
        )

        # c - range of the external refference ################################
        LOW_FREQUENCY_MODE = 0
        HIGH_FREQUENCY_MODE = 1
        ADQAPI.ADQ214_SetClockFrequencyMode(self.adq_cu_ptr, 1, HIGH_FREQUENCY_MODE)

        # d - Synthesise 400MHz sampling signal from the f_clock=10MHz ########
        # (phase locked loop samples at f_clock*80/divider_value, so in this case its 400MHz. That is the sampling frequency)
        ADQAPI.ADQ214_SetPllFreqDivider(self.adq_cu_ptr, 1, 2)

        # e - Set the trigger type ############################################
        ADQAPI.ADQ214_SetTriggerMode(
            self.adq_cu_ptr, 1, self.sp_digitiser_parameters["trigger_type"]
        )

        # f - Set the data format to 14 bit unpacked, to map 1to1 the collected data memory inefficiently, but quickly #
        ADQAPI.ADQ214_SetDataFormat(self.adq_cu_ptr, 1, 1)

        # g - Offset found by taking measurements of the 2 channels #
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
            self.sp_digitiser_parameters["number_of_records"],
            self.sp_digitiser_parameters["samples_per_record"],
        )

    def blink(self):
        self.log("Blinking device!")
        ADQAPI.ADQ214_Blink(self.adq_cu_ptr, 1)
        self.log("Blinking finished")
