"""
Launching of power measurements:
1. Compling cpp library that evaluates measurements on GPU
2. Setup of digitiser
3. Running of measurements
4. Real time update of graph
"""
from typing import Dict

from python_app.sp_digitiser import SpDigitiser

class SpDigitiserPower(SpDigitiser):
    def __init__(
        self, sp_digitiser_parameters: Dict, power_measurement_parameters: Dict
    ):
        super().__init__()

        self.power_measurement_parameters = power_measurement_parameters
        assert (
            set(["chA_background", "chB_background"]) in power_measurement_parameters
        ), "Missing (chA_background/chB_background) parameters"

    def dump_to_file(self):
        pass

    def via_python(self):
        """
        Program passed to c layer
        """

        # =============================================================================
        #       declare arrays for storing average data and cumulative data
        # =============================================================================
        chA_average = np.zeros(self.samples_per_record, dtype=np.double)
        chB_average = np.zeros(self.samples_per_record, dtype=np.double)
        sq_average = np.zeros(self.samples_per_record, dtype=np.double)
        chA_cumulative = np.zeros(
            self.samples_per_record * self.number_of_records, dtype=np.int64
        )
        chB_cumulative = np.zeros(
            self.samples_per_record * self.number_of_records, dtype=np.int64
        )
        sq_cumulative = np.zeros(
            self.samples_per_record * self.number_of_records, dtype=np.uint64
        )

        # =============================================================================
        #         #setup multirecord mode
        # =============================================================================

        for i in range(0, self.power_measurement_parameters["number_of_runs"]):
            # =============================================================================
            #                 #1) collect data
            # =============================================================================
            ADQilya.ilya_incoherent_python(
                self.adq_cu_ptr,
                chA_average.ctypes.data,
                chB_average.ctypes.data,
                sq_average.ctypes.data,
                chA_back,
                chB_back,
                self.samples_per_record,
                self.number_of_records,
                chA_cumulative.ctypes.data,
                chB_cumulative.ctypes.data,
                sq_cumulative.ctypes.data,
                i + 1,
            )
            print(" - DLL processed")
            # =============================================================================
            #                 #2) plot data
            # =============================================================================
            if i > 0:
                self.currentA.remove()
                self.currentB.remove()
                self.currentSq.remove()

            # =============================================================================
            #                 #3) Periodically save data
            # =============================================================================
            if (i % 20) == 0:
                self.saveToFile(
                    chA_average,
                    chB_average,
                    sq_average,
                    "%s%s.txt" % (self.folderName, saveName),
                )

        # =============================================================================
        #             # Final save
        # =============================================================================
        self.saveToFile(
            chA_average,
            chB_average,
            sq_average,
            "%s%s.txt" % (self.folderName, save),
        )

    def via_c(self):
        """
        Run everything in C, dumping periodically to a file for python to display plot
        """
        pass
