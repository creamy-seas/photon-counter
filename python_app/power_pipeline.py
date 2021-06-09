"""
Launching of power measurements:
1. Compling cpp library that evaluates measurements on GPU
2. Setup of digitiser
3. Running of measurements
4. Real time update of graph
"""

import os
import re
import math
import time
import ctypes
import shutil
import threading

import numpy as np
from pyprind import ProgBar
import matplotlib.pyplot as plt
from watchdog.observers import Observer


from python_app.utils.terminal_colour import TerminalColour
from python_app.sp_digitiser import SpDigitiser
from python_app.utils.file_watcher import PlotTrigger
from python_app.utils import library_manager
from python_app.utils import gpu_utils


class PowerPipeline:
    HEADING = (
        TerminalColour.CSKYBLUEBG
        + TerminalColour.UNDERLINE
        + "POWER-PIPELINE"
        + TerminalColour.ENDC
        + ":"
    )
    LOG_TEMPLATE = f"{HEADING:<31}{{info}}"

    # Constant - almost no reason to change them since they are optimised for benchmarking
    R_POINTS_PER_GPU_CHUNK = 1024  # How each batch of data on the digitiser is further split up for processing on GPU.
    NO_STREAMS = 2
    NS_PER_POINT = 2.5

    INIT_ARGUMENTS_OPTION_1 = {"time_in_ns", "averages"}
    INIT_ARGUMENTS_OPTION_2 = {
        "NO_RUNS",
        "SP_POINTS",
        "R_POINTS",
        "R_POINTS_PER_GPU_CHUNK",
    }

    LOG_LOCATION = "./libia.log"
    LIBRARY_LOCATION = "./csrc/bin/libia.so"
    FILE_FOLDER = "./dump"

    @classmethod
    def log(cls, message: str):
        print(cls.LOG_TEMPLATE.format(info=str(message)))

    def __init__(self, **kwargs):
        os.makedirs(self.FILE_FOLDER, exist_ok=True)

        if self.INIT_ARGUMENTS_OPTION_1.issubset(set(kwargs)):
            (self.SP_POINTS, self.R_POINTS, self.NO_RUNS) = self.derive_parameters(
                kwargs["time_in_ns"], kwargs["averages"]
            )
        elif self.INIT_ARGUMENTS_OPTION_2.issubset(set(kwargs)):
            self.NO_RUNS = kwargs["NO_RUNS"]
            self.SP_POINTS = kwargs["SP_POINTS"]
            self.R_POINTS = kwargs["R_POINTS"]
            self.R_POINTS_PER_GPU_CHUNK = kwargs["R_POINTS_PER_GPU_CHUNK"]
        else:
            raise RuntimeError(
                f"Must supply either {self.INIT_ARGUMENTS_OPTION_1} or {self.INIT_ARGUMENTS_OPTION_2}"
            )

        self.log(
            "Building kernel with:\n"
            + f"R_POINTS={self.R_POINTS}"
            + "\t\t"
            + f"R_POINTS_PER_GPU_CHUNK={self.R_POINTS_PER_GPU_CHUNK}"
            + "\n"
            + f"SP_POINTS={self.SP_POINTS}"
            + "\n"
            + f"RUNS={self.NO_RUNS}"
            + "\n"
        )

        self.libia = self.build_library(
            self.SP_POINTS, self.R_POINTS, self.R_POINTS_PER_GPU_CHUNK
        )

        (self.fig, self.ax, self.plot_dict) = self.prepare_plot(
            self.SP_POINTS * self.NS_PER_POINT, self.SP_POINTS
        )

    def execute_run(
        self,
        digitiser_parameters: dict,
        run_name: str,
        chA_background: np.array = None,
        chB_background: np.array = None,
    ):
        if os.path.exists(self.LOG_LOCATION):
            os.remove(self.LOG_LOCATION)

        # Check that background arrays are of the correct type
        if chA_background is not None:
            assert (
                chA_background.dtype == np.short
            ), "chA_background must be of type np.short"
        else:
            chA_background = np.zeros(self.SP_POINTS, dtype=np.short)
        if chB_background is not None:
            assert (
                chB_background.dtype == np.short
            ), "chB_background must be of type np.short"
        else:
            chB_background = np.zeros(self.SP_POINTS, dtype=np.short)

        # Prepare digitiser
        spd = SpDigitiser(
            {
                **digitiser_parameters,
                **{
                    "r_points": self.R_POINTS,
                    "sp_points": self.SP_POINTS,
                },
            }
        )

        (observer, plot_trigger) = self.prepare_plot_trigger()

        # Launch measurements
        cpp_thread = threading.Thread(
            target=self.libia.run_power_measurements,
            name="Power Kernel Runner",
            args=(
                spd.adq_cu_ptr,
                chA_background.ctypes.data,
                chB_background.ctypes.data,
                self.NO_RUNS,
                ctypes.create_string_buffer(
                    f"{self.FILE_FOLDER}/{run_name}".encode("utf-8"), size=40
                ),
            ),
        )
        cpp_thread.start()
        self.log("Measurements started")

        progress_bar = ProgBar(self.NO_RUNS, bar_char="█")
        plotted_run = 0
        while True:
            time.sleep(1)

            if plot_trigger.update:

                # 1. Get hold of the lock
                with plot_trigger.lock:
                    plot_trigger.update = False
                    filename = plot_trigger.filename

                # 2. Read run number
                with open(plot_trigger.filename, "r") as fin:
                    run_number = re.search("\d+", fin.readline()).group()

                # 3. If this is a new run (sometimes a trigger occurs mid-file) update plots
                if run_number != plotted_run:
                    plotted_run = run_number
                    data = np.transpose(np.loadtxt(filename))
                    for ch in ["CHA", "CHB", "CHASQ", "CHBSQ", "SQ"]:
                        self.plot_dict[ch]["plot"].set_ydata(
                            data[self.plot_dict[ch]["idx"]]
                        )
                        self.ax[self.plot_dict[ch]["ax"]].relim()
                        self.ax[self.plot_dict[ch]["ax"]].autoscale_view()

                    self.fig.suptitle(f"Run {plotted_run}/{self.NO_RUNS}")
                    self.fig.canvas.draw()
                    progress_bar.update()

            if not cpp_thread.is_alive():
                observer.join(0)
                # Report on any log errors
                library_manager.try_to_read_log_file()
                break

        result_file = f"{self.FILE_FOLDER}/{run_name}.csv"
        shutil.copyfile(filename, result_file)
        self.log(f"Measurements done -> data dumped to {result_file}")

    @classmethod
    def prepare_plot_trigger(cls):
        """
        Process will monitor the file folder and update the state of `PlotTrigger`.
        It's state will be polled in order to determine if plot should be updated
        """
        observer = Observer()
        plot_trigger = PlotTrigger()
        observer.schedule(plot_trigger, cls.FILE_FOLDER)
        observer.start()

        return (observer, plot_trigger)

    @classmethod
    def derive_parameters(cls, time_in_ns: int, averages: int) -> (int, int, int):

        # Data from digitiser is requested as: sample_points (SP_POINTS) x repetitions (R_POINTS)
        SP_POINTS = math.ceil(time_in_ns / cls.NS_PER_POINT)

        # Number of requested repetitions is chosen so that it can be split into CHUNKS and STREAMS on the GPU
        R_POINTS = (
            math.floor(
                SpDigitiser(None).get_max_noRecords_from_noSamples(SP_POINTS)
                / (cls.NO_STREAMS * cls.R_POINTS_PER_GPU_CHUNK)
            )
            * cls.R_POINTS_PER_GPU_CHUNK
            * cls.NO_STREAMS
        )

        # Mutliple runs will be required to accumulate all data
        NO_RUNS = math.ceil(averages / R_POINTS)

        cls.log(f"Performing {NO_RUNS} runs (to acquire {averages} averages)")

        return (SP_POINTS, R_POINTS, NO_RUNS)

    @classmethod
    def build_library(
        cls, SP_POINTS: int, R_POINTS: int, R_POINTS_PER_GPU_CHUNK: int
    ) -> ctypes.CDLL:

        library_manager.build_library(
            {
                "R_POINTS_PER_GPU_CHUNK": str(R_POINTS_PER_GPU_CHUNK),
                "SP_POINTS": str(SP_POINTS),
                "R_POINTS": str(R_POINTS),
            }
        )

        libia = ctypes.cdll.LoadLibrary(cls.LIBRARY_LOCATION)

        # Check parameters that kernel was compiled with - some checks are easier to do in python.
        libia.check_power_kernel_parameters()
        gpu_utils.check_gpu_allocation(
            **{
                "grid_dim_x": libia.fetch_power_kernel_blocks(),
                "block_dim_x": libia.fetch_power_kernel_threads(),
            }
        )
        return libia

    @staticmethod
    def prepare_plot(time_in_ns: int, SP_POINTS: int) -> (plt.Figure, plt.Axes, dict):
        time_axis = np.linspace(0, time_in_ns, SP_POINTS)

        plot_dict = {
            "CHA": {"idx": 0, "ax": 0, "color": "red", "label": "CHA"},
            "CHASQ": {"idx": 2, "ax": 1, "color": "red", "label": "ChA$^2$"},
            "CHB": {"idx": 1, "ax": 2, "color": "blue", "label": "ChB"},
            "CHBSQ": {"idx": 3, "ax": 3, "color": "blue", "label": "ChB$^2$"},
            "SQ": {"idx": 4, "ax": 4, "color": "black", "label": "ChA$^2$ + ChB$^2$"},
        }

        # Prepare plot
        fig, ax = plt.subplots(5, 1, figsize=(8, 8), sharex=True)
        for ch in ["CHA", "CHB", "CHASQ", "CHBSQ", "SQ"]:
            i = plot_dict[ch]["ax"]

            # Plot line that will be constantly updateed
            (plot_dict[ch]["plot"],) = ax[i].plot(
                time_axis, [0] * SP_POINTS, color=plot_dict[ch]["color"]
            )
            # Labels
            ax[i].grid(color="#bfbfbf", linestyle="-", linewidth=1)
            ax[i].set_ylabel(plot_dict[ch]["label"], color=plot_dict[ch]["color"])
        ax[-1].set_xlabel("Time (ns)", fontsize=14)

        return (fig, ax, plot_dict)
