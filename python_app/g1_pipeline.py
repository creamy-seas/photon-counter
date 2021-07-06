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


class G1Pipeline:
    HEADING = (
        TerminalColour.CSKYBLUEBG
        + TerminalColour.UNDERLINE
        + "G1-PIPELINE"
        + TerminalColour.ENDC
        + ":"
    )
    LOG_TEMPLATE = f"{HEADING:<31}{{info}}"

    # Constant - almost no reason to change them since they are optimised for benchmarking
    G1_DIGITISER_POINTS = 262144
    NS_PER_POINT = 2.5

    LOG_LOCATION = "./libia.log"
    LIBRARY_LOCATION = "./csrc/bin/libia.so"
    DUMP_FOLDER = "./dump"
    STORE_FOLDER = "./store"

    @classmethod
    def log(cls, message: str):
        print(cls.LOG_TEMPLATE.format(info=str(message)))

    def __init__(self, ipython: bool, recompile: bool=True, **kwargs):
        self.ipython = ipython

        os.makedirs(self.DUMP_FOLDER, exist_ok=True)
        self.NO_RUNS = kwargs["NO_RUNS"]
        self.TAU_POINTS = kwargs["TAU_POINTS"]

        self.log(
            "Building kernel with:\n"
            + f"TAU_POINTS={self.TAU_POINTS}"
            + "\n"
            + f"RUNS={self.NO_RUNS}"
            + "\n"
        )

        self.libia = self.build_library(recompile)
        (self.fig, self.ax, self.plot_dict) = self.prepare_plot(self.TAU_POINTS)
        
    @classmethod
    def build_library(cls, recompile: bool) -> ctypes.CDLL:

        if recompile:
            library_manager.build_library(
                {
                    "R_POINTS_PER_GPU_CHUNK": str(8),
                    "SP_POINTS": str(16),
                    "R_POINTS": str(16),
                }
            )
        libia = ctypes.cdll.LoadLibrary(cls.LIBRARY_LOCATION)
        # Check parameters that kernel was compiled with - some checks are easier to do in python.
        libia.check_g1_kernel_parameters()
        
        return libia
    
    @classmethod
    def prepare_plot_trigger(cls):
        """
        Process will monitor the file folder and update the state of `PlotTrigger`.
        It's state will be polled in order to determine if plot should be updated
        """
        observer = Observer()
        plot_trigger = PlotTrigger()
        observer.schedule(plot_trigger, cls.DUMP_FOLDER)
        observer.start()

        return (observer, plot_trigger)
    
    @classmethod
    def prepare_plot(cls, TAU_POINTS: int) -> (plt.Figure, plt.Axes, dict):
        time_axis = np.linspace(0, (TAU_POINTS - 1) * cls.NS_PER_POINT, TAU_POINTS)

        plot_dict = {
            "CHAG1": {"idx": 0, "ax": 0, "color": "red", "label": "CHA $g^{(1)}(\\tau)$"},
            "CHBG1": {"idx": 1, "ax": 1, "color": "blue", "label": "CHB $g^{(1)}(\\tau)$"},
            "SQG1": {"idx": 2, "ax": 2, "color": "black", "label": "SQ $g^{(1)}(\\tau)$"}
        }

        # Prepare plot
        fig, ax = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
        for ch in ["CHAG1", "CHBG1", "SQG1"]:
            i = plot_dict[ch]["ax"]

            # Plot line that will be constantly updateed
            (plot_dict[ch]["plot"],) = ax[i].plot(
                time_axis, [0] * TAU_POINTS, color=plot_dict[ch]["color"]
            )
            # Labels
            ax[i].grid(color="#bfbfbf", linestyle="-", linewidth=1)
            ax[i].set_ylabel(plot_dict[ch]["label"], color=plot_dict[ch]["color"])
        ax[-1].set_xlabel("$\\tau$ (ns)", fontsize=14)

        return (fig, ax, plot_dict)


    def update_plot(self, data: np.array, run: int, NO_RUNS):
        """
        Using the supplied plotting handles and parameters, update the plot
        """

        for ch in ["CHAG1", "CHBG1", "SQG1"]:
            self.plot_dict[ch]["plot"].set_ydata(
                data[self.plot_dict[ch]["idx"][:TAU_POINTS]]
            )
            self.ax[self.plot_dict[ch]["ax"]].relim()
            self.ax[self.plot_dict[ch]["ax"]].autoscale_view()

        self.fig.suptitle(f"Run {run}/{NO_RUNS}")
        if self.ipython:
            self.fig.canvas.draw()
        else:
            plt.pause(0.1)
            plt.draw()  # non-blocking drawing

    def execute_run(
        self,
        digitiser_parameters: dict,
        run_name: str,
        NO_RUNS: int=None,
    ):
        if os.path.exists(self.LOG_LOCATION):
            os.remove(self.LOG_LOCATION)

        if (NO_RUNS):
            self.log(f"Overriding number of runs {self.NO_RUNS} -> {NO_RUNS}")
        else:
            NO_RUNS = self.NO_RUNS

        # Prepare digitiser
        spd = SpDigitiser(
            {
                **digitiser_parameters,
                **{
                    "r_points": 1,
                    "sp_points": self.G1_DIGITISER_POINTS,
                },
            }
        )

        (observer, plot_trigger) = self.prepare_plot_trigger()

        # Launch measurements
        cpp_thread = threading.Thread(
            target=self.libia.run_g1_measurements,
            name="G1 Kernel Runner",
            args=(
                spd.adq_cu_ptr,
                NO_RUNS,
                ctypes.create_string_buffer(
                    f"{self.DUMP_FOLDER}/{run_name}".encode("utf-8"), size=40
                ),
            ),
        )
        cpp_thread.start()
        self.log("Measurements started")

        progress_bar = ProgBar(NO_RUNS, bar_char="█")
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
                    self.update_plot(data, plotted_run, NO_RUNS)
                    progress_bar.update()

            if not cpp_thread.is_alive():
                observer.join(0)
                # Report on any log errors
                library_manager.try_to_read_log_file()
                break

        self.update_plot(np.transpose(np.loadtxt(plot_trigger.filename)), NO_RUNS, NO_RUNS)
        result_file = f"{self.STORE_FOLDER}/{run_name}.csv"
        shutil.copyfile(filename, result_file)
        self.log(f"Measurements done -> data dumped to {result_file}")