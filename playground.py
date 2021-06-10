import matplotlib.pyplot as plt
from python_app.sp_digitiser import SpDigitiser
from python_app.power_pipeline import PowerPipeline
from python_app.utils import file_ops

RUN_NAME = "repetition-02MHz_width-3000ns"

TIME_IN_NS = 5000
AVERAGES = 1_000_00

R_POINTS = 2048
SP_POINTS = 2000
R_POINTS_PER_GPU_CHUNK = 64
NO_RUNS = 10

chA_background=None; chB_background=None
# (chA_background, chB_background) = file_ops.load_chA_chB_arrays("./dump/ttt.csv")

# Setup
pp = PowerPipeline(
    ipython=False,
    recompile=False,
    #time_in_ns=TIME_IN_NS, averages=AVERAGES
    R_POINTS=R_POINTS, SP_POINTS=SP_POINTS, R_POINTS_PER_GPU_CHUNK=R_POINTS_PER_GPU_CHUNK, NO_RUNS=NO_RUNS
)
plt.draw()  # Non-blocking drawing
plt.pause(.001)

# Execution
pp.execute_run(
    #NO_RUNS=10,
    NO_RUNS=None,
    digitiser_parameters={
        "delay": 0,
        "trigger_type": SpDigitiser.TRIGGER_EXTERNAL,
        #     "trigger_type": SpDigitiser.TRIGGER_SOFTWARE,
        "channelA_gain": 1,
        "channelB_gain": 1,
        "channelA_offset": 53,
        "channelB_offset": 38,
#         "clock_source": SpDigitiser.INTERNAL_CLOCK_SOURCE_INTERNAL_10MHZ_REFFERENCE,
        "clock_source": SpDigitiser.INTERNAL_CLOCK_SOURCE_EXTERNAL_10MHZ_REFFERENCE
    },
    run_name=RUN_NAME,
    chA_background=chA_background, chB_background=chB_background
)

plt.savefig(f"./{pp.STORE_FOLDER}/{RUN_NAME}.pdf")
plt.show()
