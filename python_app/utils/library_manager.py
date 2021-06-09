"""
Functions for building and running cpp libraries:
- ADQAPIia.so: Rewrapped functions for the digitiser.
- libia.so: Library with the measurement methods.
"""
import os
import subprocess

from python_app.utils.terminal_colour import TerminalColour

LOG_FILE = "./libia.log"


def HANDLE_CPP_ERROR(result: int):
    """All CPP methods return a status code. 0 means success.

    Otherwise, read the log file which contains error message on the last line.
    """

    if result != 0:
        with open(LOG_FILE, "r") as fin:
            for line in fin:
                pass
            print(
                f"{TerminalColour.FAIL}Error in CPP function:{TerminalColour.ENDC} {line}"
            )


def try_to_read_log_file():
    try:
        with open(LOG_FILE, "r") as fin:
            line = None
            for line in fin:
                pass
            if line:
                print(
                    f"{TerminalColour.FAIL}Error in CPP function:{TerminalColour.ENDC} {line}"
                )
    except FileNotFoundError:
        pass


def build_library(build_flags: dict, library: str = "libia"):
    """Library is built in the csrc directory using the Makefile instructions.
    - Prints results
    - Throws errors is build was not succesful
    """

    p = subprocess.run(
        f"cd csrc && make --no-print-directory {library}",
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={**os.environ, **build_flags},
    )
    print(p.stdout)
    if p.returncode != 0:
        print(p.stderr)
        raise RuntimeError("Failed building of library! See error message above.")
