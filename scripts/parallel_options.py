"""Logging when parallel tests."""  # pylint: disable=redefined-outer-name

# %% Imports
from itertools import repeat
import logging
from time import sleep

import slog as lg

import dstauffman as dcs

# %% Globals
logger = logging.getLogger(__name__)


# %% Functions
def func(num: int, pause) -> dict[str, int]:
    """Long-running function that sometimes errors."""
    if num in {5, 15}:
        raise RuntimeError(f"Bad things happened at {num}.")
    out = {"time": 1, "data": num}
    logger.log(lg.LogLevel.L5, 'Ran "%i"', num)
    sleep(pause)
    return out


# %% Script
if __name__ == "__main__":
    lg.activate_logging(lg.LogLevel.L8)
    num_pts = 5
    pause = 0.5

    # Normal run
    args = zip(range(num_pts), repeat(pause, num_pts))
    results1 = dcs.parfor_wrapper(func, args, use_parfor=False, max_cores=4, ignore_errors=True)
    print(results1)

    # parallel run 1
    args = zip(range(num_pts), repeat(pause, num_pts))  # Note: build args as the generator may get consumed
    results2 = dcs.parfor_wrapper(func, args, use_parfor=True, max_cores=4, ignore_errors=True)
    print(results2)
