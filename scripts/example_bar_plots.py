"""Earth Plotting Examples."""

# %% Imports
import numpy as np

import dstauffman as dcs
import dstauffman.plotting as plot

# %% Script
if __name__ == "__main__":
    # Create data
    time = np.datetime64("2024-12-16 23:50:00", dcs.NP_DATETIME_UNITS) + np.arange(11.0) * dcs.NP_ONE_MINUTE
    # fmt: off
    data = np.array([
        [0.50, 0.45, 0.55, np.nan, 0.35, 0.36, np.nan, 0.4, np.nan, 0.5, np.nan],
        [0.50, 0.55, 0.45, np.nan, 0.65, 0.64, np.nan, 0.6, np.nan, 0.5, 0.4],
    ])
    # fmt: on
    elements = ("A", "B")

    # create Opts
    opts = plot.Opts()

    # plot the time history of the breakdown, with gaps
    plot.plot_bar_breakdown("Breakdown", time, data, opts=opts, elements=elements, time_units="numpy")
