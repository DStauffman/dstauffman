"""Earth Plotting Examples."""

# %% Imports
import datetime

import numpy as np

import dstauffman as dcs
import dstauffman.plotting as plot

# %% Script
if __name__ == "__main__":
    # Settings
    use_datetime = True
    date_zero = datetime.datetime(2024, 12, 16, 23, 50, 0)

    # Create data
    if use_datetime:  # noqa: SIM108
        time = dcs.convert_datetime_to_np(date_zero) + np.arange(11.0) * dcs.NP_ONE_MINUTE
    else:
        time = 60 * np.arange(11.0)  # type: ignore[assignment]
    # fmt: off
    data = np.array([
        [0.50, 0.45, 0.55, np.nan, 0.35, 0.36, np.nan, 0.4, np.nan, 0.5, np.nan],
        [0.50, 0.55, 0.45, np.nan, 0.65, 0.64, np.nan, 0.6, np.nan, 0.5, 0.4],
    ])
    # fmt: on
    elements = ("A", "B")

    # create Opts
    opts = plot.Opts()
    if use_datetime:
        opts.convert_dates("numpy")
    else:
        opts.time_unit = "min"  # TODO: make this work!?
        opts.date_zero = date_zero

    # plot the time history of the breakdown, with gaps
    plot.plot_bar_breakdown("Breakdown", time, data, opts=opts, elements=elements)
