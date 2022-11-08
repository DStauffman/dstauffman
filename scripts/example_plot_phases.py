"""Script to use the plot_phases directly or threw a wrapper."""

#%% Imports
import datetime

from dstauffman import convert_datetime_to_np, NP_ONE_SECOND
from dstauffman.plotting import make_time_plot, plot_phases, COLOR_LISTS
import matplotlib.pyplot as plt
import numpy as np


#%% Functions
def extra_plotter(time_phases, labels="Times", group_all=True, use_legend=False):  # pylint: disable=redefined-outer-name
    """Wrapper to the plot_phases to be passed into other functions."""
    def _plot_phases(fig, ax):  # pylint: disable=redefined-outer-name,unused-argument
        for this_axes in ax:
            plot_phases(this_axes, time_phases, labels=labels, group_all=group_all, use_legend=use_legend)
    return _plot_phases


#%% Script
if __name__ == "__main__":
    #%% Use directly
    fig = plt.figure()
    fig.canvas.manager.set_window_title("Sine Wave")
    ax = fig.add_subplot(111)
    time = np.arange(101)
    data = np.cos(time / 10)
    _ = ax.plot(time, data, ".-", label="data")
    times = np.array([5, 20, 60, 90])
    # times = np.array([[5, 20, 60, 90], [10, 60, 90, 95]])
    labels = ["Part 1", "Phase 2", "Watch Out", "Final"]
    colors = COLOR_LISTS["quat"]
    plot_phases(ax, times, colors, labels, use_legend=False)
    ax.legend(loc="best")
    plt.show(block=False) # doctest: +SKIP

    #%% Through wrapper
    dates = [
        (datetime.datetime(2022, 11, 11, 11, 0, 0), datetime.datetime(2022, 11, 11, 11, 0, 30)),
        (datetime.datetime(2022, 11, 11, 11, 5, 0), datetime.datetime(2022, 11, 11, 11, 6, 0)),
        (datetime.datetime(2022, 11, 11, 11, 8, 15), datetime.datetime(2022, 11, 11, 11, 8, 30)),
    ]
    time_phases = np.vstack([convert_datetime_to_np([d1, d2]) for d1, d2 in dates]).T
    temp = np.arange(601)
    time2 = convert_datetime_to_np(datetime.datetime(2022, 11, 11, 11, 0, 0, 0)) + temp * NP_ONE_SECOND
    data2 = np.vstack([np.sin(temp / 10), np.cos(temp / 20)])
    extra_plotter = extra_plotter(time_phases, labels="Times", group_all=True, use_legend=False)
    fig2 = make_time_plot(
        "Same Times",
        time2,
        data2,
        time_units="numpy",
        elements=("Sin", "Cos"),
        label_vert_lines=False,
        extra_plotter=extra_plotter,
    )
