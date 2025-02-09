r"""
Defines useful plotting utilities related to health policy.

Notes
-----
#.  Written by David C. Stauffer in March 2015.
#.  Separated into plot_health.py from plotting.py by David C. Stauffer in May 2020.
#.  Moved to consolidated plotting submodule by David C. Stauffer in July 2020.
"""

# %% Imports
from __future__ import annotations

import doctest
from typing import TYPE_CHECKING
import unittest
import warnings

from dstauffman import bins_to_str_ranges, Frozen, get_unit_conversion, HAVE_MPL, HAVE_NUMPY, rms
from dstauffman.plotting.plotting import Opts, setup_plots
from dstauffman.plotting.support import (
    ColorMap,
    DEFAULT_COLORMAP,
    disp_xlimits,
    ignore_plot_data,
    plot_second_units_wrapper,
    show_zero_ylim,
    whitten,
    z_from_ci,
)

if HAVE_MPL:
    from matplotlib.axes import Axes
    from matplotlib.colors import Colormap, ListedColormap
    from matplotlib.figure import Figure
    import matplotlib.pyplot as plt
    from matplotlib.ticker import StrMethodFormatter
if HAVE_NUMPY:
    import numpy as np

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

    _CM = str | Colormap | ListedColormap | ColorMap
    _I = NDArray[np.int_]
    _M = NDArray[np.floating]  # 2D
    _N = NDArray[np.floating]


# %% Classes - TruthPlotter
class TruthPlotter(Frozen):
    r"""
    Class wrapper for the different types of truth data to include on plots.

    Examples
    --------
    >>> from dstauffman.plotting import TruthPlotter
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> fig = plt.figure()
    >>> fig.canvas.manager.set_window_title("Figure Title")
    >>> x = np.arange(0, 10, 0.1)
    >>> y = np.sin(x)
    >>> truth = TruthPlotter(x, y+0.01, lo=y, hi=y+0.03)
    >>> ax = fig.add_subplot(111)
    >>> _ = ax.plot(x, y, label="data")
    >>> truth.plot_truth(ax)
    >>> _ = ax.legend(loc="best")

    >>> plt.show(block=False) # doctest: +SKIP

    Close plot
    >>> plt.close(fig)

    """

    def __init__(
        self,
        time: _N | None = None,
        data: _M | None = None,
        *,
        lo: _N | None = None,
        hi: _N | None = None,
        type_: str = "normal",
        name: str = "Observed",
    ):
        self.time = time
        self.data = None
        self.type_ = type_  # from {"normal", "errorbar"}
        self.data_lo = lo
        self.data_hi = hi
        self.name = name
        # TODO: keep this old API? (otherwise just: self.data = data)
        if data is not None:
            if data.ndim == 1:
                self.data = data
            elif data.shape[1] == 1:
                self.data = data[:, 0]
            elif data.shape[1] == 3:
                self.data = data[:, 1]
                self.data_lo = data[:, 0]
                self.data_hi = data[:, 2]
            else:
                raise ValueError(f"Bad shape for data of {data.shape}.")

    @property
    def is_null(self) -> bool:
        r"""Determine if there is no truth to plot, and thus nothing is done."""
        return self.data is None and self.data_lo is None and self.data_hi is None

    @staticmethod
    def get_data(data: _M | None, scale: float = 1.0, ix: int | None = None) -> _M | None:
        r"""Scale and index the data, returning None if it is not there."""
        if data is None:
            return data
        if ix is None:
            return scale * data
        return scale * data[:, ix]

    def plot_truth(  # noqa: C901
        self, ax: Axes, scale: float = 1.0, ix: int | None = None, *, hold_xlim: bool = True, hold_ylim: bool = False
    ) -> None:
        r"""Add the information in the TruthPlotter instance to the given axes, with the optional scale."""
        # check for null case
        if self.is_null:
            return
        # get original limits
        x_lim = ax.get_xbound()
        y_lim = ax.get_ybound()
        # plot the new data
        this_data = self.get_data(self.data, scale, ix)
        if this_data is not None and not np.all(np.isnan(this_data)):
            ax.plot(self.time, this_data, "k.-", linewidth=2, zorder=8, label=self.name)  # type: ignore[arg-type]
        if self.type_ == "normal":
            this_data = self.get_data(self.data_lo, scale, ix)
            if this_data is not None and not np.all(np.isnan(this_data)):
                ax.plot(self.time, this_data, ".-", color="0.5", linewidth=2, zorder=6)  # type: ignore[arg-type]
            this_data = self.get_data(self.data_hi, scale, ix)
            if self.data_hi is not None and not np.all(np.isnan(this_data)):  # type: ignore[arg-type]
                ax.plot(self.time, this_data, ".-", color="0.5", linewidth=2, zorder=6)  # type: ignore[arg-type]
        elif self.type_ == "errorbar":
            if self.data_lo is not None and self.data_hi is not None:
                if ix is None:
                    yerr = np.vstack((self.data - self.data_lo, self.data_hi - self.data))  # type: ignore[operator]
                    ax.errorbar(
                        self.time, scale * self.data, scale * yerr, linestyle="None", marker="None", ecolor="c", zorder=6  # type: ignore[arg-type, operator]
                    )
                else:
                    yerr = np.vstack((self.data[:, ix] - self.data_lo[:, ix], self.data_hi[:, ix] - self.data[:, ix])).T  # type: ignore[index]
                    ax.errorbar(
                        self.time,  # type: ignore[arg-type]
                        scale * self.data[:, ix],  # type: ignore[index]
                        scale * yerr[:, ix],
                        linestyle="None",
                        marker="None",
                        ecolor="c",
                        zorder=6,
                    )
        else:
            raise ValueError(f'Unexpected value for type_ of "{self.type_}".')
        # potentially restore the original limits, since they might have been changed by the truth data
        if hold_xlim and x_lim != ax.get_xbound():
            ax.set_xbound(*x_lim)
        if hold_ylim and y_lim != ax.get_ybound():
            ax.set_ybound(*y_lim)


# %% Functions - plot_health_time_history
def plot_health_time_history(  # noqa: C901
    description: str,
    time: ArrayLike,
    data: ArrayLike | None,
    *,
    units: str = "",
    opts: Opts | None = None,
    legend: list[str] | None = None,
    second_units: str | int | float | tuple[str, float] | None = None,
    ignore_empties: bool = False,
    data_lo: _N | None = None,
    data_hi: _N | None = None,
    colormap: _CM | None = None,
) -> Figure | None:
    r"""
    Plot multiple metrics over time.

    Parameters
    ----------
    description : str
        Name to label on the plots
    time : 1D ndarray
        time history
    data : 1D, 2D or 3D ndarray
        data for corresponding time history, time is first dimension, last dimension is bin
        middle dimension if 3D is the cycle
    units : str, optional
        units of the data to be displayed on the plot
    opts : class Opts, optional
        plotting options
    legend : list of str, optional
        Names to use for each channel of data
    second_units : str or tuple of (str, float), optional
        Name and conversion factor to use for scaling data to a second Y axis and in legend
    ignore_empties : bool, optional
        Removes any entries from the plot and legend that contain only zeros or only NaNs
    data_lo : same as data
        Lower confidence bound on data, plot if not None
    data_hi : same as data
        Upper confidence bound on data, plot if not None
    colormap : str or matplotlib.colors.Colormap, optional
        Name of colormap to use, if specified, overrides the opts.colormap

    Returns
    -------
    fig : object
        figure handle, if None, no figure was created

    Notes
    -----
    #.  Written by David C. Stauffer in September 2015.
    #.  Updated by David C. Stauffer in October 2017 to do comparsions of multiple runs.

    Examples
    --------
    >>> from dstauffman.plotting import plot_health_time_history
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> description = "Random Data"
    >>> time  = np.arange(0, 5, 1./12) + 2000
    >>> data  = np.random.rand(len(time), 5).cumsum(axis=1)
    >>> data  = 10 * data / np.expand_dims(data[:, -1], axis=1)
    >>> fig   = plot_health_time_history(description, time, data)

    Close plot
    >>> plt.close(fig)

    """
    # force inputs to be ndarrays
    time = np.atleast_1d(np.asanyarray(time))
    data = np.asanyarray(data)

    # check optional inputs
    if opts is None:
        opts = Opts()
    if colormap is None:
        if opts.colormap is None:
            colormap = DEFAULT_COLORMAP
        else:
            colormap = opts.colormap
    legend_loc = opts.leg_spot
    show_zero = opts.show_zero
    time_units = opts.time_base
    unit_text = " [" + units + "]" if units else ""
    (new_units, unit_mult) = get_unit_conversion(second_units, units)

    # check for valid data
    if ignore_plot_data(data, ignore_empties):
        print(" " + description + " plot skipped due to missing data.")
        return None
    assert time.ndim == 1, "Time must be a 1D array."  # type: ignore[union-attr]

    # ensure that data is at least 2D
    if data.ndim == 0:
        data = np.atleast_2d(data)
    elif data.ndim == 1:
        data = data[:, np.newaxis]  # forces to grow in second dimension, instead of first  # type: ignore[union-attr]

    # get shape information
    if data.ndim == 2:  # type: ignore[union-attr]
        normal = True
        num_loops = 1
        num_bins = data.shape[1]  # type: ignore[union-attr]
        names = [""]
    elif data.ndim == 3:  # type: ignore[union-attr]
        assert len(opts.names) == data.shape[1], "Names must match the number of channels is the 3rd axis of data."  # type: ignore[union-attr]
        normal = False
        num_loops = data.shape[1]  # type: ignore[union-attr]
        num_bins = data.shape[2]  # type: ignore[union-attr]
        names = opts.names
    else:
        assert False, "Data must be 0D to 3D array."
    assert time.shape[0] == data.shape[0], f"Time and data must be the same length. Current {time.shape=} and {data.shape=}"  # type: ignore[union-attr]
    if legend is not None:
        assert len(legend) == num_bins, "Number of data channels does not match the legend."
    else:
        legend = [f"Channel {i + 1}" for i in range(num_bins)]

    # process other inputs
    this_title = description + " vs. Time"

    # get colormap based on high and low limits
    cm = ColorMap(colormap, num_colors=num_bins)

    # plot data
    fig = plt.figure()
    assert (manager := fig.canvas.manager) is not None
    manager.set_window_title(description)
    ax = fig.add_subplot(111)
    cm.set_colors(ax)
    for i in range(num_bins):
        for j in range(num_loops):
            this_name = names[j] + " - " if names[j] else ""
            if normal:
                this_data = data[:, i]  # type: ignore[call-overload, index]
                this_data_lo = data_lo[:, i] if data_lo is not None else None
                this_data_hi = data_hi[:, i] if data_hi is not None else None
            else:
                this_data = data[:, j, i]  # type: ignore[call-overload, index]
                this_data_lo = data_lo[:, j, i] if data_lo is not None else None
                this_data_hi = data_hi[:, j, i] if data_hi is not None else None
            if not ignore_plot_data(this_data, ignore_empties):  # type: ignore[arg-type]
                color_dt = j * 0.5 / num_loops
                if j // 2:
                    this_color = whitten(cm.get_color(i), white=(0, 0, 0, 1), dt=color_dt)
                else:
                    this_color = whitten(cm.get_color(i), white=(1, 1, 1, 1), dt=color_dt)
                ax.plot(time, this_data, ".-", label=this_name + legend[i], color=this_color, zorder=10)
                if this_data_lo is not None:
                    ax.plot(time, this_data_lo, "o:", markersize=2, label="", color=whitten(cm.get_color(i)), zorder=6)
                if this_data_hi is not None:
                    ax.plot(time, this_data_hi, "o:", markersize=2, label="", color=whitten(cm.get_color(i)), zorder=6)

    # add labels and legends
    ax.set_xlabel("Time [" + time_units + "]")
    ax.set_ylabel(description + unit_text)
    ax.set_title(this_title)
    ax.legend(loc=legend_loc)
    ax.grid(True)
    # set the display period
    disp_xlimits(ax, xmin=opts.disp_xmin, xmax=opts.disp_xmax)
    # optionally force zero to be plotted
    if show_zero:
        show_zero_ylim(ax)
    # set years to always be whole numbers on the ticks
    if time_units == "year" and (np.max(time) - np.min(time)) >= 4:
        ax.xaxis.set_major_formatter(StrMethodFormatter("{x:.0f}"))

    # optionally add second Y axis
    plot_second_units_wrapper(ax, (new_units, unit_mult))

    # setup plots
    setup_plots(fig, opts)
    return fig


# %% Functions - plot_health_monte_carlo
def plot_health_monte_carlo(  # noqa: C901
    description: str,
    time: ArrayLike,
    data: ArrayLike,
    *,
    units: str = "",
    opts: Opts | None = None,
    plot_indiv: bool = True,
    truth: TruthPlotter | None = None,
    plot_as_diffs: bool = False,
    second_units: str | int | float | tuple[str, float] | None = None,
    plot_sigmas: float = 1.0,
    plot_confidence: float = 0.0,
    colormap: _CM | None = None,
) -> Figure:
    r"""
    Plot the given data channel versus time, with a generic label argument.

    Parameters
    ----------
    time : array_like
        time history
    data : array_like
        data for corresponding time history
    label : str
        generic text to put on the plot title and figure name
    units : str, optional
        units of the data to be displayed on the plot
    opts : class Opts, optional
        plotting options
    plot_indiv : bool, optional
        Plot the individual cycles, default is true
    truth : TruthPlotter, optional
        Truth instance for adding to the plot
    plot_as_diffs : bool, optional, default is False
        Plot each entry in results against the other ones, default is False
    second_units : str or tuple of (str, float), optional
        Name and conversion factor to use for scaling data to a second Y axis and in legend
    plot_sigmas : numeric, optional
        If value converts to true as bool, then plot the sigma values of the given value
    plot_confidence : numeric, optional
        If value converts to true as bool, then plot the confidence intervals of the given value
    colormap : str or matplotlib.colors.Colormap, optional
        Name of colormap to use, if specified, overrides the opts.colormap

    Returns
    -------
    fig : object
        figure handle

    Notes
    -----
    #.  Written by David C. Stauffer in March 2015.
    #.  Updated by David C. Stauffer in December 2015 to include an optional secondary Y axis.
    #.  Updated by David C. Stauffer in October 2016 to use the new TruthPlotter class.
    #.  Updated by David C. Stauffer in June 2017 to put some basic stuff in Opts instead of kwargs.
    #.  Updated by David C. Stauffer in October 2017 to include one sigma and 95% confidence intervals.
    #.  If ndim == 2, then dimension 0 is time and dimension 1 is the number of runs.

    Examples
    --------
    >>> from dstauffman.plotting import plot_health_monte_carlo
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> description = "Sin"
    >>> time = np.arange(0, 10, 0.1)
    >>> data = np.sin(time)
    >>> units = "population"
    >>> fig = plot_health_monte_carlo(description, time, data, units=units)

    Close plot
    >>> plt.close(fig)

    """
    # force inputs to be ndarrays
    time = np.asanyarray(time)
    data = np.asanyarray(data)

    # check optional inputs
    if opts is None:
        opts = Opts()
    if colormap is None:
        if opts.colormap is None:
            colormap = DEFAULT_COLORMAP
        else:
            colormap = opts.colormap
    rms_in_legend = opts.show_rms
    legend_loc = opts.leg_spot
    show_zero = opts.show_zero
    time_units = opts.time_base
    show_legend = rms_in_legend or plot_as_diffs or (truth is not None and not truth.is_null)
    unit_text = " [" + units + "]" if units else ""
    (new_units, unit_mult) = get_unit_conversion(second_units, units)

    # ensure that data is at least 2D
    if data.ndim == 0:
        data = np.atleast_2d(data)
    elif data.ndim == 1:
        data = data[:, np.newaxis]  # forces to grow in second dimension, instead of first
    elif data.ndim == 2:
        pass
    else:
        raise ValueError(
            "Unexpected number of dimensions in data. Monte carlo can currently only "
            + "handle single channels with multiple runs. Split the channels apart into"
            + "separate plots if desired."
        )

    # get number of different series
    num_series = data.shape[1]  # type: ignore[union-attr]

    if plot_as_diffs:
        # build colormap
        cm = ColorMap(colormap, num_colors=data.shape[1])  # type: ignore[union-attr]
        # calculate RMS
        if rms_in_legend:
            rms_data = np.atleast_1d(rms(unit_mult * data, axis=0, ignore_nans=True))  # type: ignore[operator]
    else:
        # calculate the mean and std of data, while disabling warnings for time points that are all NaNs
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Mean of empty slice")
            warnings.filterwarnings("ignore", message="Degrees of freedom <= 0 for slice.")
            mean = np.nanmean(data, axis=1)  # type: ignore[arg-type]
            std = np.nanstd(data, axis=1)  # type: ignore[arg-type]

        # calculate an RMS
        if rms_in_legend:
            rms_data = rms(unit_mult * mean, axis=0, ignore_nans=True)  # type: ignore[assignment]

    # alias the title
    this_title = description + " vs. Time"
    # create the figure and set the title
    fig = plt.figure()
    assert (manager := fig.canvas.manager) is not None
    manager.set_window_title(this_title)
    # add an axis and plot the data
    ax = fig.add_subplot(111)
    if plot_as_diffs:
        cm.set_colors(ax)
        for ix in range(data.shape[1]):  # type: ignore[union-attr]
            this_label = opts.get_names(ix)
            if not this_label:
                this_label = f"Series {ix + 1}"
            if rms_in_legend:
                this_label += f" (RMS: {rms_data[ix]:.2f})"
            ax.plot(time, data[:, ix], ".-", linewidth=2, zorder=10, label=this_label)  # type: ignore[call-overload, index]
    else:
        this_label = opts.get_names(0) + description
        if rms_in_legend:
            this_label += f" (RMS: {rms_data:.2f})"
        ax.plot(time, mean, ".-", linewidth=2, color="#0000cd", zorder=10, label=this_label)
        if plot_sigmas and num_series > 1:
            sigma_label = f"$\\pm {plot_sigmas}\\sigma$"
            ax.plot(time, mean + plot_sigmas * std, ".-", markersize=2, color="#20b2aa", zorder=6, label=sigma_label)
            ax.plot(time, mean - plot_sigmas * std, ".-", markersize=2, color="#20b2aa", zorder=6)
        if plot_confidence and num_series > 1:
            conf_label = f"{100 * plot_confidence}% C.I."
            conf_z = z_from_ci(plot_confidence)
            conf_std = conf_z * std / np.sqrt(num_series)
            ax.plot(time, mean + conf_std, ".-", markersize=2, color="#2e8b57", zorder=7, label=conf_label)
            ax.plot(time, mean - conf_std, ".-", markersize=2, color="#2e8b57", zorder=7)
        # inidividual line plots
        if plot_indiv and data.ndim > 1:  # type: ignore[union-attr]
            for ix in range(num_series):
                ax.plot(time, data[:, ix], color="0.75", zorder=1)  # type: ignore[call-overload, index]
    # optionally plot truth (without changing the axis limits)
    if truth is not None:
        truth.plot_truth(ax)
    # add labels and legends
    ax.set_xlabel("Time [" + time_units + "]")
    ax.set_ylabel(description + unit_text)
    ax.set_title(this_title)
    if show_legend:
        ax.legend(loc=legend_loc)
    # show a grid
    ax.grid(True)
    # optionally force zero to be plotted
    if show_zero and min(ax.get_ylim()) > 0:
        ax.set_ylim(bottom=0)
    # set years to always be whole numbers on the ticks
    if time_units == "year" and np.any(time) and (np.max(time) - np.min(time)) >= 4:
        ax.xaxis.set_major_formatter(StrMethodFormatter("{x:.0f}"))
    # optionally add second Y axis
    plot_second_units_wrapper(ax, (new_units, unit_mult))
    # Setup plots
    setup_plots(fig, opts)
    return fig


# %% Functions - plot_icer
def plot_icer(
    qaly: _N,
    cost: _N,
    ix_front: int,
    *,
    baseline: int | None = None,
    names: list[str] | None = None,
    opts: Opts | None = None,
) -> Figure:
    r"""Plot the icer results."""
    # check optional inputs
    if opts is None:
        opts = Opts()
    # create a figure and axis
    fig = plt.figure()
    title = "Cost Benefit Frontier"
    assert (manager := fig.canvas.manager) is not None
    manager.set_window_title(title)
    ax = fig.add_subplot(111)
    # plot the data
    ax.plot(qaly, cost, "ko", label="strategies")
    ax.plot(qaly[ix_front], cost[ix_front], "r.", markersize=20, label="frontier")
    # get axis limits before (0,0) point is added
    lim = ax.axis()
    # add ICER lines
    if baseline is None:
        ax.plot(np.hstack((0, qaly[ix_front])), np.hstack((0, cost[ix_front])), "r-", label="ICERs")
    else:
        ax.plot(np.hstack((0, qaly[ix_front[0]])), np.hstack((0, cost[ix_front[0]])), "r:")  # type: ignore[arg-type, index]
        ax.plot(np.hstack((qaly[baseline], qaly[ix_front])), np.hstack((cost[baseline], cost[ix_front])), "r-", label="ICERs")
    # Label each point
    dy = (lim[3] - lim[2]) / 100
    for i in range(cost.size):
        assert names is not None, "You must give a list of names."
        ax.annotate(
            names[i],
            xy=(qaly[i], cost[i] + dy),
            xycoords="data",
            horizontalalignment="center",
            verticalalignment="bottom",
            fontsize=12,
        )
    # add some labels and such
    ax.set_title(title)
    ax.set_xlabel("Benefits")
    ax.set_ylabel("Costs")
    ax.legend(loc="upper left")
    ax.grid(True)
    # reset limits with including (0,0) point in case it skews everything too much
    ax.axis(lim)
    # add standard plotting features
    setup_plots(fig, opts)
    return fig


# %% Functions - plot_population_pyramid
def plot_population_pyramid(
    age_bins: _I | _N | list[int],
    male_per: _I | _N | list[int],
    fmal_per: _I | _N | list[int],
    description: str = "Population Pyramid",
    *,
    opts: Opts | None = None,
    name1: str = "Male",
    name2: str = "Female",
    color1: str | tuple[float, ...] = "xkcd:blue",
    color2: str | tuple[float, ...] = "xkcd:red",
) -> Figure:
    r"""
    Plot the standard population pyramid.

    Parameters
    ----------
    age_bins : (N+1,) array_like of float/ints
        Age boundaries to plot
    male_per : (N,) array_like of int
        Male population percentage in each bin
    fmal_per : (N,) array_like of int
        Female population percentage in each bin
    description : str, optional, default is "Population Pyramid"
        Title for the plot
    opts : class Opts, optional
        Plotting options
    name1 : str, optional
        Name for data source 1
    name2 : str, optional
        Name for data source 2
    color1 : str or valid color tuple, optional
        Color for data source 1
    color2 : str or valid color tuple, optional
        Color for data source 2

    Returns
    -------
    fig : object
        figure handle

    Notes
    -----
    #.  Written by David C. Stauffer in April 2017.

    References
    ----------
    .. [1]  https://en.wikipedia.org/wiki/Population_pyramid

    Examples
    --------
    >>> from dstauffman.plotting import plot_population_pyramid
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> age_bins = np.array([  0,   5,  10,  15,  20, 1000], dtype=int)
    >>> male_per = np.array([500, 400, 300, 200, 100]) / 3000
    >>> fmal_per = np.array([450, 375, 325, 225, 125]) / 3000
    >>> fig      = plot_population_pyramid(age_bins, male_per, fmal_per)

    Close figure
    >>> plt.close(fig)

    """
    # hard-coded values
    scale = 100

    # check optional inputs
    if opts is None:
        opts = Opts()
    legend_loc = opts.leg_spot

    # convert data to percentages
    num_pts = len(age_bins) - 1
    y_values = np.arange(num_pts) if HAVE_NUMPY else list(range(num_pts))
    y_labels = bins_to_str_ranges(age_bins, dt=1, cutoff=200)

    # create the figure and axis and set the title
    fig = plt.figure()
    assert (manager := fig.canvas.manager) is not None
    manager.set_window_title(description)
    ax = fig.add_subplot(111)

    # plot bars
    ax.barh(y_values, -scale * male_per, 0.95, color=color1, label=name1)
    ax.barh(y_values, +scale * fmal_per, 0.95, color=color2, label=name2)

    # make sure plot is symmetric about zero
    xlim = max(abs(x) for x in ax.get_xlim())
    ax.set_xlim(-xlim, xlim)

    # add labels
    ax.set_xlabel("Population [%]")
    ax.set_ylabel("Age [years]")
    ax.set_title(description)
    ax.set_yticks(y_values)
    ax.set_yticklabels(y_labels)
    ax.set_xticks(ax.get_xticks().tolist())
    ax.set_xticklabels(np.abs(ax.get_xticks()))
    ax.legend(loc=legend_loc)

    # Setup plots
    setup_plots(fig, opts)

    return fig


# %% Unit test
if __name__ == "__main__":
    plt.ioff()
    unittest.main(module="dstauffman.tests.test_plotting_health", exit=False)
    doctest.testmod(verbose=False)
